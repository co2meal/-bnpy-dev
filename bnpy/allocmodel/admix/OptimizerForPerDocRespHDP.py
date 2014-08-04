import numpy as np
from scipy.special import gammaln, digamma
import warnings
import scipy.optimize

from bnpy.util import EPS

def find_optimum(Xd, Ld, avec, bvec, initR=None, **kwargs):
  N = Xd.size
  K = bvec.size
  if initR is None:
    initR = 1.0/K * np.ones((N,K))
  initEta = R2eta(initR)
  initeta = initEta.flatten()

  ## Define objective function (unconstrained!)
  objFunc = lambda eta: objFunc_etaflat(eta, Xd, Ld, avec, bvec)
  
  ## Run optimization and catch any overflow or NaN issues
  with warnings.catch_warnings():
    warnings.filterwarnings('error', category=RuntimeWarning,
                               message='overflow')
    try:
      besteta, fbest, Info = scipy.optimize.fmin_l_bfgs_b(objFunc, initeta,
                                                  disp=None,
                                                  approx_grad=1,
                                                  **kwargs)
    except RuntimeWarning:
      raise ValueError("FAILURE: overflow!" )
    except AssertionError:
      raise ValueError("FAILURE: NaN/Inf detected!")
      
  if Info['warnflag'] > 1:
    raise ValueError("FAILURE: " + Info['task'])

  bestEta = np.reshape(besteta, (N, K-1))
  return eta2R(bestEta), -1*fbest, Info


def calcNdkAboveK(Ndk):
  aboveNdk = np.zeros(Ndk.size)
  aboveNdk[:-1] = np.cumsum(Ndk[1:][::-1])[::-1]
  return aboveNdk

def objFunc_Resp(Resp, Xd, Ld, avec, bvec):
  '''
  '''
  XR = Xd[:,np.newaxis] * Resp
  Ndk = np.sum(XR, axis=0)
  ElogpData = np.sum(XR * Ld)
  ElogqZ = np.sum(XR * np.log(Resp+1e-100))

  U1 = Ndk + avec[np.newaxis,:]
  U0 = calcNdkAboveK(Ndk) + bvec[np.newaxis,:]
  ElogqVd = np.sum( gammaln(U1+U0) - gammaln(U1) - gammaln(U0) )
  return ElogpData - ElogqZ - ElogqVd

def objFunc_etaflat(eta, Xd, Ld, avec, bvec):
  K = avec.size
  N = Xd.size
  Eta = np.reshape(eta, (N,K-1))
  return -1 * objFunc_Eta(Eta, Xd, Ld, avec, bvec)

def objFunc_Eta(Eta, Xd, Ld, avec, bvec):
  ''' Calc variational objective for unconstrained variable Eta
  '''
  K = avec.size
  assert Eta.shape[1] == K - 1
  R = eta2R(Eta)
  return objFunc_Resp(R, Xd, Ld, avec, bvec)

def eta2R(Eta):
  VR = sigmoid(Eta)
  K = VR.shape[1] + 1
  R = np.ones((VR.shape[0], K))
  R[:, :K-1] = VR
  for k in xrange(1, K):
    R[:,k] *= np.prod(1-VR[:,:k], axis=1)
  return R

def R2eta(R):
  N,K = R.shape
  eta = np.zeros((N, K-1))
  for n in xrange(N):
    eta[n,:] = _beta2v(R[n,:])
  return invsigmoid(eta)

def sigmoid(Eta):
  V = 1.0 / (1.0 + np.exp(-1*Eta))
  return V

def invsigmoid(V):
  ''' Returns the inverse of the sigmoid function
      v = sigmoid(invsigmoid(v))

      Args
      --------
      v : positive vector with entries 0 < v < 1
  '''
  Eta = -np.log((1.0/V - 1))
  Eta = np.minimum(Eta, 50)
  Eta = np.maximum(Eta, -50)
  return Eta

def _beta2v( beta ):
  ''' Convert probability vector beta to stick-breaking fractions v
      Args
      --------
      beta : K+1-len vector, with positive entries that sum to 1
      
      Returns
      --------
      v : K-len vector, v[k] in interval [0, 1]
  '''
  beta = np.asarray(beta)
  K = beta.size
  v = np.zeros(K-1)
  cb = beta.copy()
  for k in range(K-1):
    cb[k] = 1 - cb[k]/np.prod( cb[:k] )
    v[k] = beta[k]/np.prod( cb[:k] )
  return v