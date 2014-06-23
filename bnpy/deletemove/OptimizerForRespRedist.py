import numpy as np
from scipy.special import gammaln, digamma
import warnings
import scipy.optimize

from bnpy.util import EPS

def find_optimum(Rrem, Mrem, Mdel, tau, initEta=None, **kwargs):
  if Rrem.ndim == 1:
    W = 1
    K = Rrem.size
  else:
    W, K = Rrem.shape
  if initEta is None:
    initEta = 1.0/K * np.ones((W,K))
  if initEta.ndim < 2:
    initEta = initEta[np.newaxis,:]
  initc = Eta2C(initEta).flatten()

  ## Define objective function (unconstrained!)
  objFunc = lambda c: objFunc_cflat(c, Rrem, Mrem, Mdel, tau)
  
  ## Run optimization and catch any overflow or NaN issues
  with warnings.catch_warnings():
    warnings.filterwarnings('error', category=RuntimeWarning,
                               message='overflow')
    try:
      bestc, fbest, Info = scipy.optimize.fmin_l_bfgs_b(objFunc, initc,
                                                  disp=None,
                                                  approx_grad=1,
                                                  **kwargs)
    except RuntimeWarning:
      raise ValueError("FAILURE: overflow!" )
    except AssertionError:
      raise ValueError("FAILURE: NaN/Inf detected!")
      
  if Info['warnflag'] > 1:
    raise ValueError("FAILURE: " + Info['task'])

  bestC = np.reshape(bestc, (W, K-1))
  bestEta = C2Eta(bestC)
  return bestEta, -1*fbest, Info


def objFunc_Eta(Eta, R, M, Mdel, tau):
  if Eta.ndim == 1:
    Eta = Eta[:,np.newaxis]
  if R.ndim == 1:
    R = R[:,np.newaxis]
    M = M[:,np.newaxis]
  W, K = R.shape
  f = np.zeros(K)
  for k in xrange(K):
    f[k] = cbeta( M[:,k] + Mdel * Eta[:,k] + tau)
  return -1 * np.sum(f) + np.sum(Eta * R)

def objFunc_cflat(c, R, M, Mdel, tau):
  if M.ndim == 1:
    W = 1
    K = M.size
  else:
    W, K = M.shape
  C = np.reshape(c, (W, K-1))
  return -1 * objFunc_Eta( C2Eta(C), R, M, Mdel, tau)

def cbeta(x):
  return gammaln(np.sum(x)) - np.sum(gammaln(x))


def C2Eta(C):
  ''' Convert WxK-1 matrix C of reals
           to WxK matrix Eta of probs (rows sum to one)
  '''
  VR = sigmoid(C)
  K = VR.shape[1] + 1
  Eta = np.ones((VR.shape[0], K))
  Eta[:, :K-1] = VR
  for k in xrange(1, K):
    Eta[:,k] *= np.prod(1-VR[:,:k], axis=1)
  return Eta

def Eta2C(Eta):
  ''' Convert WxK matrix Eta of probabilities (rows sum to one)
           to WxK-1 matrix C of reals
  '''
  W, K = Eta.shape
  sigC = np.zeros((W, K-1))
  for n in xrange(W):
    sigC[n,:] = _beta2v(Eta[n,:])
  return invsigmoid(sigC)

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