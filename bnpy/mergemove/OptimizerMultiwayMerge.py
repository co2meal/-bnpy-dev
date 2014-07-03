import numpy as np
from scipy.special import gammaln, digamma
import warnings
import scipy.optimize

from bnpy.util import EPS

def find_optimum(SS, kdel, aFunc, oFunc, initalph=None, **kwargs):
  K = SS.K - 1
  if initalph is None:
    initalph = 1.0/K * np.ones(K)
  initeta = alph2eta(initalph)

  ## Define objective function (unconstrained!)
  objFunc = lambda eta: objFunc_eta(eta, SS, kdel, aFunc, oFunc)
  
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

  return eta2alph(besteta), -1*fbest, Info



def objFunc_alph(alph, SS, kdel, aFunc, oFunc):
  ''' Calc objective func with respect to blend vector alph.

      Returns
      -------
      Negative gain in ELBO, so local minimum alph gives maximum gain.
  '''
  alphx = np.zeros(SS.K)
  alphx[:kdel] = alph[:kdel]
  alphx[kdel+1:] = alph[kdel:]

  Halph = -1 * np.sum((alph+1e-15) * np.log(alph+1e-15))
  elboDelta = aFunc(SS, kdel, alphx) \
              + oFunc(SS, kdel, alphx) \
              + SS.N[kdel] * Halph
  return -1 * elboDelta

def objFunc_eta(eta, *args):
  ''' Calc variational objective for unconstrained variable Eta
  '''
  alph = eta2alph(eta)
  return objFunc_alph(alph, *args)

def eta2alph(eta):
  v = sigmoid(eta)
  c1mv = np.cumprod(1-v)
  alph = np.hstack([v, 1.0])
  alph[1:] *= c1mv
  return alph

def alph2eta(alph):
  eta = invsigmoid(_beta2v(alph))
  return eta

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
