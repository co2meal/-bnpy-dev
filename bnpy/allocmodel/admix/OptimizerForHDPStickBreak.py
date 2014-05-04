'''
OptimizerForHDPStickBreak.py

CONSTRAINED Optimization Problem
----------
Variables:
Two K-length vectors
* rho = rho[0], rho[1], rho[2], ... rho[K-1]
* omega = omega[0], omega[1], ... omega[K-1]

Objective:
* argmin L(rho, omega)

Constraints: 
* rho satisfies: 0 < rho[k] < 1
* omega satisfies: 0 < omega[k]
'''

import warnings
import numpy as np
import scipy.optimize
import scipy.io
from scipy.special import gammaln, digamma, polygamma
import datetime
import logging

Log = logging.getLogger('bnpy')
EPS = 10*np.finfo(float).eps

def find_optimum_multiple_tries(sumLogVd=0, sumLog1mVd=0, nDoc=0, 
                                gamma=1.0, alpha=1.0,
                                initrho=None, initomega=None,
                                approx_grad=False,
                                factrList=[1e5, 1e7, 1e9, 1e10, 1e11],
                                **kwargs):
  ''' Estimate vectors rho and omega via gradient descent,
        gracefully using multiple restarts
        with progressively weaker tolerances until one succeeds

      Returns
      --------
      rho : 1D array, length K
      omega : 1D array, length K
      f : scalar value of minimization objective
      Info : dict

      Raises
      --------
      ValueError with FAILURE in message if all restarts fail
  '''
  rhoomega = None
  Info = dict()
  msg = ''
  nOverflow = 0
  for trial, factr in enumerate(factrList):
    try:
      rhoomega, f, Info = find_optimum(sumLogVd, sumLog1mVd, nDoc,
                                       gamma=gamma, alpha=alpha,
                                       initrho=initrho, initomega=initomega,
                                       factr=factr, approx_grad=approx_grad,
                                       **kwargs)
      Info['nRestarts'] = trial
      Info['factr'] = factr
      Info['msg'] = Info['task']
      del Info['grad']
      del Info['task']
      break
    except ValueError as err:
      if str(err).count('FAILURE') == 0:
        raise err
      msg = str(err)
      if str(err).count('overflow') > 0:
        nOverflow += 1

  if rhoomega is None:
    if initrho is not None:      
      # Last ditch effort, try different initialization
      return find_optimum_multiple_tries(sumLogVd, sumLog1mVd, nDoc, 
                                gamma=gamma, alpha=alpha,
                                initrho=None, initomega=None,
                                approx_grad=approx_grad, **kwargs)
    else:
      raise ValueError(msg)
  Info['nOverflow'] = nOverflow
  rho, omega, K = _unpack(rhoomega)
  return rho, omega, f, Info      

def find_optimum(sumLogVd=0, sumLog1mVd=0, nDoc=0, gamma=1.0, alpha=1.0,
                 initrho=None, initomega=None,
                 approx_grad=False, factr=1.0e7, **kwargs):
  ''' Run gradient optimization to estimate best parameters rho, omega

      Returns
      --------
      rhoomega : 1D array, length 2*K
      f : scalar value of minimization objective
      Info : dict

      Raises
      --------
      ValueError on an overflow, any NaN, or failure to converge
  '''
  sumLogVd = np.squeeze(np.asarray(sumLogVd, dtype=np.float64))
  sumLog1mVd = np.squeeze(np.asarray(sumLog1mVd, dtype=np.float64))

  assert sumLogVd.ndim == 1
  K = sumLogVd.size

  ## Determine initial value
  if initrho is None:
    initrho = create_initrho(K)
  if initomega is None:
    initomega = (nDoc/4 + 1) * np.ones(K)
  assert initrho.size == K
  assert initomega.size == K
  assert initrho.min() > 0.0
  assert initrho.max() < 1.0
  assert initomega.min() > 0.0
  initrhoomega = np.hstack([initrho, initomega])
  initc = rhoomega2c(initrhoomega)

  ## Define objective function (unconstrained!)
  objArgs = dict(sumLogVd=sumLogVd, sumLog1mVd=sumLog1mVd,
                  nDoc=nDoc, gamma=gamma, alpha=alpha,
                  approx_grad=approx_grad)

  c_objFunc = lambda c: objFunc_unconstrained(c, **objArgs)
  ro_objFunc = lambda ro: objFunc_constrained(ro, **objArgs)                            
  
  ## Run optimization and catch any overflow or NaN issues
  with warnings.catch_warnings():
    warnings.filterwarnings('error', category=RuntimeWarning,
                               message='overflow')
    try:
      chat, fhat, Info = scipy.optimize.fmin_l_bfgs_b(c_objFunc, initc,
                                                  disp=None,
                                                  approx_grad=approx_grad,
                                                  factr=factr,
                                                  **kwargs)
    except RuntimeWarning:
      raise ValueError("FAILURE: overflow!" )
    except AssertionError:
      raise ValueError("FAILURE: NaN/Inf detected!")
      
  if Info['warnflag'] > 1:
    raise ValueError("FAILURE: " + Info['task'])

  Info['init'] = initrhoomega
  Info['objFunc'] = ro_objFunc
  rhoomega = c2rhoomega(chat, doGrad=False)
  return rhoomega, fhat, Info

def create_initrho(K):
  rem = 1.0/(K*K)
  beta = (1.0 - rem)/K * np.ones(K+1)
  beta[-1] = rem
  return _beta2v(beta)

########################################################### Objective
###########################################################  unconstrained
def objFunc_unconstrained(c, approx_grad=False, **kwargs):
  rhoomega, drodc = c2rhoomega(c, doGrad=True)
  if approx_grad:
    f = objFunc_constrained(rhoomega, approx_grad=True, **kwargs)
    return f
  f, grad = objFunc_constrained(rhoomega, **kwargs)
  return f, grad

def c2rhoomega(c, doGrad=False):
  K = c.size/2
  rho = sigmoid(c[:K])
  omega = np.exp(c[K:])
  rhoomega = np.hstack([rho, omega])
  if not doGrad:
    return rhoomega
  drodc = np.hstack([rho*(1-rho), omega])
  return rhoomega, drodc

def rhoomega2c( rhoomega):
  K = rhoomega.size/2
  return np.hstack([invsigmoid(rhoomega[:K]), np.log(rhoomega[K:])])

def sigmoid(c):
  ''' Calculates the sigmoid function at provided value (vectorized)
      sigmoid(c) = 1./(1+exp(-c))

      Notes
      -------
      Automatically enforces result away from "boundaries" [0, 1]
      This step is crucial to avoid overflow/NaN problems in optimization
  '''
  v = 1.0/(1.0 + np.exp(-c))
  v = np.minimum(np.maximum(v, EPS), 1-EPS)
  #assert np.all(v >= EPS)
  #assert np.all(v <= 1 - EPS)
  return v

def invsigmoid(v):
  ''' Returns the inverse of the sigmoid function
      v = sigmoid(invsigmoid(v))

      Args
      --------
      v : positive vector with entries 0 < v < 1
  '''
  assert np.max(v) <= 1-EPS
  assert np.min(v) >= EPS
  return -np.log((1.0/v - 1))

########################################################### Objective
###########################################################  constrained
def objFunc_constrained(rhoomega,
                     sumLogVd=0, sumLog1mVd=0, nDoc=0, gamma=1.0, alpha=1.0,
                     approx_grad=False):
  ''' Returns constrained objective function and its gradient

      Args
      -------
      rhoomega := 1D array, size 2*K

      Returns
      -------
      f := -1 * L(rhoomega), 
           where L is ELBO objective function (log posterior prob)
      g := gradient of f
  '''
  rho, omega, K = _unpack(rhoomega)
  assert not np.any(np.isnan(rhoomega))
  assert not np.any(np.isinf(omega))

  u1 = rho * omega
  u0 = (1 - rho) * omega

  gammalnomega = gammaln(omega)
  digammaomega =  digamma(omega)
  assert not np.any(np.isinf(gammalnomega))
  assert not np.any(np.isinf(digammaomega))
  if not approx_grad:
    psiP_omega = polygamma(1, omega)
    assert not np.any(np.isinf(psiP_omega))

  logc = np.sum(gammaln(u1) + gammaln(u0) - gammalnomega)
  if nDoc > 0:
    logc = logc/nDoc
    B1 = 1 + (1.0 - u1)/nDoc
    kvec = K + 1 - np.arange(1, K+1)
    C1 = kvec + (alpha - u0)/nDoc
  else:
    B1 = 1 - u1
    C1 = alpha - u0
    
  B2 = digamma(u1) - digammaomega
  C2 = digamma(u0) - digammaomega

  elbo = logc + np.inner( B1, B2) \
              + np.inner( C1, C2)
  if nDoc > 0:
    rho1m = 1 - rho
    cumprod1mrho = np.ones(K)
    cumprod1mrho[1:] = np.cumprod(rho1m[:-1])
    P = sumLogVd/nDoc
    Q = sumLog1mVd/nDoc
    rPand1mrQ = rho * P + (1-rho) * Q
    elbo += gamma * np.inner(cumprod1mrho, rPand1mrQ) 

  if approx_grad:
    return -1.0 * elbo
  
  psiP_u1 = polygamma(1, u1)
  psiP_u0 = polygamma(1, u0)

  gradrho = B1 * omega * psiP_u1 - C1 * omega * psiP_u0
  gradomega = B1 * (   rho * psiP_u1 - psiP_omega) \
            + C1 * ((1-rho)* psiP_u0 - psiP_omega)

  if nDoc > 0:
    RMat = calc_drho_dcumprod1mrho(cumprod1mrho, rho, K)
    gB = np.dot(RMat, rPand1mrQ) + cumprod1mrho * (P-Q)
    gradrho += gamma * gB
    # Alternate version (slower)
    #Mat1, Mat0 = calc_drho_dU1U0(cumprod1mrho, rho, K)
    #gA = np.dot(Mat1, P) + np.dot(Mat0, Q)

  grad = np.hstack([gradrho, gradomega])
  return -1.0 * elbo, -1.0 * grad
  

###########################################################
###########################################################

def _unpack(rhoomega):
  K = rhoomega.size / 2
  rho = rhoomega[:K]
  omega = rhoomega[-K:]
  return rho, omega, K

def _v2beta(v):
  ''' Convert to stick-breaking fractions v to probability vector beta
      Args
      --------
      v : K-len vector, rho[k] in interval [0, 1]
      
      Returns
      --------
      beta : K+1-len vector, with positive entries that sum to 1
  '''
  beta = np.hstack([1.0, np.cumprod(1-v)])
  beta[:-1] *= v
  return beta


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
  # Force away from edges 0 or 1 for numerical stability  
  v = np.maximum(v,EPS)
  v = np.minimum(v,1-EPS)
  return v


def calc_drho_dU1U0(cumprod1mrho, rho, K):
  ''' Calculate partial derivative of cumprod1mrho w.r.t. rho
      Returns
      ---------
      RMat : 2D array, size K x K
  '''
  RMat = np.tile(-1*cumprod1mrho, (K,1))
  RMat /= (1-rho)[:,np.newaxis]
  RMat[_get_lowTriIDs(K)] = 0
  diagIDs = np.diag_indices(K)
  Mat1 = RMat * rho[np.newaxis,:]
  Mat2 = RMat * (1-rho)[np.newaxis,:]
  Mat1[diagIDs] = cumprod1mrho
  Mat2[diagIDs] = -1 * cumprod1mrho
  return Mat1, Mat2

def calc_drho_dcumprod1mrho(cumprod1mrho, rho, K):
  ''' Calculate partial derivative of cumprod1mrho w.r.t. rho
      Returns
      ---------
      RMat : 2D array, size K x K
  '''
  RMat = np.tile(-1*cumprod1mrho, (K,1))
  RMat /= (1-rho)[:,np.newaxis]
  RMat[_get_lowTriIDs(K)] = 0
  return RMat

lowTriIDsDict = dict()
def _get_lowTriIDs(K):
  if K in lowTriIDsDict:
    return lowTriIDsDict[K]
  else:
    ltIDs = np.tril_indices(K)
    lowTriIDsDict[K] = ltIDs
    return ltIDs

