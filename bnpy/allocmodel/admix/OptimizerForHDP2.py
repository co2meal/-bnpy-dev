'''
OptimizerForHDP2.py

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

UNCONSTRAINED Problem
----------


'''

import warnings
import numpy as np
import scipy.optimize
import scipy.io
from scipy.special import gammaln, digamma, polygamma
import datetime
import logging
import itertools

Log = logging.getLogger('bnpy')
EPS = 10*np.finfo(float).eps

def find_optimum_multiple_tries(
          sumLogPi=None, nDoc=0, gamma=1.0, alpha=1.0,
          initrho=None, initomega=None,
          approx_grad=False, fList=[1e7, 1e8, 1e10], **kwargs):
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
  K = sumLogPi.size - 1
  if initEta is not None:
    uList = [initEta, None]
  else:
    uList = [None] 

  nOverflow = 0
  u = None
  Info = dict()
  msg = ''
  for trial, myTuple in enumerate(itertools.product(uList, fList)):
    initeta, factr = myTuple
    try:
      eta, feta, Info = estimate_eta(sumLogPi, nDoc, gamma, alpha0,
                                initeta=initeta, factr=factr, 
                                approx_grad=approx_grad)
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

  if eta is None:
    raise ValueError("FAILURE! " + msg)
  Info['nOverflow'] = nOverflow
  return eta, feta, Info      

def find_optimum(
          sumLogPi=None, nDoc=0, gamma=1.0, alpha=1.0,
          initrho=None, initomega=None,
          approx_grad=False, factr=1.0e7, **kwargs):
  ''' Run gradient optimization to estimate best vectors
        rho, omega for specified problem

      Returns
      --------
      rhoomega : 1D array, length 2*K
      f : scalar value of minimization objective
      Info : dict

      Raises
      --------
      ValueError on an overflow, any NaN, or failure to converge
  '''
  sumLogPi = np.squeeze(np.asarray(sumLogPi, dtype=np.float64))
  assert sumLogPi.ndim == 1
  K = sumLogPi.size - 1

  ## Determine initial value
  if initrho is None:
    initrho = create_initrho(K)
  if initomega is None:
    initomega = (nDoc + 1) * np.ones(K)
  assert initrho.size == K
  assert initomega.size == K
  assert initrho.min() > 0.0
  assert initrho.max() < 1.0
  assert initomega.min() > 0.0
  initrhoomega = np.hstack([initrho, initomega])

  initc = rhoomega2c(initrhoomega)

  ## Define objective function (unconstrained!)
  objFunc = lambda c: objFunc_unconstrained(c, 
                              sumLogPi=sumLogPi, nDoc=nDoc, 
                              gamma=gamma, alpha=alpha,
                              approx_grad=approx_grad)
  
  ## Run optimization and catch any overflow or NaN issues
  with warnings.catch_warnings():
    warnings.filterwarnings('error', category=RuntimeWarning,
                               message='overflow')
    try:
      chat, fhat, Info = scipy.optimize.fmin_l_bfgs_b(objFunc, initc,
                                                  disp=None,
                                                  approx_grad=approx_grad,
                                                  factr=factr,
                                                  **kwargs)
    except RuntimeWarning:
      raise ValueError("FAILURE: overflow!" )
    except AssertionError:
      raise ValueError("FAILURE: NaN found!")
      
  if Info['warnflag'] > 1:
    raise ValueError("FAILURE: " + Info['task'])

  Info['init'] = initrhoomega
  Info['objFunc'] = lambda rhoomega: objFunc_constrained(rhoomega,
                              sumLogPi=sumLogPi, nDoc=nDoc, 
                              gamma=gamma, alpha=alpha,
                              approx_grad=False)
  rhoomega = c2rhoomega(chat, doGrad=False)
  return rhoomega, fhat, Info

def create_initrho(K):
  REM = 1.0/(K*K*K)
  beta = np.ones(K+1)
  beta[:-1] = (1.0-REM)/K
  beta[-1] = REM  
  return _beta2v(beta)

########################################################### Objective
###########################################################  unconstrained
def objFunc_unconstrained(c, approx_grad=False, **kwargs):
  rhoomega, drodc = c2rhoomega(c, doGrad=True)
  f, grad = objFunc_constrained(rhoomega, **kwargs)
  if approx_grad:
    return f
  return f, grad * drodc

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
  ''' sigmoid(c) = 1./(1+exp(-c))
  '''
  return 1.0/(1.0 + np.exp(-c))

def invsigmoid(v):
  ''' Returns the inverse of the sigmoid function
      v = sigmoid(invsigmoid(v))

      Args
      --------
      v : positive vector with entries 0 < v < 1
  '''
  assert np.all( v <= 1-EPS)
  assert np.all( v >= EPS)
  return -np.log((1.0/v - 1))

########################################################### Objective
###########################################################  constrained
def objFunc_constrained(rhoomega,
                     sumLogPi=0, nDoc=0, gamma=1.0, alpha=1.0,
                     approx_grad=False):
  ''' Returns scalar value of constrained objective function
        and the gradient

      Args
      -------
      rhoomega := 1D array, size 2*K

      Returns
      -------
      f := -1 * L(rhoomega), 
           where L is ELBO objective function (log posterior prob)
      g := gradient of f
  '''
  assert not np.any(np.isnan(rhoomega))
  rho, omega, K = _unpack(rhoomega)
  Ebeta = _v2beta(rho)

  u1 = rho * omega
  u0 = (1 - rho) * omega


  logc = np.sum(gammaln(u1) + gammaln(u0) - gammaln(omega))
  if nDoc > 0:
    logc = logc/nDoc
    B1 = 1 + (1.0 - u1)/nDoc
    kvec = K + 1 - np.arange(1, K+1)
    C1 = kvec + (alpha - u0)/nDoc
  else:
    B1 = 1 - u1
    C1 = alpha - u0
    
  digammaomega =  digamma(omega)
  B2 = digamma(u1) - digammaomega
  C2 = digamma(u0) - digammaomega

  elbo = logc \
         + np.inner( B1, B2) \
         + np.inner( C1, C2)
  if nDoc > 0:
    elbo += gamma * np.inner(Ebeta, sumLogPi/nDoc)

  if approx_grad:
    return -1.0 * elbo

  psiP_u1 = polygamma(1, u1)
  psiP_u0 = polygamma(1, u0)
  psiP_omega = polygamma(1, omega)

  gradrho = B1 * omega * psiP_u1 - C1 * omega * psiP_u0
  gradomega = B1 * (   rho * psiP_u1 - psiP_omega) \
            + C1 * ((1-rho)* psiP_u0 - psiP_omega)

  if nDoc > 0:
    # BetaMat : K x K+1 matrix
    #  BetaMat[j,k] = Ebeta[k] / rho[j] or Ebeta[k] / 1-rho[j]
    BetaMat = np.tile(-1*Ebeta, (K,1))
    BetaMat /= (1-rho)[:,np.newaxis]
    diagIDs = np.diag_indices(K)
    BetaMat[diagIDs] /= -1 * rho/(1-rho)
    BetaMat[_get_lowTriIDs(K)] = 0
    gradrho += gamma * np.dot(BetaMat, sumLogPi/nDoc)

  grad = np.hstack([gradrho, gradomega])
  return -1.0 * elbo, -1.0 * grad

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


lowTriIDsDict = dict()
def _get_lowTriIDs(K):
  if K in lowTriIDsDict:
    return lowTriIDsDict[K]
  else:
    ltIDs = np.tril_indices(K, -1)
    lowTriIDsDict[K] = ltIDs
    return ltIDs

