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

def find_optimum(sumLogPi=None, nDoc=0, gamma=1.0, alpha=1.0, initeta=None, approx_grad=False, factr=1.0e7, **kwargs):
  ''' Run gradient optimization to estimate best vectors
        rho, omega for specified problem

      Returns
      --------
      rho : 1D array, length K
      omega : 1D array, length K
      f : scalar value of minimization objective
      Info : dict

      Raises
      --------
      ValueError with FAILURE in message if all restarts fail
      Raises
      --------
      ValueError on an overflow, any NaN, or failure to converge
  '''
  sumLogPi = np.squeeze(np.asarray(sumLogPi, dtype=np.float64))
  assert sumLogPi.ndim == 1
  K = sumLogPi.size 
  if initeta is None:
    if nDoc == 0:
      initeta = 0.5 * gamma.copy()
    elif len(gamma) == 1:
      initeta = gamma * np.ones(K)
    else:
      initeta = gamma.copy()
  assert initeta.size == K

  initc = np.log(initeta)
  myFunc = lambda c: objFunc_c(c, sumLogPi=sumLogPi, nDoc=nDoc,
                                   gamma=gamma, alpha=alpha,
                                   approx_grad=approx_grad)
  
  with warnings.catch_warnings():
    warnings.filterwarnings('error', category=RuntimeWarning,
                               message='overflow')
    try:
      chat, fhat, Info = scipy.optimize.fmin_l_bfgs_b(myFunc, initc,
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

  etahat = np.exp(chat)
  Info['init'] = initeta
  Info['objFunc'] = lambda eta: objFunc_eta(eta, sumLogPi=sumLogPi, nDoc=nDoc,
                                   gamma=gamma, alpha=alpha,
                                   approx_grad=approx_grad)
  
  return etahat, fhat, Info

########################################################### Objective
###########################################################  unconstrained
def objFunc_unconstrained(c, approx_grad=False, **kwargs):
  rhoomega, drodc = c2rhoomega(c)
  f, grad = objFunc_constrained(rhoomega, **kwargs)
  if approx_grad:
    return f
  return f, grad * drodc

def c2rhoomega(c):
  K = c.size
  rho = sigmoid(c[:K])
  omega = np.exp(c[K:])
  rhoomega = np.hstack([rho, omega])
  drodc = np.hstack([rho*(1-rho), omega])
  return rhoomega, drodc

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
  Ebeta = _rho2beta(rho)
  kvec = K + 1 - np.arange(1, K+1)

  u1 = rho*omega
  u0 = (1-rho)*omega
  if nDoc > 0:
    B1 = 1 + (1 - u1)/nDoc
    C1 = kvec + (alpha - u0)/nDoc
  else:
    B1 = 1 - u1
    C1 = alpha - u0
    
  digammaomega =  digamma(omega)
  B2 = digamma(u1) - digammaomega
  C2 = digamma(u0) - digammaomega

  elbo = np.sum(gammaln(u1) + gammaln(u0) - gammaln(omega)) \
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
  K = rhoomega.size
  rho = rhoomega[:K]
  omega = rhoomega[-K:]
  return rho, omega, K

def _rho2beta(rho):
  ''' Convert to stick-breaking fractions rho to probability vector beta
      Args
      --------
      rho : K-len vector, rho[k] in interval [0, 1]
      
      Returns
      --------
      beta : K+1-len vector, with positive entries that sum to 1
  '''
  beta = np.hstack([1.0, np.cumprod(1-rho)])
  beta[:-1] *= rho
  return beta

lowTriIDsDict = dict()
def _get_lowTriIDs(K):
  if K in lowTriIDsDict:
    return lowTriIDsDict[K]
  else:
    ltIDs = np.tril_indices(K, -1)
    lowTriIDsDict[K] = ltIDs
    return ltIDs

