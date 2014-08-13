'''
OptimizerHDPDir.py

CONSTRAINED Optimization Problem
----------
Variables:
Two K-length vectors
* rho = rho[0], rho[1], rho[2], ... rho[K-1]
* omega = omega[0], omega[1], ... omega[K-1]

Objective:
* argmin ELBO(rho, omega)

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

def find_optimum_multiple_tries(sumLogPi=0, nDoc=0, 
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
      rhoomega, f, Info = find_optimum(sumLogPi, nDoc,
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
      msg = str(err)
      if str(err).count('FAILURE') > 0:
        # Catch line search problems
        pass
      elif str(err).count('overflow') > 0:
        nOverflow += 1
      else:
        raise err

  if rhoomega is None:
    if initrho is not None:      
      # Last ditch effort, try different initialization
      return find_optimum_multiple_tries(sumLogPi, nDoc, 
                                gamma=gamma, alpha=alpha,
                                initrho=None, initomega=None,
                                approx_grad=approx_grad, **kwargs)
    else:
      raise ValueError(msg)
  Info['nOverflow'] = nOverflow
  rho, omega, K = _unpack(rhoomega)
  return rho, omega, f, Info


def find_optimum(sumLogPi=0, nDoc=0, gamma=1.0, alpha=1.0,
                 initrho=None, initomega=None, scaleVector=None,
                 approx_grad=False, factr=1.0e5, **kwargs):
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
  if sumLogPi.ndim > 1:
    sumLogPi = np.squeeze(np.asarray(sumLogPi, dtype=np.float64))

  assert sumLogPi.ndim == 1
  K = sumLogPi.size - 1

  ## Determine initial value
  if initrho is None or initrho.min() <= EPS or initrho.max() >= 1 - EPS:
    initrho = create_initrho(K)
  if initomega is None or initomega.min() <= EPS:
    initomega = create_initomega(K, nDoc, gamma)
  if scaleVector is None:
    scaleVector = np.hstack([np.ones(K), np.ones(K)])
  assert initrho.size == K
  assert initomega.size == K
  assert initrho.min() > EPS
  assert initrho.max() < 1.0 - EPS
  assert initomega.min() > EPS
  initrhoomega = np.hstack([initrho, initomega])
  initc = rhoomega2c(initrhoomega, scaleVector=scaleVector)

  ## Define objective function (unconstrained!)
  objArgs = dict(sumLogPi=sumLogPi,
                  nDoc=nDoc, gamma=gamma, alpha=alpha,
                  approx_grad=approx_grad, scaleVector=scaleVector)

  c_objFunc = lambda c: objFunc_unconstrained(c, **objArgs)
  
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
  rhoomega = c2rhoomega(chat, scaleVector=scaleVector, returnSingleVector=1)
  return rhoomega, fhat, Info

def create_initrho(K):
  ''' Make initial guess for rho s.t. E[beta_k] \approx (1-r)/K
      where r is a small amount of remaining/leftover mass 
  '''
  remMass = np.minimum(0.1, 1.0/(K*K))
  # delta = 0, -1 + r, -2 + 2r, ...
  delta = (-1 + remMass) * np.arange(0, K, 1, dtype=np.float)
  rho = (1-remMass)/(K+delta)
  return rho

def create_initomega(K, nDoc, gamma):
  ''' Make initial guess for omega. 
  '''
  return (nDoc / K + gamma) * np.ones(K)

########################################################### Objective
###########################################################  unconstrained
def objFunc_unconstrained(c, scaleVector=None, approx_grad=False, **kwargs):
  rho, omega = c2rhoomega(c, scaleVector)
  rhoomega = np.hstack([rho, omega])
  if approx_grad:
    f = objFunc_constrained(rhoomega, approx_grad=1, **kwargs)
    return f
  else:
    f, grad = objFunc_constrained(rhoomega, approx_grad=0, **kwargs)
    drodc = np.hstack([rho*(1-rho), omega])
    return f, grad * drodc

def c2rhoomega(c, scaleVector=None, returnSingleVector=False):
  ''' Transform unconstrained variable c into constrained rho, omega

      Returns
      --------
      rho : 1D array, size K, entries between [0, 1]
      omega : 1D array, size K, positive entries

      OPTIONAL: may return as one concatenated vector (length 2K)
  '''
  K = c.size/2
  rho = sigmoid(c[:K])
  omega = np.exp(c[K:])
  if scaleVector is not None:
    rho *= scaleVector[:K]
    omega *= scaleVector[K:]
  if returnSingleVector:
    return np.hstack([rho, omega])
  return rho, omega

def rhoomega2c(rhoomega, scaleVector=None):
  K = rhoomega.size/2
  if scaleVector is not None:
    rhoomega = rhoomega / scaleVector
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
                     sumLogPi=0, nDoc=0, gamma=1.0, alpha=1.0,
                     approx_grad=False, **kwargs):
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
  assert not np.any(np.isnan(rhoomega))
  assert not np.any(np.isinf(rhoomega))
  rho, omega, K = _unpack(rhoomega)
  
  g1 = rho * omega
  g0 = (1 - rho) * omega
  digammaomega =  digamma(omega)
  assert not np.any(np.isinf(digammaomega))

  Elogu = digamma(g1) - digammaomega
  Elog1mu = digamma(g0) - digammaomega

  if nDoc > 0:
    scale = nDoc
    ONcoef = 1 + (1.0 - g1)/nDoc
    OFFcoef = kvec(K) + (gamma - g0)/nDoc

    ## Calc local term
    Tvec = sumLogPi/nDoc
    Ebeta = np.hstack([rho, 1.0])
    Ebeta[1:] *= np.cumprod(1-rho)
    elbo_local = alpha * np.inner(Ebeta, Tvec) 

  else:
    scale = 1
    ONcoef = 1 - g1
    OFFcoef = gamma - g0
    elbo_local = 0

  elbo = -1 * c_Beta(g1, g0)/scale \
           + np.inner(ONcoef, Elogu) \
           + np.inner(OFFcoef, Elog1mu) \
           + elbo_local

  if approx_grad:
    return -1.0 * elbo

  ## Gradient computation!  
  trigamma_omega = polygamma(1, omega)
  trigamma_g1 = polygamma(1, g1)
  trigamma_g0 = polygamma(1, g0)
  assert np.all(np.isfinite(trigamma_omega))
  assert np.all(np.isfinite(trigamma_g1))

  gradrho = ONcoef * omega * trigamma_g1 \
            - OFFcoef * omega * trigamma_g0
  gradomega = ONcoef * (rho * trigamma_g1 - trigamma_omega) \
            + OFFcoef * ((1-rho) * trigamma_g0 - trigamma_omega)
  if nDoc > 0:
    Delta = calc_dEbeta_drho(Ebeta, rho, K)
    gradrho += alpha * np.dot(Delta, Tvec)
  grad = np.hstack([gradrho, gradomega])

  return -1.0 * elbo, -1.0 * grad
  
########################################################### Util fcns
###########################################################
def _unpack(rhoomega):
  K = rhoomega.size / 2
  rho = rhoomega[:K]
  omega = rhoomega[-K:]
  return rho, omega, K

def kvec(K):
  ''' Obtain descending vector of [K, K-1, ... 1]

      Returns
      --------
      kvec : 1D array, size K
  '''
  return K + 1 - np.arange(1, K+1)

def c_Beta(g1, g0):
  ''' Calculate cumulant function of the Beta distribution

      Returns
      -------
      c : scalar sum of the cumulants defined by provided parameters
  '''
  return np.sum(gammaln(g1 + g0) - gammaln(g1) - gammaln(g0))

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

def calc_dEbeta_drho(Ebeta, rho, K):
  ''' Calculate partial derivative of Ebeta w.r.t. rho

      Returns
      ---------
      Delta : 2D array, size K x K+1
  '''
  Delta = np.tile(-1 * Ebeta, (K,1))
  Delta /= (1-rho)[:,np.newaxis]
  Delta[_get_diagIDs(K)] *= -1 * (1-rho)/rho
  Delta[_get_lowTriIDs(K)] = 0
  return Delta

lowTriIDsDict = dict()
def _get_lowTriIDs(K):
  if K in lowTriIDsDict:
    return lowTriIDsDict[K]
  else:
    ltIDs = np.tril_indices(K,-1)
    lowTriIDsDict[K] = ltIDs
    return ltIDs

diagIDsDict = dict()
def _get_diagIDs(K):
  if K in diagIDsDict:
    return diagIDsDict[K]
  else:
    diagIDs = np.diag_indices(K)
    diagIDsDict[K] = diagIDs
    return diagIDs

########################################################### Convert rho <-> beta
###########################################################
def rho2beta_active(rho):
  ''' Calculate probability of each active component

      Returns
      --------
      beta : 1D array, size K
             beta[k] := active probability of topic k
             will have positive entries whose sum is <= 1
  '''
  rho = np.asarray(rho, dtype=np.float64)
  beta = rho.copy()
  beta[1:] *= np.cumprod(1 - rho[:-1])
  return beta

def beta2rho(beta, K):
  ''' Returns K-length vector rho of stick-lengths that recreate appearance probs beta
  '''
  beta = np.asarray(beta, dtype=np.float64)
  rho = beta.copy()
  rho /= np.hstack([1.0, 1 - np.cumsum(beta[:-1])])
  if beta.size == K + 1:
    return rho[:-1]
  elif beta.size == K:
    return rho
  else:
    raise ValueError('Provided beta needs to be of length K or K+1')
