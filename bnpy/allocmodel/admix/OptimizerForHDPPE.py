'''
OptimizerForHDPPE.py

Model Notation
--------
Dirichlet-Multinomial model with K+1 possible components

v    := K-length vector with entries in [0,1]
beta := K+1-length vector with entries in [0,1]
          entries must sum to unity.  sum(beta) = 1.
alpha0 := scalar, alpha0 > 0

Generate stick breaking fractions v 
  v[k] ~ Beta(1, alpha0)
Then deterministically obtain beta
  beta[k] = v[k] prod(1 - v[:k]), k = 1, 2, ... K
  beta[K+1] = prod_k=1^K 1-v[k]
Then draw observed probability vectors
  pi[d] ~ Dirichlet(gamma * beta), for d = 1, 2, ... D

CONSTRAINED Optimization Problem
----------
v* = argmax_v  log p(pi | v) + log p( v ), subject to 0 <= v <= 1

UNCONSTRAINED Problem
----------
c* = argmax_c log p(pi | v) + log p (v), where v = sigmoid(c),  -Inf < c < Inf
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


def find_optimum_multiple_tries(sumLogPi=None, nDoc=0, gamma=1.0, alpha=1.0,
                                initv=None, approx_grad=False,
                                factrList=[1e5, 1e7, 1e9, 1e10, 1e11],
                                **kwargs):
  ''' Find v via gradient descent, with multiple tries until first succeeds.

      Returns
      --------
      v : 1D array, length K
      f : scalar value of minimization objective
      Info : dict

      Raises
      --------
      ValueError with FAILURE in message if all restarts fail
  '''
  v = None
  Info = dict()
  msg = ''
  nOverflow = 0
  for trial, factr in enumerate(factrList):
    try:
      v, f, Info = find_optimum(sumLogPi, nDoc, gamma, alpha,
                                initv=initv,
                                factr=factr, approx_grad=approx_grad)
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

  if v is None: # optimization failed
    if initv is not None:
      # last ditch attempt with different initial values
      return find_optimum_multiple_tries(sumLogPi, nDoc, gamma=gamma, 
                                         alpha=alpha, approx_grad=approx_grad,
                                         **kwargs)
    else:
      raise ValueError(msg)
  Info['nOverflow'] = nOverflow
  return v, f, Info      

def find_optimum(sumLogPi=None, nDoc=0, gamma=1.0, alpha=1.0,
                 initv=None, 
                 approx_grad=False, factr=1.0e7, **kwargs):
  ''' Run gradient optimization to estimate best vector v

      Returns
      --------
      v : 1D array, length K
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
  if initv is None:
    initv = create_initv(K)
  assert initv.size == K
  assert initv.min() > 0.0
  assert initv.max() < 1.0
  initc = v2c(initv)

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

  Info['init'] = initv
  Info['objFunc'] = lambda v: objFunc_constrained(v,
                              sumLogPi=sumLogPi, nDoc=nDoc, 
                              gamma=gamma, alpha=alpha,
                              approx_grad=False)
  v = c2v(chat, doGrad=False)
  return v, fhat, Info

def create_initv(K):
  ''' Create initial guess for vector v, s.t. beta(v) is uniform over topics
  '''
  rem = np.minimum( 0.1, 1.0/(K*K))
  beta = (1.0-rem)/K * np.ones(K+1)
  beta[-1] = rem  
  assert np.allclose( beta.sum(), 1.0)
  return _beta2v(beta)

########################################################### Objective
###########################################################  unconstrained
def objFunc_unconstrained(c, approx_grad=False, **kwargs):
  v, dvdc = c2v(c, doGrad=True)
  if approx_grad:
    f = objFunc_constrained(v, approx_grad=approx_grad, **kwargs)
    return f
  f, grad = objFunc_constrained(v, **kwargs)
  return f, grad * dvdc

def c2v(c, doGrad=False):
  v = sigmoid(c)
  if not doGrad:
    return v
  dvdc = v * (1-v)
  return v, dvdc

def v2c(v):
  return invsigmoid(v)

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
  assert np.all(v <= 1-EPS)
  assert np.all(v >= EPS)
  return -np.log((1.0/v - 1))

########################################################### Objective
###########################################################  constrained
def objFunc_constrained(v,
                     sumLogPi=0, nDoc=0, gamma=1.0, alpha=1.0,
                     approx_grad=False):
  ''' Returns constrained objective function value and its gradient

      Args
      -------
      v := 1D array, size K

      Returns
      -------
      f := -1 * L(v), 
           where L is ELBO objective function (log posterior prob)
      g := gradient of f
  '''
  logpV = (alpha - 1) * np.sum(np.log(1.-v))

  if nDoc > 0:
    beta = _v2beta(v)
    logpPi_const = gammaln(gamma) - np.sum(gammaln(gamma*beta))
    logpPi = np.inner(gamma*beta, sumLogPi/nDoc)
    elbo = logpV / nDoc + logpPi_const + logpPi
  else:
    elbo = logpV

  if approx_grad:
    return -1.0 * elbo
  
  if nDoc > 0:
    K = v.size
    dBdv = np.tile(-1*beta, (K,1))
    dBdv /= (1-v)[:,np.newaxis]
    diagIDs = np.diag_indices(K)
    dBdv[diagIDs] /= -1 * v/(1-v)
    dBdv[_get_lowTriIDs(K)] = 0

    grad_logPi = gamma * (np.dot(dBdv, sumLogPi/nDoc) \
                        - np.dot(dBdv, digamma(gamma * beta)))
    grad = (1 - alpha) / (1-v) / nDoc + grad_logPi
  else:
    grad = (1 - alpha) / (1-v)

  return -1.0 * elbo, -1.0 * grad

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







"""
def estimate_v(sumLogPi=None, nDoc=0, gamma=1.0, alpha0=1.0, initv=None, approx_grad=False, **kwargs):
  ''' Run gradient optimization to estimate best v for specified problem

      Returns
      -------- 
      vhat : K-vector of values, 0 < v < 1
      fofvhat: objective function value at vhat
      Info : dict with info about estimation algorithm
  '''
  sumLogPi = np.squeeze(np.asarray(sumLogPi, dtype=np.float64))
  assert sumLogPi.ndim == 1
  K = sumLogPi.size - 1
  if initv is None:
    initv = 1.0/(1.0+alpha0) * np.ones(K)
  assert initv.size == K

  initc = invsigmoid(initv)
  myFunc = lambda c: objFunc_c(c, sumLogPi, nDoc, gamma, alpha0)
  myGrad = lambda c: objGrad_c(c, sumLogPi, nDoc, gamma, alpha0)
  
  with warnings.catch_warnings():
    warnings.filterwarnings('error', category=RuntimeWarning,
                               message='overflow')
    try:
      chat, fhat, Info = scipy.optimize.fmin_l_bfgs_b(myFunc, initc,
                                                  approx_grad=approx_grad,
                                                  fprime=myGrad, disp=None,
                                                  factr=1e10,
                                                  **kwargs)
    except RuntimeWarning:
      Info = dict(warnflag=2, task='Overflow!')
      chat = initc
      fhat = myFunc(chat)

  if Info['warnflag'] > 1:
    print "******", Info['task']
    raise ValueError("Optimization failed")

  vhat = sigmoid(chat)
  Info['initv'] = initv
  Info['objFunc'] = lambda v: objFunc_v(v, sumLogPi, nDoc, gamma, alpha0)
  Info['gradFunc'] = lambda v: objGrad_v(v, sumLogPi, nDoc, gamma, alpha0)
  return vhat, fhat, Info

########################################################### Objective/gradient
########################################################### in terms of v

def objFunc_v(v, sumLogPi, nDoc, gamma, alpha0):
  ''' Returns scalar value of constrained objective function
      Args
      -------
      v := K-vector of real numbers, subject to 0 < v < 1

      Returns
      -------
      f := -1 * L(v), where L is ELBO objective function (log posterior prob)
  '''
  # log prior
  logpV = (alpha0 - 1) * np.sum(np.log(1.-v))
  # log likelihood
  beta = v2beta(v)
  logpPi_const = gammaln(gamma) - np.sum(gammaln(gamma*beta))
  logpPi = np.inner(gamma*beta - 1, sumLogPi)
  return -1.0 * (nDoc*logpPi_const + logpPi + logpV)

def objGrad_v(v, sumLogPi, nDoc, gamma, alpha0):
  ''' Returns K-vector gradient of the constrained objective
      Args
      -------
      v := K-vector of real numbers, subject to 0 < v < 1

      Returns
      -------
      g := K-vector of real numbers, 
            g[k] = gradient of -1 * L(v) with respect to v[k]
  '''
  K = v.size
  beta = v2beta(v)
  dv_logpV = (1 - alpha0) / (1-v)

  dv_logpPi_const = np.zeros(K)
  psibeta = digamma(gamma*beta) * beta
  for k in xrange(K):
    Sk = -1.0*psibeta[k]/v[k] + np.sum( psibeta[k+1:]/(1-v[k]) )
    dv_logpPi_const[k] = nDoc * gamma * Sk

  dv_logpPi = np.zeros(K)
  sbeta = sumLogPi * beta
  for k in xrange(K):
    Sk = sbeta[k]/v[k] - np.sum( sbeta[k+1:]/(1-v[k]) )
    dv_logpPi[k] = gamma * Sk

  return -1.0* ( dv_logpV + dv_logpPi_const + dv_logpPi )


def objGrad_v_FAST(v, sumLogPi, nDoc, gamma, alpha0):
  ''' Returns K-vector gradient of the constrained objective
      Args
      -------
      v := K-vector of real numbers, subject to 0 < v < 1

      Returns
      -------
      g := K-vector of real numbers, 
            g[k] = gradient of -1 * L(v) with respect to v[k]
  '''
  K = v.size
  beta = v2beta(v)
  dv_logpV = (1 - alpha0) / (1-v)

  diagIDs = np.diag_indices(K)
  lowTriIDs = np.tril_indices(K, -1)
  S = np.tile( sumLogPi * beta, (K,1))
  S /= (1.0 - v[:, np.newaxis])
  S[diagIDs] *= -1 * (1.0 - v)/v
  S[lowTriIDs] = 0
  dv_logpPi = gamma * np.sum(S, axis=1)

  S = np.tile( digamma(gamma*beta) * beta, (K,1))
  S /= (1.0 - v[:, np.newaxis])
  S[diagIDs] *= -1 * (1.0 - v)/v
  S[lowTriIDs] = 0
  dv_logpPi_const = nDoc * gamma * np.sum(S, axis=1)

  return -1.0* ( dv_logpV + dv_logpPi_const + dv_logpPi )


########################################################### Objective/gradient
########################################################### in terms of c

def objFunc_c(c, *args):
  ''' Returns scalar value of unconstrained objective function
      Args
      -------
      c := K-vector of real numbers

      Returns
      -------
      f := -1 * L( v2c(c) ), where L is ELBO objective (log posterior)
  '''
  v = sigmoid(c)
  # Force away from edges 0 or 1 for numerical stability  
  #v = np.maximum(v,EPS)
  #v = np.minimum(v,1.0-EPS)

  return objFunc_v(v, *args)

def objGrad_c(c, *args):
  ''' Returns K-vector gradient of unconstrained objective function
      Args
      -------
      c := K-vector of real numbers

      Returns
      -------
      g := K-vector of real numbers, 
            g[k] = gradient of -1 * L( v2c(c) ) with respect to c[k]
  '''
  v = sigmoid(c)
  # Force away from edges 0 or 1 for numerical stability  
  #v = np.maximum(v,EPS)
  #v = np.minimum(v,1.0-EPS)

  dfdv = objGrad_v(v, *args)
  dvdc = v * (1-v)
  dfdc = dfdv * dvdc
  return dfdc

########################################################### Transform funcs
########################################################### v2c, c2v

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

########################################################### Transform funcs
########################################################### v2beta, beta2v

def v2beta(v):
  ''' Convert to stick-breaking fractions v to probability vector beta
      Args
      --------
      v : K-len vector, v[k] in interval [0, 1]
      
      Returns
      --------
      beta : K+1-len vector, with positive entries that sum to 1
  '''
  v = np.asarray(v)
  beta = np.hstack([1.0, np.cumprod(1-v)])
  beta[:-1] *= v
  # Force away from edges 0 or 1 for numerical stability  
  #beta = np.maximum(beta,EPS)
  #beta = np.minimum(beta,1-EPS)
  return beta

def beta2v( beta ):
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
"""