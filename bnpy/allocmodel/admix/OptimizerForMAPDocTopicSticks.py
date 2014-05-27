'''
OptimizerForMAPDocTopicSticks.py
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


def find_optimum_multiple_tries(Xd=0, Ld=0, avec=0, bvec=0,
                                initeta=None,
                                approx_grad=1,
                                factrList=[1e5, 1e7, 1e9, 1e10, 1e11, 1e12],
                                return_pi=0,
                                **kwargs):
  ''' 
      Raises
      --------
      ValueError with FAILURE in message if all restarts fail
  '''
  eta = None
  Info = dict()
  msg = ''
  nOverflow = 0
  for trial, factr in enumerate(factrList):
    try:
      eta, f, Info = find_optimum(Xd, Ld, avec, bvec, 
                                  initeta,
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

  if eta is None:
    raise ValueError(msg)
  Info['nOverflow'] = nOverflow

  if return_pi:
    eta = _v2beta(sigmoid(eta))[:-1]
  return eta, f, Info      

########################################################### find_optimum
###########################################################
def find_optimum(Xd=0, Ld=0, avec=0, bvec=0,
                 initeta=None,
                 approx_grad=1, factr=1.0e5, **kwargs):
  ''' Run gradient optimization to find MAP estimate

      Returns
      --------
      eta : 1D array, size K
      f : scalar value of minimization objective
      Info : dict

      Raises
      --------
      ValueError on an overflow, any NaN, or failure to converge
  '''
  assert Ld.ndim == 2
  Nd, K = Ld.shape

  assert Xd.ndim == 1 and Xd.size == Nd
  assert avec.ndim == 1 and avec.size == K
  assert bvec.ndim == 1 and bvec.size == K

  if initeta is None:
    initeta = create_initeta__FromPrior(avec, bvec)
  assert initeta.size == K and not np.any(np.isinf(initeta))
  
  ## Define objective function (unconstrained!)
  objArgs = dict(Xd=Xd, Ld=Ld, avec=avec, bvec=bvec)

  objFunc = lambda eta: objFunc_unconstrained(eta, **objArgs)
  
  ## Run optimization and catch any overflow or NaN issues
  with warnings.catch_warnings():
    warnings.filterwarnings('error', category=RuntimeWarning,
                               message='overflow')
    try:
      bestEta, fbest, Info = scipy.optimize.fmin_l_bfgs_b(objFunc, initeta,
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

  Info['objFunc'] = objFunc
  return bestEta, fbest, Info


def create_initeta__FromPrior(avec, bvec):
  v = avec / (avec + bvec)
  return invsigmoid(v)


def create_initeta__Uniform(K, remMass=0.01):
  if K == 1:
    return (1-remMass) * np.ones(1)
  rem = np.minimum( remMass, 1.0/(K*K))
  beta = (1.0 - rem)/K * np.ones(K+1)
  beta[-1] = rem
  return invsigmoid(_beta2v(beta))


########################################################### Objective
###########################################################  unconstrained
def objFunc_unconstrained(eta, Xd, Ld, avec, bvec, approx_grad=False):
  K = eta.size
  v = sigmoid(eta)
  pi = _v2beta(v)[:-1]

  logv = np.log(v)
  log1mv = np.log(1-v)
  logPrior = np.inner(avec, logv) + np.inner(bvec, log1mv)
  logLik = np.inner(Xd, np.log(np.dot(Ld, pi)))
  return -1 * (logPrior + logLik)

########################################################### Var conversion funcs
########################################################### 
def pi2eta(pi):
  pi = np.asarray(pi)
  pi = np.hstack([pi, 1-pi.sum()])
  return invsigmoid(_beta2v(pi))

def eta2pi(eta):
  return _v2beta(sigmoid(eta))[:-1]

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


