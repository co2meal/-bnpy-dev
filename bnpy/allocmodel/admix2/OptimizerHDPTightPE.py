'''
OptimizerHDPTightPE.py

CONSTRAINED Optimization Problem
----------
Variables:
* rho : 1D array, size K

Objective:
* argmin -1 * E_{q(u|rho)} [ log p( \pi | u, \alpha) 
                              + log Beta(u | 1, gamma)]

Constraints: 
* rho satisfies: 0 < rho[k] < 1
'''

import warnings
import numpy as np
import scipy.optimize
import scipy.io
from scipy.special import gammaln, digamma, polygamma
import datetime
import logging

from RhoBetaUtil import rho2beta_active, beta2rho, sigmoid, invsigmoid
from RhoBetaUtil import forceRhoInBounds
from RhoBetaUtil import create_initrho

Log = logging.getLogger('bnpy')

def find_optimum_multiple_tries(DocTopicCount=0, nDoc=0, 
                                gamma=1.0, alpha=1.0,
                                initrho=None,
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
  rho = None
  Info = dict()
  msg = ''
  nOverflow = 0
  for trial, factr in enumerate(factrList):
    try:
      rho, f, Info = find_optimum(DocTopicCount, nDoc,
                                       gamma=gamma, alpha=alpha,
                                       initrho=initrho,
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

  if rho is None:
    raise ValueError(msg)
  Info['nOverflow'] = nOverflow
  return rho, f, Info


def find_optimum(DocTopicCount=0, nDoc=0, gamma=1.0, alpha=1.0,
                 initrho=None, 
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
  assert DocTopicCount.ndim == 2
  K = DocTopicCount.shape[1]

  ## Determine initial value
  if initrho is None:
    initrho = create_initrho(K)
  initrho = forceRhoInBounds(initrho)
  assert initrho.size == K
  initc = rho2c(initrho)

  ## Define objective function (unconstrained!)
  objArgs = dict(DocTopicCount=DocTopicCount,
                 nDoc=nDoc, gamma=gamma, alpha=alpha,
                 approx_grad=approx_grad)
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

  Info['init'] = initrho
  rho = c2rho(chat)
  rho = forceRhoInBounds(rho)
  return rho, fhat, Info

def create_initrho(K):
  ''' Make initial guess for rho s.t. E[beta_k] \approx uniform (1/K)
      except that a small amount of remaining/leftover mass is reserved
  '''
  remMass = np.minimum(0.1, 1.0/(K*K))
  # delta = 0, -1 + r, -2 + 2r, ...
  delta = (-1 + remMass) * np.arange(0, K, 1, dtype=np.float)
  rho = (1-remMass)/(K+delta)
  return rho


########################################################### Objective
###########################################################  unconstrained
def objFunc_unconstrained(c, approx_grad=False, **kwargs):
  rho = c2rho(c)
  if approx_grad:
    f = objFunc_constrained(rho, approx_grad=1, **kwargs)
    return f
  else:
    f, grad = objFunc_constrained(rho, approx_grad=0, **kwargs)
    drhodc = rho * (1-rho)
    return f, grad * drhodc

def c2rho(c, returnSingleVector=False):
  ''' Transform unconstrained variable c into constrained rho

      Returns
      --------
      rho : 1D array, size K, entries between [0, 1]
  '''
  rho = sigmoid(c)
  return rho

def rho2c(rho):
  return invsigmoid(rho)

########################################################### Objective
###########################################################  constrained
def objFunc_constrained(rho,
                     DocTopicCount=0, nDoc=0, gamma=1.0, alpha=1.0,
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
  rho = rho
  assert np.all(np.isfinite(rho))
  K = rho.size

  if nDoc > 0:
    OFFcoef = (gamma - 1.0) / nDoc
    
    Ebeta = rho.copy()
    Ebeta[1:] *= np.cumprod(1-rho[:-1])
    alphaEbeta = alpha * Ebeta
    theta = DocTopicCount + alphaEbeta
    thetaRem = alpha * (1-np.sum(Ebeta))
    elbo_local = np.sum(gammaln(theta)) / nDoc \
                 - np.sum(gammaln(alphaEbeta))

  else:
    OFFcoef = gamma - 1
    elbo_local = 0

  Elog1mU = np.log(1.0 - rho)
  elbo = np.sum(OFFcoef * Elog1mU) \
         + elbo_local

  if approx_grad:
    return -1.0 * elbo

  ## Gradient computation!  
  gradrho = - OFFcoef / (1 - rho)
  if nDoc > 0:
    Delta = calc_dBeta_drho(Ebeta, rho, rho.size)
    Tvec = np.sum(digamma(theta), axis=0) / nDoc \
           - digamma(alpha * Ebeta)
    gradrho += alpha * np.dot(Delta, Tvec)
  return -1.0 * elbo, -1.0 * gradrho
  
########################################################### Util fcns
###########################################################

def c_Beta(g1, g0):
  ''' Calculate cumulant function of the Beta distribution

      Returns
      -------
      c : scalar sum of the cumulants defined by provided parameters
  '''
  return np.sum(gammaln(g1 + g0) - gammaln(g1) - gammaln(g0))


lowTriIDsDict = dict()
def _get_lowTriIDs(K):
  if K in lowTriIDsDict:
    return lowTriIDsDict[K]
  else:
    ltIDs = np.tril_indices(K, -1)
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

def calc_dBeta_drho(Ebeta_active, rho, K):
  ''' Calculate partial derivative of Ebeta w.r.t. rho

      Returns
      ---------
      Delta : 2D array, size K x K
  '''
  Delta = np.tile(-1 * Ebeta_active, (K,1))
  Delta /= (1-rho)[:,np.newaxis]
  Delta[_get_diagIDs(K)] *= -1 * (1-rho)/rho
  Delta[_get_lowTriIDs(K)] = 0
  return Delta