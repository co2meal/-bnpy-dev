'''
LocalOptimizerDir.py

Perform local-step at a single document via gradient descent.

CONSTRAINED Optimization Problem
----------
Variables:
K-length vectors
* theta

Objective:
* argmin ELBO(theta)

Constraints: 
* 0 < theta
'''

import warnings
import numpy as np
import scipy.optimize
import scipy.io
from scipy.special import gammaln, digamma
from scipy.special import polygamma as psi
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
      return find_optimum_multiple_tries(sumLogVd, sumLog1mVd, nDoc, 
                                gamma=gamma, alpha=alpha,
                                initrho=None, initomega=None,
                                approx_grad=approx_grad, **kwargs)
    else:
      raise ValueError(msg)
  Info['nOverflow'] = nOverflow
  rho, omega, K = _unpack(rhoomega)
  return rho, omega, f, Info


def find_optimum(alphaEbeta=0, Lik_d=0, wc_d=1, 
                 inittheta=None, Data=None,
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
  K = alphaEbeta.size - 1
  assert Lik_d.ndim == 2
  assert Lik_d.shape[1] == K

  ## Determine initial value
  if inittheta is None or inittheta.min() <= EPS:
    inittheta = create_inittheta(alphaEbeta)

  assert inittheta.size == K+1
  assert inittheta.min() > EPS
  initc = theta2c(inittheta)

  ## Define objective function (unconstrained!)
  objArgs = dict(Lik_d=Lik_d, wc_d=wc_d,
                 alphaEbeta=alphaEbeta,
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

  Info['init'] = inittheta
  theta = c2theta(chat)
  return theta, fhat, Info

def create_inittheta(alphaEbeta):
  ''' Initial guess

      Returns
      -------
      theta
  '''
  return alphaEbeta

########################################################### Objective
###########################################################  unconstrained
def objFunc_unconstrained(c, approx_grad=False, **kwargs):
  theta = c2theta(c)
  if approx_grad:
    f = objFunc_constrained(theta, approx_grad=1, **kwargs)
    return f
  else:
    f, grad = objFunc_constrained(theta, approx_grad=0, **kwargs)
    dthetadc = theta
    return f, grad * dthetadc

def c2theta(c):
  ''' Transform unconstrained variable c into constrained eta

      Returns
      --------
      theta
  '''
  theta = np.exp(c)
  return theta

def theta2c(theta):
  return np.log(theta)

########################################################### Objective
###########################################################  constrained
def objFunc_constrained(theta,
                     alphaEbeta=0, Lik_d=0, wc_d=1, 
                     approx_grad=False, **kwargs):
  ''' Returns constrained objective function and its gradient

      Args
      -------
      eta := 1D array, size 2*K

      Returns
      -------
      f := -1 * L(eta), 
           where L is ELBO objective function (log posterior prob)
      g := gradient of f
  '''
  assert np.all(np.isfinite(theta))
  
  digammaAll = digamma(theta.sum())
  ElogPi = digamma(theta) - digammaAll
  expElogPi = np.exp(ElogPi[:-1])

  ## Calculate R[n] = \sum_k exp[ E log[ Lik_k + log pi_k ] ]
  Rtilde = np.dot(Lik_d, expElogPi)
  logRtilde = np.log(Rtilde)
  logRtilde *= wc_d

  ONcoef = alphaEbeta - theta
  elbo = -1 * c_Dir(theta) \
          + np.inner(ONcoef, ElogPi) \
          + np.sum(logRtilde)

  if approx_grad:
    return -1.0 * elbo

  ## Gradient computation!
  # First, do "prior" terms
  trig_theta = psi(1, theta)
  trig_ALL = psi(1, theta.sum())
  gradtheta = ONcoef * trig_theta \
              - np.sum(ONcoef) * trig_ALL

  # Now, the likelihood part
  Delta = dElogPi_dtheta(trig_theta, trig_ALL)
  LPi_d = Lik_d * expElogPi[np.newaxis,:]
  gradtheta += np.sum(np.dot(Delta, LPi_d.T) * (wc_d / Rtilde)[np.newaxis,:],
                      axis=1)
  return -1.0 * elbo, -1.0 * gradtheta
  
########################################################### Util fcns
###########################################################
def _unpack(eta):
  K = eta.size / 2
  return eta[:K], eta[K:], K

def c_Dir(avec):
  ''' Calculate cumulant function of the Dirichlet distribution

      Returns
      -------
      c : scalar sum of the cumulants defined by provided parameters
  '''
  return gammaln(np.sum(avec)) - np.sum(gammaln(avec))

def dElogPi_dtheta(trig_theta, trig_ALL):
  K = trig_theta.size - 1
  Delta = -trig_ALL * np.ones((K+1, K))
  Delta[_get_diagIDs(K)] += trig_theta[:-1]
  return Delta

diagIDsDict = dict()
def _get_diagIDs(K):
  if K in diagIDsDict:
    return diagIDsDict[K]
  else:
    diagIDs = np.diag_indices(K)
    diagIDsDict[K] = diagIDs
    return diagIDs