'''
LocalOptimizer.py

Perform local-step at a single document via gradient descent.

CONSTRAINED Optimization Problem
----------
Variables:
Two K-length vectors
* etaON
* etaOFF

Objective:
* argmin ELBO(etaON, etaOFF)

Constraints: 
* 0 < etaOFF < 1
* 0 < etaON < 1
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


def find_optimum(alphaEbeta=0, alphaEbeta_gt=0, Lik_d=0, wc_d=1, 
                 initeta=None,
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
  K = alphaEbeta.size
  assert alphaEbeta_gt.size == K
  assert Lik_d.ndim == 2
  assert Lik_d.shape[1] == K

  ## Determine initial value
  if initeta is None or initeta.min() <= EPS:
    initeta = create_initeta(alphaEbeta, alphaEbeta_gt)

  assert initeta.size == 2*K
  assert initeta.min() > EPS
  initc = eta2c(initeta)

  ## Define objective function (unconstrained!)
  objArgs = dict(Lik_d=Lik_d, wc_d=1,
                 alphaEbeta=alphaEbeta, alphaEbeta_gt=alphaEbeta_gt,
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

  Info['init'] = initrhoomega
  etaON, etaOFF = c2eta(chat, returnSingleVector=0)
  return etaON, etaOFF, fhat, Info

def create_initeta(alphaEbeta, alphaEbeta_gt):
  ''' Initial guess

      Returns
      -------
      etaON
      etaOFF
  '''
  return np.hstack([alphaEbeta, alphaEbeta_gt])

########################################################### Objective
###########################################################  unconstrained
def objFunc_unconstrained(c, approx_grad=False, **kwargs):
  eta = c2eta(c, returnSingleVector=1)
  if approx_grad:
    f = objFunc_constrained(eta, approx_grad=1, **kwargs)
    return f
  else:
    f, grad = objFunc_constrained(eta, approx_grad=0, **kwargs)
    detadc = eta
    return f, grad * detadc

def c2eta(c, returnSingleVector=False):
  ''' Transform unconstrained variable c into constrained eta

      Returns
      --------
      etaON
      etaOFF

      OPTIONAL: may return as one concatenated vector (length 2K)
  '''
  eta = np.exp(c)
  if returnSingleVector:
    return eta
  K = c.size/2
  return eta[:K], eta[K:]

def eta2c(eta, scaleVector=None):
  return np.log(eta)

########################################################### Objective
###########################################################  constrained
def objFunc_constrained(eta,
                     alphaEbeta=0, alphaEbeta_gt=0, Lik_d=0, wc_d=1, 
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
  assert np.all(np.isfinite(eta))
  etaON, etaOFF, K = _unpack(eta)
  
  digammaBoth = digamma(etaON+etaOFF)
  ElogV = digamma(etaON) - digammaBoth
  Elog1mV = digamma(etaOFF) - digammaBoth

  ## Calculate R[n] = \sum_k exp[ E log[ Lik_k + log pi_k ] ]
  expElogPi = ElogV.copy()
  expElogPi[1:] -= np.cumsum(Elog1mV[:-1])
  np.exp(expElogPi, out=expElogPi)
  logRtilde = np.dot(Lik_d, expElogPi)
  np.log(logRtilde, out=logRtilde)
  logRtilde *= wc_d

  ONcoef = alphaEbeta - etaON
  OFFcoef = alphaEbeta_gt - etaOFF
  elbo = -1 * c_Beta(etaON, etaOFF) \
          + np.inner(ONcoef, ElogV) \
          + np.inner(OFFcoef, Elog1mV) \
          + np.sum(logRtilde)

  if approx_grad:
    return -1.0 * elbo

  ## Gradient computation!
  # First, do part from the "prior" terms
  trig_ON = psi(1, etaON)
  trig_OFF = psi(1, etaOFF)
  trig_BOTH = psi(1, etaON+etaOFF)
  gradetaON = ONcoef * (trig_ON - trig_BOTH) \
              - OFFcoef * trig_BOTH
  gradetaOFF = OFFcoef * (trig_OFF - trig_BOTH) \
               - ONcoef * trig_BOTH
  # Now, the likelihood part
  raise NotImplementedError('ToDo')
  
  grad = np.hstack([gradetaON, gradetaOFF])
  return -1.0 * elbo, -1.0 * grad
  
########################################################### Util fcns
###########################################################
def _unpack(eta):
  K = eta.size / 2
  return eta[:K], eta[K:], K

def c_Beta(g1, g0):
  ''' Calculate cumulant function of the Beta distribution

      Returns
      -------
      c : scalar sum of the cumulants defined by provided parameters
  '''
  return np.sum(gammaln(g1 + g0) - gammaln(g1) - gammaln(g0))