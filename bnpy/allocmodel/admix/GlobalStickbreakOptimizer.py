'''
GlobalStickbreakOptimizer.py

Functions for learning global-level appearance probabilities given observed group-level probabilities via constrained optimization.

Model Notation
--------
HDP with a finite truncation to K active components
v    := K-length vector with entries in [0,1]
beta := K+1-length vector with entries in [0,1]
          entries must sum to unity.  sum(beta) = 1.
alpha0 := scalar, alpha0 > 0
gamma := scalar, gamma > 0

Generate stick breaking fractions v 
  v[k] ~ Beta(1, alpha0)
Then deterministically obtain beta
  beta[k] = v[k] prod(1 - v[:k])

Generate each group-level distribution
for g in [1, 2, ... g ... G]: 
  pi[g] ~ Dirichlet( gamma * beta )
'''

import numpy as np
import scipy.optimize
from scipy.special import digamma, gammaln

# Define numerical constants
EPS = 10*np.finfo(float).eps

################################################################## User-facing func
##################################################################
def estimate_v( G, logpiMat, alpha0, gamma, vinitIN=None, LB=1e-5, Ntrial=1, method='tnc'):
  ''' Find point-estimate of K-len vector v of stick-breaking fractions
      that optimizes the posterior objective function for truncated HDP stick-breaking.
      Args
      --------
      G : number of groups / documents for group-level of HDP
      logPiMat : G x K+1 array of per-group log probabilities
      alpha0 : global-level param of HDP
      gamma : group-level param of HDP
      
      Returns
      ---------
      v : K-len vector of global-level stick break fractions
          each entry v[k] lies in interval [0,1]
  '''
  if logpiMat.ndim == 2:
    logpi = np.sum(logpiMat, axis=0)
  else:
    logpi = logpiMat
  K = logpi.size - 1
  
  objfunc = lambda v: neglogp( v, G, logpi, alpha0, gamma)
  objgrad = lambda v: gradneglogp( v, G, logpi, alpha0, gamma)
 
  # Define the bounds for target variable v during L-BFGS optimization
  Bounds = [(LB, 1-LB) for k in range(K)]
  
  # Run Ntrial attempts (from distinct initializations)
  # until some trial passes a check
  for trial in xrange(Ntrial):
    if vinitIN is None:
      pi = np.exp(logpi)
      pi = beta2v(pi) + 0.001*np.random.randn(K)
      pi = np.maximum(pi, LB)
      pi = np.minimum(pi, 1- LB)
      vinit = pi/sum(pi)
      assert np.all(vinit > 0)
      assert np.all(vinit < 1)    
    else:
      vinit = 0.00001*np.random.randn(K) + vinitIN
      vinit = np.maximum( vinit, LB)
      vinit = np.minimum( vinit, 1-LB)
      
    finit = objfunc(vinit)
    
    if method == 'tnc':
      v,f,d = scipy.optimize.fmin_tnc(objfunc, x0=vinit, fprime=objgrad, bounds=Bounds, messages=0)
    elif method == 'L-BFGS-B':
      v,f,d = scipy.optimize.fmin_l_bfgs_b(objfunc, x0=vinit, fprime=objgrad, bounds=Bounds)
    elif method == 'minimize':
      optimResult = scipy.optimize.minimize(objfunc, x0=vinit, bounds=Bounds, method='TNC')
      v = optimResult.x
      f = objfunc(v)
    
    if check_bounds( v, f, finit, LB):
      return v
  return vinit  
  
def check_bounds(x, f, finit, LB):
  isGood = np.all( x >= LB )
  isGood = isGood and np.all( x <= 1-LB )
  return isGood

##################################################################  Objective function
##################################################################
def neglogp( v, G, logpi, alpha0, gamma ):
  ''' Compute negative log posterior prob of v
        up to an additive constant
      Args
      -------
      logPi
      Returns
      -------
      -1 * log p ( v | logpiMat)
  '''
  assert np.all( v > 0)
  assert np.all( v < 1 )

  beta = v2beta(v)
  
  logp = (alpha0 - 1) * np.sum(np.log(1 - v))
  logp -= G * np.sum(gammaln(gamma*beta))
  logp += gamma * np.sum(beta * logpi)
  assert np.all(np.logical_not(np.isnan(logp)))
  return -1.0*logp

################################################################## Gradient of obj func
##################################################################
def gradneglogp( v, G, logpi, alpha0, gamma ):
  ''' Compute gradient of the negative log posterior prob of v
        up to an additive constant
      Returns
      --------
      gradvec : K-length vector,
          gradvec[k] gives partial derivative w.r.t. k
  '''
  assert np.all(v > 0)
  assert np.all(v < 1)

  beta = v2beta(v)
  dBdv = dbetadv(v, beta)

  gradvec = -1 * (alpha0 - 1.0) / (1.0 - v)
  gradvec -= G * gamma * np.dot(dBdv, digamma(gamma*beta))
  gradvec += gamma * np.dot(dBdv, logpi)
  
  assert np.all(np.logical_not(np.isnan(gradvec)))
  return -1.0*gradvec

def dbetadv( v, beta):
  ''' Compute gradient of beta with respect to v
      Returns
      -------
      dbdv : K x K+1 matrix, where
      dbdv[m,k] = d beta[k] / d v[m]
  '''
  K = v.size
  dbdv = np.zeros( (K, K+1) )
  for k in xrange( K ):
    dbdv[k, k] = beta[k]/v[k]
    dbdv[k, k+1:] = -1.0*beta[k+1:]/(1-v[k])
  return dbdv
  
################################################################## Transform v <--> beta
##################################################################
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
  
def v2beta( v ):
  ''' Convert to stick-breaking fractions v to probability vector beta
      Args
      --------
      v : K-len vector, v[k] in interval [0, 1]
      
      Returns
      --------
      beta : K+1-len vector, with positive entries that sum to 1
  '''
  v = np.hstack( [v, 1] )
  c1mv = np.cumprod( 1 - v )
  c1mv = np.hstack( [1, c1mv] )
  beta = v * c1mv[:-1]
  
  # Force away from edges 0 or 1 for numerical stability  
  beta = np.maximum(beta,EPS)
  beta = np.minimum(beta,1-EPS)
  return beta

