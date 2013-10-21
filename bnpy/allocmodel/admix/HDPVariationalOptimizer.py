'''
HDPVariationalOptimizer.py

Functions for variational approximation to HDP global appearance probabilities.

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

Generate each document-level distribution
for d in [1, 2, ... d ... nDoc]: 
  pi[d] ~ Dirichlet_{K+1}( gamma * beta )

Notes
-------
Relies on approximation to E[ log norm const of Dirichlet],
  which requires parameter gamma < 1

Set gamma close to 1 if you want variance low enough that
recovering true "beta" parameters is feasible

Set gamma close to zero (even like 0.1) makes recovered E[beta]
very different than the "true" beta
'''
import numpy as np
import scipy.optimize
from scipy.special import gammaln, digamma, polygamma

EPS = 10*np.finfo(float).eps

def createToyData(v, alpha0=1.0, gamma=0.5, nDoc=0, seed=42):
  ''' Generate example Pi matrix, 
        each row is a sample
  '''
  v = np.asarray(v, dtype=np.float64)
  K = v.size  
  beta = v2beta(v)

  PRNG = np.random.RandomState(seed)
  Pi = PRNG.dirichlet( gamma*beta, size=nDoc)
  return dict(Pi=Pi, alpha0=alpha0, gamma=gamma, nDoc=nDoc, K=K)

def estimate_u(alpha0=1.0, gamma=0.5, nDoc=0, K=2, sumLogPi=None, Pi=None, **kwargs):
  ''' Solve optimization problem to estimate parameters u
      for the approximate posterior on stick-breaking fractions v
      q(v | u) = Beta( v_k | u_k1, u_k0)

      Returns
      -------
      u1 : 
      u0 : 
  '''
  if nDoc == 0:
    initU = np.hstack( [0.1*np.ones(K), 0.1 * alpha0*np.ones(K)])
    sumLogPi = np.zeros(K+1)
  elif Pi is not None:
    logPi = np.maximum(np.log(Pi), -100)
    sumLogPi = np.sum(logPi, axis=0)
    initMeanBeta = np.mean(Pi, axis=0)
    initMeanV = beta2v(initMeanBeta)
    initSum = nDoc
    initU = np.hstack( [initSum*initMeanV, initSum*(1-initMeanV)])
    initU += 1 # so that it has a mode
  else:
    initU = np.hstack( [np.ones(K), alpha0*np.ones(K)])      
  
  print "INITIAL GUESS:"
  print "   U1   : ", initU[:K]
  print "   U0   : ", initU[K:]
  initBeta = v2beta(initU[:K]/(initU[:K]+initU[K:]))
  print "   E[beta] : ", initBeta
  
  myFunc = lambda Cvec: objectiveFunc(Cvec, alpha0, gamma, nDoc, sumLogPi)
  myGrad = lambda Cvec: objectiveGradient(Cvec, alpha0, gamma, nDoc, sumLogPi)

  bestCvec = scipy.optimize.fmin_bfgs(myFunc, np.log(initU), fprime=myGrad)
  bestUvec = np.exp(bestCvec)
  bestU1 = bestUvec[:K]
  bestU0 = bestUvec[K:]
  return bestU1, bestU0

def objectiveFunc2(Uvec, alpha0=1, gamma=1, nDoc=1, sumLogPi=1, **kwargs):
  Cvec = np.log(Uvec)
  return objectiveFunc(Cvec, alpha0, gamma, nDoc, sumLogPi)

def objectiveGradient2(Uvec, alpha0, gamma, nDoc, sumLogPi):
  Cvec = np.log(Uvec)
  return objectiveGradient(Cvec, alpha0, gamma, nDoc, sumLogPi)


def objectiveFunc(Cvec, alpha0, gamma, nDoc, sumLogPi):
  ''' Calculate unconstrained objective function for HDP variational learning
  '''
  assert not np.any(np.isnan(Cvec))

  # UNPACK unconstrained input Cvec into intended params U
  Uvec = np.exp(Cvec)
  K = Uvec.size/2
  U1 = Uvec[:K]
  U0 = Uvec[K:]


  # PREPARE building-block expectations
  E = calcExpectations(U1, U0)
  kvec = K+1 - np.arange(1, K+1)

  # CALCULATE each term in the function
  E_logp_v = (alpha0 - 1) * np.sum(E['log1mv'])

  E_logp_pi = nDoc * np.sum(E['logv']) \
            + nDoc * np.inner(kvec, E['log1mv']) \
            + np.inner(gamma * E['beta'] - 1, sumLogPi)
        
  E_logq_v = np.sum(gammaln(U1 + U0) - gammaln(U1) - gammaln(U0)) \
            + np.inner(U1 - 1, E['logv']) \
            + np.inner(U0 - 1, E['log1mv'])

  f = -1 * (E_logp_v + E_logp_pi - E_logq_v)
  return f

def objectiveGradient(Cvec, alpha0, gamma, nDoc, sumLogPi, E=dict()):
  ''' Calculate gradient of objectiveFunc, objective for HDP variational 
      Returns
      -------
        gvec : 2*K length vector,
              where each entry gives partial derivative with respect to
                  the corresponding entry of Cvec
  '''
  # UNPACK unconstrained input Cvec into intended params U
  Uvec = np.exp(Cvec)
  K = Uvec.size/2
  U1 = np.asarray(Uvec[:K], dtype=np.float64)
  U0 = np.asarray(Uvec[K:], dtype=np.float64)

  kvec = K+1 - np.arange(1, K+1)

  digammaU1 = digamma(U1)
  digammaU0 = digamma(U0)
  digammaBoth = digamma(U1+U0)

  if E is None or len(E.keys()) == 0:
    E = calcExpectations(U1, U0)
  dU1, dU0 = calcDerivatives(U1, U0, E=E)

  dU1_Elogp_v = (alpha0 - 1) * dU1['Elog1mv']
  dU0_Elogp_v = (alpha0 - 1) * dU0['Elog1mv']

  dU1_Elogp_pi = nDoc * dU1['Elogv'] \
                  + nDoc * kvec * dU1['Elog1mv'] \
                  + gamma * np.dot(dU1['Ebeta'], sumLogPi)
  dU0_Elogp_pi = nDoc * dU0['Elogv'] \
                  + nDoc * kvec * dU0['Elog1mv'] \
                  + gamma * np.dot(dU0['Ebeta'], sumLogPi)
  
  dU1_Elogq_v = digammaBoth - digammaU1 \
                  + E['logv'] + (U1 - 1) * dU1['Elogv'] \
                  + (U0 - 1) * dU1['Elog1mv']
  dU0_Elogq_v = digammaBoth - digammaU0 \
                  + (U1 - 1) * dU0['Elogv'] \
                  + E['log1mv'] + (U0 - 1) * dU0['Elog1mv']

  gvecU1 = dU1_Elogp_v + dU1_Elogp_pi - dU1_Elogq_v
  gvecU0 = dU0_Elogp_v + dU0_Elogp_pi - dU0_Elogq_v
  gvecU = -1 * np.hstack([gvecU1, gvecU0])

  # Apply chain rule!
  gvecC = Uvec * gvecU
  return gvecC

def calcDerivatives(U1, U0, E=dict()):
  ''' Calculate derivatives of building-block terms
  '''
  U1 = np.asarray(U1, dtype=np.float64)
  U0 = np.asarray(U0, dtype=np.float64)
  K = U1.size

  if E is None or len(E.keys()) == 0:
    E = calcExpectations(U1, U0)

  dU1 = dict()
  dU0 = dict()

  polygamma1Both = polygamma(1, U0 + U1)
  dU1['Elogv'] = polygamma(1,U1) - polygamma1Both
  dU0['Elogv'] = -polygamma1Both
  dU1['Elog1mv'] = -polygamma1Both
  dU0['Elog1mv'] = polygamma(1,U0) - polygamma1Both

  dU1_Ebeta = np.zeros((K,K+1))
  dU0_Ebeta = np.zeros((K,K+1))
  
  Usum = U1 + U0
  Q1 = U1 / (Usum * Usum)
  Q0 = U0 / (Usum * Usum)
  for m in xrange(K):
    dU1_Ebeta[m,m] = Q0[m] * E['beta'][m]/E['v'][m]
    dU0_Ebeta[m,m] = -Q1[m] * E['beta'][m]/E['v'][m]
    for k in xrange(m+1, K+1):
      dU1_Ebeta[m,k] = -Q0[m] * E['beta'][k]/E['1mv'][m]
      dU0_Ebeta[m,k] = Q1[m] * E['beta'][k]/E['1mv'][m]

  dU1['Ebeta'] = dU1_Ebeta
  dU0['Ebeta'] = dU0_Ebeta
  return dU1, dU0

def calcExpectations(U1, U0):
  ''' Calculate expectations of v and beta(v)
        under the model v[k] ~ Beta(U1[k], U0[k])
  '''
  U1 = np.asarray(U1, dtype=np.float64)
  U0 = np.asarray(U0, dtype=np.float64)

  E = dict()
  E['v'] = U1 / (U1 + U0)
  E['1mv'] = U0 / (U1 + U0)
  assert not np.any(np.isnan(U1))
  assert not np.any(np.isnan(U0))
  assert not np.any(np.isnan(E['v']))

  E['beta'] = v2beta(E['v'])

  digammaAll = digamma(U1 + U0)
  E['logv'] = digamma(U1) - digammaAll
  E['log1mv'] = digamma(U0) - digammaAll
  return E
  
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
  c1mv = np.hstack([1.0, np.cumprod(1 - v)])
  beta = np.hstack([v,1.0]) * c1mv
  assert np.allclose(beta.sum(), 1)
  # Force away from edges 0 or 1 for numerical stability  
  beta = np.maximum(beta,EPS)
  beta = np.minimum(beta,1-EPS)
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