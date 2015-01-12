import numpy as np
EPS = 1e-8

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

def forceRhoInBounds(rho, EPS=EPS):
  ''' Verify every entry of rho (and beta) are within [EPS, 1-EPS]
  '''
  rho = np.maximum(np.minimum(rho, 1.0-EPS), EPS)
  beta = rho2beta_active(rho)
  didFix = 0
  badMask = beta < EPS
  if np.any(beta < EPS):
    beta[badMask] = 1.01 * EPS
    addedMass = np.sum(badMask) * 1.01 * EPS
    beta[np.logical_not(badMask)] *= (1-addedMass)
    didFix = 1
  if (1.0 - np.sum(beta)) < EPS:
    kmax = beta.argmax()
    beta[kmax] -= 1.01 * EPS
    didFix = 1
  if didFix:
    rho = beta2rho(beta, rho.size)
  return rho

def forceOmegaInBounds(omega, EPS=EPS):
  ''' Verify every entry of omega is bigger than EPS
  '''
  np.maximum(omega, EPS, out=omega)
  return omega

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

def rho2beta(rho):
  ''' Calculate probability of each component, including "leftover" mass

      Returns
      --------
      beta : 1D array, size K+1
             beta[k] := probability of topic k
             will have positive entries whose sum is 1
  '''
  rho = np.asarray(rho, dtype=np.float64)
  beta = np.append(rho, 1.0)
  beta[1:] *= np.cumprod(1.0 - rho)
  return beta

def beta2rho(beta, K):
  ''' Returns K-length vector rho of stick-lengths that recreate appearance probs beta
  '''
  beta = np.asarray(beta, dtype=np.float64)
  rho = beta.copy()
  beta_gteq = 1 - np.cumsum(beta[:-1])
  rho[1:] /= np.maximum(1e-100, beta_gteq)
  if beta.size == K + 1:
    return rho[:-1]
  elif beta.size == K:
    return rho
  else:
    raise ValueError('Provided beta needs to be of length K or K+1')

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
