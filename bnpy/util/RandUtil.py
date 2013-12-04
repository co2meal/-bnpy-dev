'''
RandUtil.py

Utilities for sampling (pseudo) random numbers
'''
import numpy as np

def multinomial(Nsamp, ps, randstate=np.random):
  ps = np.asarray(ps, dtype=np.float64)
  Pmat = np.tile(ps, (Nsamp,1))
  choiceVec = discrete_single_draw_vectorized(Pmat, randstate)
  choiceHist, bins = np.histogram(choiceVec, np.arange(-.5,ps.size + .5))
  return choiceHist

def discrete_single_draw_vectorized( Pmat, randstate=np.random):
  Ts = np.cumsum(Pmat, axis=1)
  throws = randstate.rand( Pmat.shape[0] )*Ts[:,-1]
  Ts[ Ts > throws[:,np.newaxis] ] = np.inf
  choices = np.argmax( Ts, axis=1 ) # relies on argmax returning first id
  return choices

def discrete_single_draw( ps, randstate=None):
  ''' Given vector of K weights "ps",
         draw a single integer assignment in {1,2, ...K}
      such that Prob( choice=k) = ps[k]
  '''
  totals = np.cumsum(ps)
  if randstate is None:
    throw = np.random.rand()*totals[-1]
  else:
    throw = randstate.rand()*totals[-1]
  return np.searchsorted(totals, throw)

def mvnrand(mu, Sigma, N=1, PRNG=np.random.RandomState()):
  if type(PRNG) == int:
    PRNG = np.random.RandomState(PRNG)
  return PRNG.multivariate_normal(mu, Sigma, (N))
  
def rotateCovMat( Sigma, theta=np.pi/4):
  ''' Returns valid covariance matrix with same eigen structure, rotated by theta radians
  '''
  RotMat = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
  RotMat = np.asarray( RotMat)
  Lam,V = np.linalg.eig( Sigma )
  Lam = np.diag(Lam)
  Vrot = np.dot( V, RotMat )
  return np.dot( Vrot, np.dot( Lam, Vrot.T) )