'''
StarCovarK5
'''
import scipy.linalg
import numpy as np
from bnpy.util.RandUtil import rotateCovMat

from bnpy.data import XData

######################################################################  Generate Toy Params
K = 5
D = 2

w = np.asarray( [5., 4., 3., 2., 1.] )
w = w/w.sum()

Mu = np.zeros( (K,D) )

# Create basic 2D cov matrix with major axis much longer than minor one
V = 1.0/16.0
SigmaBase = np.asarray([[ V, 0], [0, V/100.0]])

# Create several Sigmas by rotating this basic covariance matrix
Sigma = np.zeros( (5,D,D) )
for k in xrange(4):
  Sigma[k] = rotateCovMat(SigmaBase, k*np.pi/4.0)

# Add final Sigma with large isotropic covariance
Sigma[4] = 4*V*np.eye(D)

# Precompute cholesky decompositions
cholSigma = np.zeros( Sigma.shape )
for k in xrange( K ):
  cholSigma[k] = scipy.linalg.cholesky( Sigma[k] )

######################################################################  Module Util Fcns
def sample_data_from_comp( k, Nk, PRNG ):
  return Mu[k,:] + np.dot(cholSigma[k].T, PRNG.randn(D, Nk) ).T

def get_short_name( ):
  return 'StarCovarK5'

def get_data_info():
  return 'Overlapping Star Toy Data. K=%d. D=%d.' % (K,D)

######################################################################  MixModel Data
def get_X( seed, nObsTotal):
  PRNG = np.random.RandomState( seed )
  trueList = list()
  Npercomp = PRNG.multinomial( nObsTotal, w )
  X = list()
  for k in range(K):
    X.append( sample_data_from_comp( k, Npercomp[k], PRNG) )
    trueList.append( k*np.ones( Npercomp[k] ) )
  X = np.vstack( X )
  TrueZ = np.hstack( trueList )
  permIDs = PRNG.permutation( X.shape[0] )
  X = X[permIDs]
  TrueZ = TrueZ[permIDs]
  return X, TrueZ

def get_data( seed=8675309, nObsTotal=25000, **kwargs ):
  X, TrueZ = get_X(seed, nObsTotal)
  Data = XData(X=X)
  Data.summary = get_data_info()
  return Data

'''
def minibatch_iterator( batch_size=5000, nBatch=10, nRep=1, seed=8675309, orderseed=42, **kwargs):
  # NB: X, Z are already shuffled
  X, TrueZ = get_X( seed, batch_size*nBatch )
  Data = dict( X=X, nObs=X.shape[0])
  MBG = MinibatchIterator( Data, nBatch=nBatch, batch_size=batch_size, nRep=nRep, orderseed=orderseed )
  return MBG

def minibatch_generator( batch_size=5000, nBatch=10, nRep=1, seed=8675309, orderseed=42, **kwargs):
  # NB: X, Z are already shuffled
  X, TrueZ = get_X( seed, batch_size*nBatch )

  # Divide data into permanent set of minibatches
  obsIDs = range( X.shape[0] )
  obsIDByBatch = dict()
  for batchID in range( nBatch):
    obsIDByBatch[batchID] = obsIDs[:batch_size]
    del obsIDs[:batch_size]

  # Now generate the minibatches one at a time
  PRNG = np.random.RandomState( orderseed )
  for repID in range( nRep):
    batchIDs = PRNG.permutation( nBatch )
    print '-------------------------------------------------------  batchIDs=', batchIDs[:4], '...'
    for passID,bID in enumerate(batchIDs):
      curX = X[ obsIDByBatch[bID] ].copy()
      curData = dict( X=curX, nObs=curX.shape[0], nTotal=batch_size*nBatch, bID=bID, passID=passID )
      yield curData
'''
