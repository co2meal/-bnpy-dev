'''
TrickyToyGaussData

  Streaming data generator that draws samples from a tough-to-learn GMM
     with 16 components

  Author: Mike Hughes (mike@michaelchughes.com)
'''
import scipy.linalg
import numpy as np
from MinibatchIterator import MinibatchIterator

K = 5
D = 2
alpha = 0.5
nGroupPerBatch = 5

######################################################################  Generate Toy Params
w = np.asarray( [5., 4., 3., 2., 1.] )
w = w/w.sum()

Mu = np.zeros( (K,D) )

theta = np.pi/4
RotMat = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
RotMat = np.asarray( RotMat)
varMajor = 1.0/16.0
SigmaBase = [[ varMajor, 0], [0, varMajor/100.0]]
SigmaBase = np.asarray(SigmaBase)

Lam,V = np.linalg.eig( SigmaBase )
Lam = np.diag(Lam)
Sigma = np.zeros( (5,D,D) )
for k in xrange(4):
  Sigma[k] = np.dot( V, np.dot( Lam, V.T) )
  V = np.dot( V, RotMat )
Sigma[4] = 1.0/5.0*np.eye(D)

cholSigma = np.zeros( Sigma.shape )
for k in xrange( K ):
  cholSigma[k] = scipy.linalg.cholesky( Sigma[k] )

######################################################################  Module Util Fcns
def sample_data_from_comp( k, Nk, PRNG ):
  return Mu[k,:] + np.dot( cholSigma[k].T, PRNG.randn( D, Nk) ).T

def get_short_name( ):
  return 'TrickyToy'

def print_data_info( mName ):
  print 'Tricky Toy Data. K=%d. D=%d.' % (K,D)

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
  X, TrueZ = get_X( seed, nObsTotal)
  Data = dict( X=X, TrueZ=TrueZ, nObs=nObsTotal )
  return Data


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
  for repID in range( nRep ):
    obsIDs = range( X.shape[0] )
    for batchID in range( nBatch):
      if len(obsIDs) < batch_size:
        break
      bINDS = obsIDs[ :batch_size ]
      del obsIDs[:batch_size]
      curX = X[bINDS].copy()
      curData = dict( X=curX, nObs=len(bINDS), nTotal=batch_size*nBatch, bID=batchID )
      yield curData
  '''
'''
def get_data( seed=8675309, nObsTotal=None, batch_size=25000, **kwargs ):
  if nObsTotal is not None:
    batch_size = nObsTotal
  DG = minibatch_generator( batch_size=batch_size, nBatch=1, seed=seed)
  Data = DG.next()
  if 'nTotal' in Data:
    del Data['nTotal']
    del Data['bID']
  return Data

def minibatch_generator(  batch_size=1000, nBatch=50, nRep=1, seed=8675309, **kwargs):
  for repID in range( nRep ):
    np.random.seed( seed )
    for batchID in range( nBatch ):
      trueList = list()
      Npercomp = np.random.mtrand.multinomial( batch_size, w )
      X = list()
      for k in range(K):
        X.append( sample_data_from_comp( k, Npercomp[k]) )
        trueList.append( k*np.ones( Npercomp[k] ) )
      X = np.vstack( X )
      TrueZ = np.hstack( trueList )
      yield {'X':X, 'nObs':X.shape[0], 'nTotal':batch_size*nBatch, 'TrueZ':TrueZ, 'bID':batchID}
'''
######################################################################  AdmixModel Data
def get_data_by_groups( seed=8675309, batch_size=25000, **kwargs ):
  DG = group_minibatch_generator( batch_size=batch_size, nBatch=1, seed=seed)
  Data = DG.next()
  if 'nTotal' in Data:
    del Data['nTotal']
    del Data['nGroupTotal']
  return Data

def group_minibatch_generator(  batch_size=5000, nBatch=50, nRep=1, seed=8675309, **kwargs):
  for repID in range( nRep ):
    np.random.seed( seed )
    for batchID in range( nBatch ):
      GroupIDs = list()
      curID = 0
      Xall = np.empty( (batch_size,D) )
      GroupSizes = batch_size/nGroupPerBatch*np.ones( nGroupPerBatch)
      if np.sum( GroupSizes) < batch_size:
        GroupSizes[-1] = batch_size - np.sum( GroupSizes[:-1])
      TrueW = np.zeros( (nGroupPerBatch,K) )
      for gID in xrange( nGroupPerBatch):
        w = np.random.mtrand.dirichlet( alpha*np.ones(K) )
        TrueW[gID] = w
        Npercomp = np.random.mtrand.multinomial( GroupSizes[gID], w )
        X = list()
        for k in range(K):
          X.append( sample_data_from_comp( k, Npercomp[k]) )
        X = np.vstack( X )
        GroupIDs.append( (curID, curID+X.shape[0]) )
        Xall[ curID:curID+X.shape[0] ] = X
        curID += X.shape[0]
      yield {'X':Xall, 'GroupIDs':GroupIDs, 'nObs':Xall.shape[0], 'nTotal':batch_size*nBatch, \
              'bID':batchID, 'nGroup':nGroupPerBatch, 'nGroupTotal':nGroupPerBatch*nBatch, 'TrueW':TrueW}
