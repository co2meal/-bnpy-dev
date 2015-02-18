import numpy as np
import bnpy
from scipy.spatial.distance import cdist

import dill
from pathos.multiprocessing import Pool
import unittest

def localStep_Vectorized(X, Mu, start=None, stop=None):
  ''' K-means step

      Returns
      -----------
      SuffStatBag with fields
      * N : 1D array, size K
      * x : 2D array, K x D
  '''
  # Unpack size variables
  K, D = Mu.shape

  # Assign each row of X to closest row of Mu
  if start is not None:
    print 'WORKING ON CHUNK %d:%d' % (start, stop)
    Xcur = X[start:stop]
  else:
    Xcur = X

  Dist = cdist(Xcur, Mu,'sqeuclidean')
  Z = Dist.argmin(axis=1)

  CountVec = np.zeros(K)
  DataStatVec = np.zeros((K,D))
  for k in xrange(K):
    mask_k = Z == k
    CountVec[k] = np.sum(mask_k)
    DataStatVec[k] = np.sum(Xcur[mask_k], axis=0)

  SS = bnpy.suffstats.SuffStatBag(K=K, D=D)
  SS.setField('CountVec', CountVec, dims=('K'))
  SS.setField('DataStatVec', DataStatVec, dims=('K','D'))
  return SS

class TestKMeans(unittest.TestCase):

  def setUp(self, N=1000, D=25, K=10):
    ''' Create a dataset X (2D array, N x D) and cluster means Mu (2D, KxD)
    '''
    rng = np.random.RandomState(N*D*K)
    self.X = rng.rand(N, D)
    self.Mu = rng.rand(K, D)

  def test_correctness_localStep(self):
    ''' Verify that the local step works as expected
    '''
    print ''
    N = self.X.shape[0]

    # Version A: summarize entire dataset
    SSall = localStep_Vectorized(self.X, self.Mu)  

    # Version B: summarize first and second halves separately, then add together
    SSpart1 = localStep_Vectorized(self.X, self.Mu, start=0, stop=N/2)
    SSpart2 = localStep_Vectorized(self.X, self.Mu, start=N/2, stop=N)
    SSbothparts = SSpart1 + SSpart2

    # Both A and B better give the same answer
    assert np.allclose(SSall.CountVec, SSbothparts.CountVec)
    assert np.allclose(SSall.DataStatVec, SSbothparts.DataStatVec)


  def localStep_SingleProcess(self, procID, processes):
    N = self.X.shape[0]
    batch_size = np.floor(N / float(processes))
    start = procID * batch_size
    if procID == processes - 1:
      stop = N
    else:
      stop = (procID-1) * batch_size
    return localStep_Vectorized(self.X, self.Mu, start=start, stop=stop)

  def test_parallel_localStep(self, processes=2):

    pool = Pool(processes=processes)
    output = pool.map(self.localStep_SingleProcess, range(processes))
    print output

