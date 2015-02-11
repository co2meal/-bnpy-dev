import numpy as np
import bnpy
from scipy.spatial.distance import cdist2

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
    Dist = cdist2(X[start:stop], Mu,'sqeuclidean')
  else:
    Dist = cdist2(X, Mu,'sqeuclidean')
  Z = Dist.argmin(axis=1)

  CountVec = np.zeros(K)
  DataStatVec = np.zeros((K,D))
  for k in xrange(K):
    mask_k = Z == k
    CountVec[k] = np.sum(mask_k)
    DataStatVec[k] = np.sum(X[mask_k], axis=0)

  SS = bnpy.suffstats.SuffStatBag(K=K, D=D)
  SS.setField('CountVec', CountVec, dims=('K'))
  SS.setField('DataStatVec', DataStatVec, dims=('K','D'))
  return SS

class TestKMeans(unittest.TestCase):

  def setUp(self, N=1000, D=25, K=10):
    ''' Create a dataset X (2D array, N x D) and cluster means Mu (2D, KxD)
    '''
    self.X = rng.rand(N, D)
    self.Mu = rng.rand(K, D)

  
