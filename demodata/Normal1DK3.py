'''
Normal1DK3

Simple toy dataset of 3 comps from 
  1D normal distributions with different means
'''

import numpy as np

from bnpy.data import XData, MinibatchIterator

########################################################### User-facing fcns
###########################################################
def get_data(seed=8675309, nObsTotal=25000, **kwargs):
  '''
    Args
    -------
    seed : integer seed for random number generator,
            used for actually *generating* the data
    nObsTotal : total number of observations for the dataset.

    Returns
    -------
      Data : bnpy XData object, with nObsTotal observations
  '''
  X, TrueZ = generate_data(seed, nObsTotal)
  Data = XData(X=X, TrueZ=TrueZ)
  Data.summary = get_data_info()
  return Data
  
def get_minibatch_iterator(seed=8675309, dataorderseed=0, nBatch=10, 
                           nObsTotal=25000, nLap=1, startLap=0, **kwargs):
  '''
    Args
    --------
    seed : integer seed for random number generator,
            used for actually *generating* the data
    dataorderseed : integer seed that determines
                     (a) how data is divided into minibatches
                     (b) order these minibatches are traversed

   Returns
    -------
      bnpy MinibatchIterator object, with nObsTotal observations
        divided into nBatch batches
  '''
  X, TrueZ = generate_data(seed, nObsTotal)
  Data = XData(X=X)
  Data.summary = get_data_info()
  DataIterator = MinibatchIterator(Data, nBatch=nBatch, nObsBatch=None,
                                   nLap=nLap, startLap=startLap,
                                   dataorderseed=dataorderseed)
  return DataIterator

def get_data_info():
  return 'Normal Data. Ktrue=3. D=1.'

########################################################### Generate Raw Data
###########################################################
mu = np.asarray([-10, 0, 10])

def generate_data(seed, nObsTotal):
  PRNG = np.random.RandomState(seed)
  K = mu.size
  N = nObsTotal / K
  Xlist = list()
  Zlist = list()
  for k in xrange(K):
    if k == K-1:
      N = nObsTotal - (K-1)*N
    X = mu[k] + PRNG.randn(N, 1)
    Xlist.append(X)
    Zlist.append(k * np.ones(N))
  X = np.vstack(Xlist)
  TrueZ = np.hstack(Zlist)

  permIDs = PRNG.permutation(X.shape[0])
  X = X[permIDs]
  TrueZ = TrueZ[permIDs]
  return X, TrueZ