'''SeqHMMK4.py

Multiple sequences of data  generated from a HMM with gaussian emission 
probabilities.
'''

import numpy as np
#from bnpy.data.SeqXData import SeqXData
from bnpy.data import SeqXData, MinibatchIterator

import scipy.io


##################################################### Set Parameters
K = 4
D = 2

#transPi = np.asarray([[0.2, 0.2, 0.2, 0.4], \
#                      [0.2, 0.6, 0.1, 0.1], \
#                      [0.2, 0.2, 0.3, 0.3], \
#                      [0.2, 0.2, 0.2, 0.4], \
#                      [1.0, 0.0, 0.0, 0.0]])

transPi = np.asarray([[0.0, 1.0, 0.0, 0.0], \
                      [0.0, 0.0, 1.0, 0.0], \
                      [0.0, 0.0, 0.0, 1.0], \
                      [1.0, 0.0, 0.0, 0.0]])
#transPi = np.identity(D)

initState = 1

mus = np.asarray([[0, 0], \
                  [100, 0], \
                  [0, 100], \
                  [10, 100]])
#mus = np.zeros((K,D))
#for i in xrange(K):
#  mus[i,i] = 10


sigmas = np.empty((K,D,D))
sigmas[0,:,:] = np.asarray([[2, 0], [0, 2]])
sigmas[1,:,:] = np.asarray([[2, 0], [0, 2]])
sigmas[2,:,:] = np.asarray([[2, 0], [0, 2]])
sigmas[3,:,:] = np.asarray([[2, 0], [0, 2]])
#for i in xrange(K):
#  sigmas[i,:,:] = 2 * np.identity(D)

def get_minibatch_iterator(seed=8675309, dataorderseed=0, nBatch=10, nObsBatch=None, nObsTotal=25000, nLap=1, startLap=0, **kwargs):
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
  X, TrueZ, seqInds = get_X(seed, ((6000, 6000, 6000, 6000, 1000)))
  Data = SeqXData(X = X, TrueZ = TrueZ, seqInds = seqInds)
  Data.summary = get_data_info()
  DataIterator = MinibatchIterator(Data, nBatch=nBatch, nObsBatch=nObsBatch, nLap=nLap, startLap=startLap, dataorderseed=dataorderseed)
  return DataIterator



def get_X(seed, seqLens): 
    prng = np.random.RandomState(seed)

    fullX = list()
    seqIndicies = list([0])
    fullZ = list()    
    seqLens = list(seqLens)
    
    if len(np.shape(seqLens)) == 0:
        rang = xrange(1)
    else:
        rang = xrange(len(seqLens))

    for i in rang:
        Z = list()
        X = list()
        Z.append(initState)
        X.append(sample_from_state(Z[0], prng))
        for j in xrange(seqLens[i]-1):
            trans = prng.multinomial(1, transPi[Z[j]])
            nextState = np.nonzero(trans)[0][0]
            Z.append(nextState)
            X.append(sample_from_state(Z[j+1], prng))

        fullZ = np.append(fullZ, Z)
        fullX.append(X)

        seqIndicies.append(seqLens[i] + seqIndicies[i])
        
    return np.vstack(fullX), np.asarray(fullZ), np.asarray(seqIndicies)


def sample_from_state(k, prng):
    return np.random.multivariate_normal(mus[k,:], sigmas[k,:,:])

def get_data_info():
    return 'Multiple sequences of simple HMM data with %d-D Gaussian observations and K=%d' % (D,K)

def get_short_name():
    return 'SeqHMMK4'

def get_data(seed=8675309, seqLens=((6000,6000,6000,6000,1000)), **kwargs):
    fullX, fullZ, seqIndicies = get_X(seed, seqLens)
    X = np.vstack(fullX)
    Z = np.asarray(fullZ)
    inds = np.asarray(seqIndicies)
    
    Data = SeqXData(X=X, seqInds = inds, nObsTotal = np.sum(inds), TrueZ = Z)
    Data.summary = get_data_info()
    scipy.io.savemat('trueZ.mat', {'trueZ':Z})
    return Data
