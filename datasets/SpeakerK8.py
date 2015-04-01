'''
Multiple sequences of data  generated from a HMM with gaussian emission 
probabilities.
'''

import numpy as np
from bnpy.data import SeqXData

import scipy.io


##################################################### Set Parameters
K = 8
D = 2


transPi = np.asarray([[0.993, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001], \
                      [0.001, 0.993, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001], \
                      [0.001, 0.001, 0.993, 0.001, 0.001, 0.001, 0.001, 0.001], \
                      [0.001, 0.001, 0.001, 0.993, 0.001, 0.001, 0.001, 0.001], \
                      [0.001, 0.001, 0.001, 0.001, 0.993, 0.001, 0.001, 0.001], \
                      [0.001, 0.001, 0.001, 0.001, 0.001, 0.993, 0.001, 0.001], \
                      [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.993, 0.001], \
                      [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.993]])

initState = 1

mus = np.asarray([[0, 1], \
                  [0, 2], \
                  [1, 4], \
                  [1, 3], \
                  [3, 2], \
                  [0, 0], \
                  [6, 2], \
                  [4, 4]])

sigmas = np.empty((K,D,D))
sigmas[0,:,:] = np.asarray([[5, 0], [0, 7]])
sigmas[1,:,:] = np.asarray([[1, 0], [0, 1]])
sigmas[2,:,:] = np.asarray([[3, 0], [0, 5]])
sigmas[3,:,:] = np.asarray([[3, 0], [0, 3]])
sigmas[4,:,:] = np.asarray([[0.5, 0], [0, 1]])
sigmas[5,:,:] = np.asarray([[4, 0], [0, 2]])
sigmas[6,:,:] = np.asarray([[1, 0], [0, 3]])
sigmas[7,:,:] = np.asarray([[1, 0], [0, 1.2]])

# def get_minibatch_iterator(seed=8675309, dataorderseed=0, nBatch=10, nObsBatch=None, nObsTotal=25000, nLap=1, startLap=0, **kwargs):
#   '''
#     Args
#     --------
#     seed : integer seed for random number generator,
#             used for actually *generating* the data
#     dataorderseed : integer seed that determines
#                      (a) how data is divided into minibatches
#                      (b) order these minibatches are traversed

#    Returns
#     -------
#       bnpy MinibatchIterator object, with nObsTotal observations
#         divided into nBatch batches
#   '''
#   X, TrueZ, seqInds = get_X(seed, ((600, 600, 600, 600, 100)))
#   Data = SeqXData(X = X, TrueZ = TrueZ, seqInds = seqInds)
#   Data.summary = get_data_info()
#   DataIterator = MinibatchIterator(Data, nBatch=nBatch, nObsBatch=nObsBatch, nLap=nLap, startLap=startLap, dataorderseed=dataorderseed)
#   return DataIterator



def get_X(seed, seqLens):
    '''
    Generates X, Z, seqInds according to the gaussian parameters specified above
      and the sequence lengths passed in.
    '''
    prng = np.random.RandomState(seed)

    fullX = list()
    seqIndicies = list([0])
    fullZ = list()    
    seqLens = list(seqLens)
    
    if len(np.shape(seqLens)) == 0:
        rang = xrange(1)
    else:
        rang = xrange(len(seqLens))

    #Each iteration generates one sequence
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

def get_data(seed=8675309, seqLens=((3000,3000,3000,3000,500)), **kwargs):
  fullX, fullZ, seqIndicies = get_X(seed, seqLens)
  X = np.vstack(fullX)
  Z = np.asarray(fullZ)
  inds = np.asarray(seqIndicies)
    
  Data = SeqXData(X=X, seqInds = inds, nObsTotal = np.sum(seqLens), TrueZ = Z)
  Data.summary = get_data_info()

  return Data
