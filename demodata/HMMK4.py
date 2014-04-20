'''
HMMK5.py

Dataset generated from a HMM with gaussian emission probabilities.
'''

import numpy as np
from bnpy.data import XData

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

initState = 1

#mus = np.asarray([[0, 0], \
#                  [0, 10], \
#                  [10, 0], \
#                  [10, 10]])

#sigmas = np.empty((4,2,2))
#sigmas[0,:,:] = np.asarray([[400, 0], [0, 400]])
#sigmas[1,:,:] = np.asarray([[400, 0], [0, 400]])
#sigmas[2,:,:] = np.asarray([[400, 0], [0, 400]])
#sigmas[3,:,:] = np.asarray([[400, 0], [0, 400]])

mus = np.asarray([[0, 0], \
                  [0, 1], \
                  [1, 0], \
                  [1, 1]])

sigmas = np.empty((4,2,2))
sigmas[0,:,:] = np.asarray([[4, 0], [0, 4]])
sigmas[1,:,:] = np.asarray([[4, 0], [0, 4]])
sigmas[2,:,:] = np.asarray([[4, 0], [0, 4]])
sigmas[3,:,:] = np.asarray([[4, 0], [0, 4]])


def get_X(seed, nObsTotal):
    prng = np.random.RandomState(seed)
    Z = list()
    X = list()
    Z.append(initState)
    X.append(sample_from_state(Z[0], prng))

    for i in xrange(nObsTotal-1):
        trans = prng.multinomial(1, transPi[Z[i]])
        nextState = np.nonzero(trans)[0][0]
        Z.append(nextState)
        X.append(sample_from_state(Z[i+1], prng))

    Z = np.asarray(Z)
    X = np.vstack(X)
    print 'X = ', X[0:10]
    print 'Z = ', Z[0:10]
    return X, Z


def sample_from_state(k, prng):
    return np.random.multivariate_normal(mus[k,:], sigmas[k,:,:])

def get_data_info():
    return 'Simple HMM data with %d-D Gaussian observations and K=%d' % (D,K)

def get_short_name():
    return 'HMMK4'

def get_data(seed=8675309, nObsTotal=25000, **kwargs):
    X, Z = get_X(seed, nObsTotal)
    Data = XData(X=X, TrueZ=Z)
    Data.summary = get_data_info()
    return Data
