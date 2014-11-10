'''
ToyHMT
'''
import scipy.linalg
import numpy as np
from bnpy.util.RandUtil import rotateCovMat
from bnpy.data.QuadTreeData import QuadTreeData
from bnpy.distr.GaussDistr import GaussDistr
from bnpy.allocmodel.tree import HMTUtil
import matplotlib.pyplot as plt

###########################################################  Set Toy Parameters
K = 4
D = 2

means = np.zeros( (K,D) )

V = 1.0/16.0
SigmaBase = np.asarray([[ V, 0], [0, V/100.0]])

# Create several Sigmas by rotating this basic covariance matrix
sigmas = np.zeros( (5,D,D) )
for k in xrange(4):
  sigmas[k] = rotateCovMat(SigmaBase, k*np.pi/4.0)

pi0 = np.asarray([.25,.25,.25,.25])
transition = np.zeros((4,4,4))
transition[0,:,:] = np.asarray([[.91,.03,.03,.03], [.91,.03,.03,.03], [.91,.03,.03,.03], [.91,.03,.03,.03]])
transition[1,:,:] = np.asarray([[.03,.91,.03,.03], [.03,.91,.03,.03], [.03,.91,.03,.03], [.03,.91,.03,.03]])
transition[2,:,:] = np.asarray([[.03,.03,.91,.03], [.03,.03,.91,.03], [.03,.03,.91,.03], [.03,.03,.91,.03]])
transition[3,:,:] = np.asarray([[.03,.03,.03,.91], [.03,.03,.03,.91], [.03,.03,.03,.91], [.03,.03,.03,.91]])

def sampleFromGaussian(state):
    return np.random.multivariate_normal(means[state,:], sigmas[state,:,:])

def generateObservations(seed,nObsTotal):
    PRNG = np.random.RandomState(seed)
    stateList = list()
    observationList = list()
    initialState = np.nonzero(PRNG.multinomial(1,pi0))[0][0]
    stateList.append(initialState)
    observationList.append(sampleFromGaussian(initialState))
    totalNonleafNodes = 0
    obs = nObsTotal/4
    while obs > 0:
        totalNonleafNodes += obs
        obs /= 4
    for i in xrange(totalNonleafNodes): #for each non-leaf node
        for j in xrange(4): # generate 4 children
            trans = PRNG.multinomial(1, transition[j][stateList[i]][:])
            state = np.nonzero(trans)[0][0]
            stateList.append(state)
            observationList.append(sampleFromGaussian(state))
    observationList = np.vstack(observationList)
    stateList = np.hstack(stateList)
    return observationList, stateList, totalNonleafNodes+nObsTotal

def get_data_info():
    return 'Simple HMT data with %d-D Gaussian observations with total states of K=%d' % (D,K)


def get_data(seed=8675309, nObsTotal=256, **kwargs):
    X, Z, totalNodes = generateObservations(seed, nObsTotal)
    l = list()
    l.append(totalNodes-1)
    trueParams = dict(initPi=pi0, transPi=transition, mu=means, Sigma=sigmas)
    Data = QuadTreeData(X=X, TrueZ=Z, nTrees=1, tree_delims=l, TrueParams=trueParams)
    plt.scatter(X[:,0], X[:,1], c=Z, alpha=.7)
    plt.show()
    Data.summary = get_data_info()
    return Data

def get_short_name( ):
    ''' Return short string used in filepaths to store solutions
    '''
    return 'ToyHMT'