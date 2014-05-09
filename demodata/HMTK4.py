'''
HMTK4.py

A module to create HMT data with 4 states
'''
import numpy as np
from bnpy.data.QuadTreeData import QuadTreeData
from bnpy.distr.GaussDistr import GaussDistr
from bnpy.allocmodel.tree import HMTUtil

K = 4
D = 2

# mean vectors for Gaussian emission probabilities
means = np.asarray([[2,2], [-2,-2], [2,-2], [-2,2]])

# covariance matrices for Gaussian emission probabilities
sigmas = np.zeros((4,2,2))
sigmas[0,:,:] = np.asarray([[1,.5], [.5,1]])
sigmas[1,:,:] = np.asarray([[1,.5], [.5,1]])
sigmas[2,:,:] = np.asarray([[1,-.5], [-.5,1]])
sigmas[3,:,:] = np.asarray([[1,-.5], [-.5,1]])

# transition matrices for each direction, default by 4
transition = np.zeros((4,4,4))
transition[0,:,:] = np.asarray([[.1,.4,.3,.2], [.2,.2,.5,.1], [.5,.2,.1,.2], [.2,.1,.2,.5]])
transition[1,:,:] = np.asarray([[.2,.1,.4,.3], [.1,.2,.2,.5], [.2,.5,.2,.1], [.5,.2,.1,.2]])
transition[2,:,:] = np.asarray([[.3,.2,.1,.4], [.5,.1,.2,.2], [.1,.2,.5,.2], [.2,.5,.2,.1]])
transition[3,:,:] = np.asarray([[.4,.3,.2,.1], [.2,.5,.1,.2], [.2,.1,.2,.5], [.1,.2,.5,.2]])

# initial state
pi0 = np.asarray([.25,.25,.25,.25])

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
    '''comp = list()
    I = np.eye(2)
    for b in range(K):
        precMat = np.linalg.solve( sigmas[b], I )
        comp.append(GaussDistr(m=means[b], L=precMat))
    lpr = np.zeros( (341, K) )
    for k in range(K):
      lpr[:,k] = comp[k].log_pdf( QuadTreeData(X) )
    resp, respPair, logMargPrSeq = HMTUtil.SumProductAlg_QuadTree(pi0, transition, lpr)'''
    trueParams = dict(initPi=pi0, transPi=transition, mu=means, Sigma=sigmas)
    Data = QuadTreeData(X=X, TrueZ=Z, nTrees=1, tree_delims=l, TrueParams=trueParams)
    Data.summary = get_data_info()
    return Data

def get_short_name( ):
    ''' Return short string used in filepaths to store solutions
    '''
    return 'HMTK4'