'''
ToyHMT
'''
import scipy.linalg
import numpy as np
from bnpy.util.RandUtil import rotateCovMat
from bnpy.data import GroupXData

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

def generateObservations( seed, nDoc, nObsPerDoc ):
    PRNG = np.random.RandomState(seed)
    stateList = list()
    observationList = list()
    totalNonleafNodes = 1
    height = 1
    while nObsPerDoc-totalNonleafNodes > 4**height:
        totalNonleafNodes += 4**height
        height += 1
    for d in xrange(nDoc):
        initialState = np.nonzero(PRNG.multinomial(1,pi0))[0][0]
        stateList.append(initialState)
        observationList.append(sampleFromGaussian(initialState))
        for i in xrange(totalNonleafNodes): #for each non-leaf node
            for j in xrange(4): # generate 4 children
                trans = PRNG.multinomial(1, transition[j][stateList[i]][:])
                state = np.nonzero(trans)[0][0]
                stateList.append(state)
                observationList.append(sampleFromGaussian(state))
    doc_range = np.arange(0, nDoc*nObsPerDoc+1, nObsPerDoc)
    observationList = np.vstack(observationList)
    stateList = np.hstack(stateList)
    return GroupXData(observationList, doc_range, TrueZ=stateList)

def get_data_info():
    return 'Simple HMT data with %d-D Gaussian observations with total states of K=%d' % (D,K)


def get_data(seed=8675309, nDocTotal=10, nObsPerDoc=341, **kwargs):
    Data = generateObservations(seed, nDocTotal, nObsPerDoc)
    Data.name = 'ToyHMT'
    Data.summary = get_data_info()
    return Data

def get_short_name( ):
    ''' Return short string used in filepaths to store solutions
    '''
    return 'ToyHMT'

def plot_true_clusters():
    from bnpy.viz import GaussViz
    for k in range(K):
        c = k % len(GaussViz.Colors)
        GaussViz.plotGauss2DContour(means[k], sigmas[k], color=GaussViz.Colors[c])

if __name__ == "__main__":
    from matplotlib import pylab
    pylab.figure()
    plot_true_clusters()
    pylab.show(block=True)