'''
MixDDToyHMM: Diagonally-dominant toy HMM dataset with mixture emissions 

'''
import numpy as np
import bnpy
from bnpy.data import GroupXData
from bnpy.viz import GaussViz

from matplotlib import pylab
rcParams = pylab.rcParams
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['text.usetex'] = False
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
rcParams['legend.fontsize'] = 25


def get_data(seed=123, nDocTotal=32, T=1000,
             **kwargs):
    ''' Generate several data sequences, returned as a bnpy data-object

    Args
    -------
    seed : integer seed for random number generator,
          used for actually *generating* the data
    seqLens : total number of observations in each sequence

    Returns
    -------
    Data : bnpy GroupXData object, with nObsTotal observations
    '''
    fullX, fullY, fullZ, doc_range = get_X(seed, T, nDocTotal)
    X = np.vstack(fullX)
    Y = np.asarray(fullY, dtype=np.int32)
    Z = np.asarray(fullZ, dtype=np.int32)

    nUsedStates = len(np.unique(Z))
    if nUsedStates < K:
        print 'WARNING: NOT ALL TRUE STATES USED IN GENERATED DATA'

    Data = GroupXData(X=X, doc_range=doc_range, TrueParams={'Y':Y, 'Z':Z})
    Data.name = get_short_name()
    Data.summary = get_data_info()
    return Data


def get_short_name():
    return 'MixDDToyHMMSpatial'


def get_data_info():
    return 'Toy HMM data with diagonally-dominant transition matrix and mixture emissions.'

D = 2
K = 8
C = 3
initPi = 1.0 / K * np.ones(K)
transPi = np.asarray([
    [.99, .01, 0, 0, 0, 0, 0, 0],
    [0, .99, .01, 0, 0, 0, 0, 0],
    [0, 0, .99, .01, 0, 0, 0, 0],
    [0, 0, 0, .99, .01, 0, 0, 0],
    [0, 0, 0, 0, .99, .01, 0, 0],
    [0, 0, 0, 0, 0, .99, .01, 0],
    [0, 0, 0, 0, 0, 0, .99, .01],
    [.01, 0, 0, 0, 0, 0, 0, .99],
])

# Means for each component
mus = np.asarray([
    [[0, 20], [5, 20], [0, 15]],          
    [[20, 0], [15, 0], [20, -5]],         
    [[-30, -30], [-25, -30], [-30, -25]], 
    [[30, -30],  [25, -30],  [30, -25]],  
    [[-20, 0],   [-15, 0],   [-20, -5]],
    [[0, -20],   [-5, -20],   [0, -25]],
    [[30, 30],   [35, 30],   [30, 35]],
    [[-30, 30],  [-35, 30],  [-30, 35]]
])

# Covariance for each component
# set to the 2x2 identity matrix
sigmas = np.tile(np.array([[3,2],[2,3]]), (K, 1, 1)) #np.eye(2)


def get_X(seed, T, nDocTotal):
    ''' Generates X, Z, seqInds
    '''
    T = int(T)
    nDocTotal = int(nDocTotal)

    prng = np.random.RandomState(seed)

    fullX = list()
    fullY = list() #
    fullZ = list()
    doc_range = np.zeros(nDocTotal + 1, dtype=np.int32)

    # Each iteration generates one time-series/sequence
    # with starting state deterministically rotating among all states
    for i in xrange(nDocTotal):
        Z = list()
        Y = list() 
        X = list()
        initState = i % K
        initMode  = i % C 
        initX = prng.multivariate_normal(list(mus[initState, initMode]), sigmas[initState, :, :])
        Z.append(initState)
        Y.append(initMode) 
        X.append(initX)
        for j in xrange(T - 1):
            nextState = prng.choice(xrange(K), p=transPi[Z[j]])
            nextMode  = prng.choice(xrange(C)) 
            nextX = prng.multivariate_normal(list(mus[nextState, nextMode]), sigmas[nextState, :, :])
            Z.append(nextState)
            Y.append(nextMode) 
            X.append(nextX)

        fullZ = np.hstack([fullZ, Z])
        fullY = np.hstack([fullY, Y])
        fullX.append(X)
        doc_range[i + 1] = doc_range[i] + T

    return (np.vstack(fullX),
            np.asarray(fullY, dtype=np.int32).flatten(), 
            np.asarray(fullZ, dtype=np.int32).flatten(),
            doc_range,
            )

Colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c',
          '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00']


def illustrate(Colors=Colors):
    if hasattr(Colors, 'colors'):
        Colors = Colors.colors

    from matplotlib import pylab
    rcParams = pylab.rcParams
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams['text.usetex'] = False
    rcParams['xtick.labelsize'] = 20
    rcParams['ytick.labelsize'] = 20
    rcParams['legend.fontsize'] = 25

    import bnpy

    Data = get_data(T=1000, nDocTotal=8)
    for k in xrange(K):
        zmask = Data.TrueParams['Z'] == k
        pylab.plot(Data.X[zmask, 0], Data.X[zmask, 1], '.', color=Colors[k],
                   markeredgecolor=Colors[k],
                   alpha=0.4)

        sigEdges = np.flatnonzero(transPi[k] > 0.0001)
        for j in sigEdges:
            if j == k:
                continue
            dx = mus[j, 0, 0] - mus[k, 0, 0]
            dy = mus[j, 0, 1] - mus[k, 0, 1]
            pylab.arrow(mus[k, 0, 0], mus[k, 0, 1],
                        0.9 * dx,
                        0.9 * dy,
                        head_width=2, head_length=4,
                        facecolor=Colors[k], edgecolor=Colors[k])

            tx = 0 - mus[k, 0, 0]
            ty = 0 - mus[k, 0, 1]
            xy = (mus[k, 0, 0] - 0.2 * tx, mus[k, 0, 1] - 0.2 * ty)
            '''
            pylab.annotate( u'\u27F2',
                      xy=(mus[k,0], mus[k,1]),
                     color=Colors[k],
                     fontsize=35,
                    )
            '''
            pylab.gca().yaxis.set_ticks_position('left')
            pylab.gca().xaxis.set_ticks_position('bottom')

            pylab.axis('image')
            pylab.ylim([-38, 38])
            pylab.xlim([-38, 38])


def plotDataWithTrueLabelColors(Data):
    for k in xrange(K):
        zmask = Data.TrueParams['Z'] == k
        pylab.plot(Data.X[zmask, 0], Data.X[zmask, 1], '.', color=Colors[k],
                   markeredgecolor=Colors[k],
                   alpha=0.4) #0.4
        ''' DEPRECATED CODE TO PLOT TRANSITION EDGES
        sigEdges = np.flatnonzero(transPi[k] > 0.0001)
        for j in sigEdges:
            if j == k:
                continue
            dx = mus[j, 0, 0] - mus[k, 0, 0]
            dy = mus[j, 0, 1] - mus[k, 0, 1]
            pylab.arrow(mus[k, 0, 0], mus[k, 0, 1],
                        0.9 * dx,
                        0.9 * dy,
                        head_width=2, head_length=4,
                        facecolor=Colors[k], edgecolor=Colors[k])

            tx = 0 - mus[k, 0, 0]
            ty = 0 - mus[k, 0, 1]
            xy = (mus[k, 0, 0] - 0.2 * tx, mus[k, 0, 1] - 0.2 * ty)
            pylab.gca().yaxis.set_ticks_position('left')
            pylab.gca().xaxis.set_ticks_position('bottom')

            pylab.axis('image')
            pylab.ylim([-38, 38])
            pylab.xlim([-38, 38])
        '''
    pylab.gca().yaxis.set_ticks_position('left')
    pylab.gca().xaxis.set_ticks_position('bottom')
    pylab.axis('image')
    pylab.ylim([-38, 38])
    pylab.xlim([-38, 38])

def plotCompsForHModel(hmodel, titleStr=None):
    pylab.figure()
    for k in xrange(K):
        for c in xrange(C):
            meanVec_kc = hmodel.obsModel.Post.m[k,c]
            CovMat_kc = hmodel.obsModel.Post.B[k,c] / \
                hmodel.obsModel.Post.nu[k,c]
            pylab.plot(meanVec_kc[0], meanVec_kc[1], 'kx')
            GaussViz.plotGauss2DContour(
                meanVec_kc,
                CovMat_kc,
                color=Colors[k],
                radiusLengths=[.5,2])
    if titleStr:
        pylab.title(titleStr)
    pylab.gca().yaxis.set_ticks_position('left')
    pylab.gca().xaxis.set_ticks_position('bottom')
    pylab.axis('image')
    pylab.ylim([-38, 38])
    pylab.xlim([-38, 38])

def test_InitFromTruthAndPlotCompsAtEachStep(Data, nIters=3):
    ''' Verify that inference works as expected from true initialization.

    Procedure
    ---------
    1) Create an HModel 
    2) Initialize it with "grouth truth" params for given dataset
    3) Run several steps of inference (local/global) forward
    4) Plot the learned substate clusters at each step

    Post Condition
    --------------
    Plots will be created of learned comps at each step
    '''
    N = Data.TrueParams['Z'].size

    # Create complete model, with HMM allocations and MixGauss observations
    hmodel = bnpy.HModel.CreateEntireModel(
        'VB', 'FiniteHMM', 'MixGaussObsModel',
        dict(gamma=10.0, alpha=0.5, startAlpha=5.0),
        dict(C=C, ECovMat=np.eye(2), sF=1.0),
        Data)

    # Create "ground truth" local parameter dict LP
    substate_resp = 1e-100 * np.ones((N,K,C))
    resp = 1e-100 * np.ones((N,K))
    for n in range(N):
        resp[n, Data.TrueParams['Z'][n]] = 1.0
        substate_resp[n, Data.TrueParams['Z'][n], Data.TrueParams['Y'][n]] = 1.
    trueLP = dict(
        substate_resp=substate_resp,
        resp=resp)
    # Expand the fields of the trueLP to include things needed for the HMM
    # Like the transition counts
    trueLP = hmodel.allocModel.initLPFromResp(Data, trueLP)

    # Do summary step and global step
    trueSS = hmodel.get_global_suff_stats(Data, trueLP)
    hmodel.update_global_params(trueSS)

    plotCompsForHModel(hmodel, 'From Truth')
    # Do several iterations of local/global steps
    for i in range(nIters):
        LP = hmodel.calc_local_params(Data)
        SS = hmodel.get_global_suff_stats(Data, LP)
        hmodel.update_global_params(SS)
        plotCompsForHModel(hmodel, 'From Truth + %d iter' % (i+1))

if __name__ == '__main__':
    Data = get_data(T=5000, nDocTotal=1)
    # Illustrate the raw data
    plotDataWithTrueLabelColors(Data)
    # Illustrate model at each step
    test_InitFromTruthAndPlotCompsAtEachStep(Data, nIters=3)
    pylab.show(block=True)
