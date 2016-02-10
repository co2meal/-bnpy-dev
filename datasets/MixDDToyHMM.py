'''
MixDDToyHMM: Diagonally-dominant toy HMM dataset with mixture emissions 

'''
import numpy as np
from bnpy.data import GroupXData
from bnpy.viz import GaussViz


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
    Y = np.asarray(fullY)
    Z = np.asarray(fullZ)

    nUsedStates = len(np.unique(Z))
    if nUsedStates < K:
        print 'WARNING: NOT ALL TRUE STATES USED IN GENERATED DATA'

    Data = GroupXData(X=X, doc_range=doc_range, TrueZ=Z)
    Data.name = get_short_name()
    Data.summary = get_data_info()
    return Data


def get_short_name():
    return 'MixDDToyHMM'


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

# Construct cluster centers by sampling without 
# replacement from a grid (using fixed seed)
import random
from random import randint
random.seed(10)

def getcoords(m=-6, n=7, sf=5):
    seen = set()
    x, y = sf*randint(m, n), sf*randint(m, n)
    seen.add((x, y))

    while True:
        yield (x, y)
        x, y = sf*randint(m, n), sf*randint(m, n)
        while (x, y) in seen:
            x, y = sf*randint(m, n), sf*randint(m, n)
g = getcoords()
mus = np.zeros((K,C,2))
for k in range(K):
    for c in range(C):
        (a,b) = next(g)
        mus[k,c,0] = a
        mus[k,c,1] = b


# Covariance for each component
# set to the 2x2 identity matrix
sigmas = np.tile(np.eye(2), (K, 1, 1))


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


if __name__ == '__main__':
    illustrate()
    pylab.savefig('DatasetIllustration-MixDDToyHMM.eps', bbox_inches='tight',
                  pad_inches=0)
    pylab.show(block=True)