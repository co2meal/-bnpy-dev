'''
SeqOfBinBars9x9.py

Binary toy bars data, with a 9x9 grid,
so each observation is a vector of size 81.

There are K=20 true topics
* one common background topic (with prob of 0.05 for all pixels)
* one rare foreground topic (with prob of 0.90 for all pixels)
* 18 bar topics, one for each row/col of the grid.
'''
import numpy as np
from bnpy.data import GroupXData
from bnpy.util import as1D

K = 20 # Number of topics
D = 81 # Vocabulary Size

Defaults = dict()
Defaults['nDocTotal'] = 12
Defaults['T'] = 500
Defaults['bgProb'] = 0.05
Defaults['fgProb'] = 0.90
Defaults['seed'] = 8675309

def get_data(**kwargs):
    ''' Create dataset as bnpy DataObj object.
    '''
    Data = generateDataset(**kwargs)
    Data.name = 'SeqOfBinBars9x9'
    Data.summary = 'Binary Bar Sequences with %d true topics.' % (K)
    return Data


def makePi(stickyProb=0.96,  nextProb=0.03, extraStickyProb=0.995,
        **kwargs):
    ''' Make phi matrix that defines probability of each pixel.
    '''
    bgProb = (1 - stickyProb - nextProb) / (K-2)
    pi = bgProb * np.ones((K, K))
    for k in xrange(18):
        if k < 9:
            # Horizontal bars
            pi[k, k] = stickyProb
            pi[k, (k+1) % 9] = nextProb
        else:
            pi[k, k] = stickyProb
            pi[k, 9 + (k+1) % 9] = nextProb
    pi[-2, :] = (1 - extraStickyProb) / (K-1)
    pi[-2, -2] = extraStickyProb
    pi[-1, :] = (1 - stickyProb) / (K-1)
    pi[-1, -1] = stickyProb
    return pi

def makePhi(fgProb=0.75, bgProb=0.05, **kwargs):
    ''' Make phi matrix that defines probability of each pixel.
    '''
    phi = bgProb * np.ones((K, np.sqrt(D), np.sqrt(D)))
    for k in xrange(18):
        if k < 9:
            rowID = k
            # Horizontal bars
            phi[k, rowID, :] = fgProb
        else:
            colID = k - 9
            phi[k, :, colID] = fgProb
    phi[-2, :, :] = bgProb
    phi[-1, :, :] = fgProb
    phi = np.reshape(phi, (K, D))
    return phi


def generateDataset(**kwargs):
    for key in Defaults:
        if key not in kwargs:
            kwargs[key] = Defaults[key]
    phi = makePhi(**kwargs)
    transPi = makePi(**kwargs)
    PRNG = np.random.RandomState(kwargs['seed'])

    nSeq = kwargs['nDocTotal']
    T = kwargs['T']
    seqLens = T * np.ones(nSeq, dtype=np.int32)
    doc_range = np.hstack([0, np.cumsum(seqLens)])
    N = doc_range[-1]
    allX = np.zeros((N,D))
    allZ = np.zeros(N, dtype=np.int32)
    
    startStates = [0, 9, 18, 19]
    states0toKm1 = np.arange(K)
    # Each iteration generates one time-series/sequence
    # with starting state deterministically rotating among all states
    for i in xrange(nSeq):
        start = doc_range[i]
        stop = doc_range[i+1]

        T = stop - start
        Z = np.zeros(T)
        X = np.zeros((T,D))
        nConsec = 0
        for t in xrange(T):
            if t == 0:
                Z[0] = startStates[i % len(startStates)]
            else:
                transPi_t = transPi[Z[t-1]].copy()
                if nConsec > T/4:
                  transPi_t[Z[t-1]] = 0
                  transPi_t /= transPi_t.sum()
                Z[t] = PRNG.choice(states0toKm1, p=transPi_t)
            X[t] = PRNG.rand(D) < phi[Z[t]]
            if Z[t] == Z[t-1]:
              nConsec += 1
            else:
              nConsec = 0

        allZ[start:stop] = Z
        allX[start:stop] = X

    TrueParams = dict()
    TrueParams['beta'] = np.mean(transPi, axis=0)
    TrueParams['phi'] = phi
    TrueParams['Z'] = allZ
    return GroupXData(allX, doc_range=doc_range, TrueParams=TrueParams)

if __name__ == '__main__':
    import bnpy.viz.BernViz as BernViz
    Data = get_data(nDocTotal=50)
    for k in xrange(20):
        print 'N[%d] = %d' % (k, np.sum(Data.TrueParams['Z'] == k))

    BernViz.plotCompsAsSquareImages(Data.TrueParams['phi'])
    BernViz.pylab.show(block=True)
    # startStates = [0, 9, 18, 19]
    # for i in xrange(4):
    #     start = Data.doc_range[i]
    #     k = startStates[i]
    #     print 'Showing 4 examples from cluster %d' % (k)
    #     BernViz.plotDataAsSquareImages(
    #         Data, unitIDsToPlot=np.arange(start, start+4), doShowNow=1)
