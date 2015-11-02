import numpy as np
from bnpy.data import XData
import DeadLeavesD25, StarCovarK5
from bnpy.suffstats import ParamBag, SuffStatBag
from bnpy.obsmodel import ZeroMeanFactorAnalyzerObsModel
from numpy.linalg import inv, solve, det, slogdet, eig, LinAlgError
from scipy.linalg import eigh
from scipy.special import psi, gammaln
from bnpy.util import dotATA
from bnpy.util import LOGTWOPI, EPS
import matplotlib.pylab as plt


C = 1
K = 1

Data = DeadLeavesD25.get_data()
Data = StarCovarK5.get_data()
X = Data.X
D = Data.X.shape[1]

k = 3
X1 = Data.X[Data.TrueParams['Z'] == k]

s = 10
t = 1
f = 100
g = 1

nTrial = 30
nIter = 30

def eigInit(xxT, N):
    eigVal, eigVec = eigh(xxT / N, eigvals=(D-C,D-1))
    sigma2 = (np.trace(xxT) / N - np.sum(eigVal)) / (D-C)
    if sigma2 <= EPS or not np.all(eigVal - sigma2 >= EPS):
        assert np.allclose(sigma2, 0)
        assert np.allclose(eigVal[eigVal<EPS], 0)
        assert np.allclose((eigVal-sigma2)[eigVal-sigma2<EPS], 0)
        sigma2 = EPS
        eigVal[eigVal<EPS] = EPS
    WMean = np.dot(eigVec[:,:C], np.diag(np.sqrt(eigVal - sigma2)))
    hShape = f + .5 * D
    hInvScale = g * np.ones(C)
    PhiShape = s + .5 * N
    PhiInvScale = sigma2 * PhiShape * np.ones(D)
    E_WT_Phi_W = np.sum((PhiShape / PhiInvScale)[:, np.newaxis, np.newaxis]
                        * np.einsum('ij,ik->ijk', WMean, WMean), axis=0)
    aCov = inv(np.eye(C) + E_WT_Phi_W)
    return WMean, hShape, hInvScale, PhiShape, PhiInvScale, aCov

def randExampleInit(X1, seed=0):
    PRNG = np.random.RandomState(seed=seed)
    idx = PRNG.choice(X1.shape[0])
    xxT = np.outer(X1[idx], X1[idx])
    return eigInit(xxT=xxT, N=1.0)

def samplingInit(seed=0):
    PRNG = np.random.RandomState(seed=seed)
    h = PRNG.gamma(f, 1./g, C)
    hShape = f + .5 * D
    hInvScale = hShape / h
    assert not np.any(np.isclose(h, 0))
    sigma = 1./np.sqrt(h)
    WMean = np.zeros((D,C))
    for c in xrange(C):
        WMean[:,c] = PRNG.normal(0., sigma[c], D)
    Phi = PRNG.gamma(s, 1./t, D)
    PhiShape = s + .5 * X1.shape[0]
    PhiInvScale = PhiShape / Phi
    E_WT_Phi_W = np.sum((PhiShape / PhiInvScale)[:, np.newaxis, np.newaxis]
                        * np.einsum('ij,ik->ijk', WMean, WMean), axis=0)
    aCov = inv(np.eye(C) + E_WT_Phi_W)
    # aCov = np.eye(C)
    return WMean, hShape, hInvScale, PhiShape, PhiInvScale, aCov

def updatePost_eig(X):
    xxT = dotATA(X)
    N = X.shape[0]
    WMean, hShape, hInvScale, PhiShape, PhiInvScale, aCov = eigInit(xxT, N)
    WMean, WCov, hShape, hInvScale, PhiShape, PhiInvScale, aCov = \
            updateAll(X, WMean, hShape, hInvScale, PhiShape, PhiInvScale, aCov)
    return WMean, WCov, hShape, hInvScale, PhiShape, PhiInvScale, aCov

def updateAll(X, WMean, hShape, hInvScale, PhiShape, PhiInvScale, aCov):
    # get xaT, aaT
    N = X.shape[0]
    xxT = dotATA(X)
    LU = np.dot(aCov, np.dot(WMean.T, np.diag(PhiShape / PhiInvScale)))
    xaT = np.inner(xxT, LU)
    aaT = N * aCov + np.dot(LU, xaT)
    # calc W
    scaled_xaT = PhiShape / PhiInvScale[:,np.newaxis] * xaT
    WCov = 1. / (hShape / hInvScale
                 + (PhiShape / PhiInvScale)[:,np.newaxis]
                 * np.tile(np.diag(aaT),(D,1)))
    WMean = WCov * scaled_xaT
    # elboW = elbo4comp(SS, WMean, WCov, hShape, hInvScale, PhiShape, PhiInvScale, aCov)
    # calc H
    hShape = f + .5 * D
    hInvScale = g + .5 * np.sum(WMean**2 + WCov, axis=0)
    # elboH = elbo4comp(SS, WMean, WCov, hShape, hInvScale, PhiShape, PhiInvScale, aCov)
    # calc Phi
    PhiShape = s + .5 * N
    E_WT_W = np.einsum('ij,ik->ijk', WMean, WMean)
    E_WT_W += WCov[:,:,np.newaxis] * np.eye(C)
    PhiInvScale = t + .5 * (np.diag(xxT)
                  - 2 * np.einsum('ij,ij->i', xaT, WMean)
                  + np.einsum('ijk,...kj->i', E_WT_W, aaT))
    # elboPhi = elbo4comp(SS, WMean, WCov, hShape, hInvScale, PhiShape, PhiInvScale, aCov)
    # calc aCov
    E_WT_Phi_W = np.sum((PhiShape / PhiInvScale)[:, np.newaxis, np.newaxis] * E_WT_W, axis=0)
    aCov = inv(np.eye(C) + E_WT_Phi_W)
    # elboA = elbo4comp(SS, WMean, WCov, hShape, hInvScale, PhiShape, PhiInvScale, aCov)
    # assert(elboW <= elboH)
    # assert(elboH <= elboPhi)
    # assert(elboPhi <= elboA)
    return WMean, WCov, hShape, hInvScale, PhiShape, PhiInvScale, aCov

def elbo4comp(SS, WMean, WCov, hShape, hInvScale, PhiShape, PhiInvScale, aCov):
    elbo = 0.0
    for d in xrange(D):
        elbo += - PhiShape * np.log(PhiInvScale[d]) \
                + s * np.log(t) \
                + gammaln(PhiShape) - gammaln(s) \
                - (PhiShape - s) * \
                (psi(PhiShape) - np.log(PhiInvScale[d])) \
                + PhiShape * (1 - t / PhiInvScale[d])
    for c in xrange(C):
        elbo += - hShape * np.log(hInvScale[c]) \
                + f * np.log(g) \
                + gammaln(hShape) - gammaln(f) \
                - (hShape - f) * (psi(hShape) - np.log(hInvScale[c])) \
                + hShape * (1 - g / hInvScale[c]) \
                - (hShape - f) * (psi(hShape) - np.log(hInvScale[c])) \
                + hShape * (1 - g / hInvScale[c])
    E_WT_W = np.einsum('ij,ik->ijk', WMean, WMean)
    E_WT_W +=  WCov[:,:,np.newaxis] * np.eye(C)
    sumLogDetWCov = np.sum(np.sum(np.log(WCov), axis=1))
    elbo += .5 * (sumLogDetWCov + D * C)
    LU = np.dot(aCov, np.dot(WMean.T, np.diag(PhiShape / PhiInvScale)))
    xaT = np.inner(SS.xxT, LU)
    aaT = SS.N * aCov + np.dot(LU, xaT)
    elbo += .5 * (np.prod(slogdet(aCov)) + C) * SS.N - .5 * np.trace(aaT)
    E_WT_Phi_W = np.sum((PhiShape / PhiInvScale)[:, np.newaxis, np.newaxis]
                        * E_WT_W, axis=0)
    elbo += -.5 * D * SS.N * LOGTWOPI \
            + .5 * (np.sum(psi(PhiShape) - np.log(PhiInvScale))) * SS.N \
            - .5 * np.dot(PhiShape / PhiInvScale, np.diag(SS.xxT)) \
            + np.dot(PhiShape / PhiInvScale, np.einsum('ij, ij->i', xaT, WMean)) \
            - .5 * np.einsum('ij,ji', E_WT_Phi_W, aaT)
    return elbo / (SS.N * D)

def calcGaussCov(WMean, WCov, PhiShape, PhiInvScale):
    return np.dot(WMean, WMean.T) + np.diag(np.sum(WCov, axis=1)) \
                  + np.diag(PhiInvScale/(PhiShape-1))


if __name__ == '__main__':
    SS = SuffStatBag(K=K, D=D, C=C)
    SS.setField('N', X1.shape[0], dims='')
    SS.setField('xxT', dotATA(X1), dims=('D','D'))

    # eig
    elboEig = np.zeros(nIter)
    WMeanEig, hShapeEig, hInvScaleEig, PhiShapeEig, PhiInvScaleEig, aCovEig = \
        eigInit(SS.xxT, SS.N)
    WCovEig = EPS * np.ones((D,C))
    elboEig[0] = elbo4comp(SS, WMeanEig, WCovEig, hShapeEig, hInvScaleEig,
                           PhiShapeEig, PhiInvScaleEig, aCovEig)
    for i in xrange(1, nIter):
        WMeanEig, WCovEig, hShapeEig, hInvScaleEig, \
        PhiShapeEig, PhiInvScaleEig, aCovEig = \
            updateAll(X1, WMeanEig, hShapeEig, hInvScaleEig,
                      PhiShapeEig, PhiInvScaleEig, aCovEig)
        elboEig[i] = elbo4comp(SS, WMeanEig, WCovEig, hShapeEig, hInvScaleEig,
                               PhiShapeEig, PhiInvScaleEig, aCovEig)
    CovEig = calcGaussCov(WMeanEig, WCovEig, PhiShapeEig, PhiInvScaleEig)

    # old + sampling
    elboSamp = np.zeros((nTrial,nIter))
    maxElboSamp = -np.inf
    for i in xrange(nTrial):
        WMeanInit, hShapeInit, hInvScaleInit, \
        PhiShapeInit, PhiInvScaleInit, aCovInit = samplingInit(seed=i)
        hShapeInit = f + D / 2
        hInvScaleInit = g * np.ones(C)
        WCovInit = EPS * np.ones((D,C))
        elboSamp[i,0] = elbo4comp(SS, WMeanInit, WCovInit,
                                  hShapeInit, hInvScaleInit,
                                  PhiShapeInit, PhiInvScaleInit, aCovInit)
        for j in xrange(1,nIter):
            if j == 1:
                WMeanSamp, WCovSamp, hShapeSamp, hInvScaleSamp, \
                PhiShapeSamp, PhiInvScaleSamp, aCovSamp \
                    = updateAll(X1, WMeanInit, hShapeInit, hInvScaleInit,
                                PhiShapeInit, PhiInvScaleInit, aCovInit)
            else:
                WMeanSamp, WCovSamp, hShapeSamp, hInvScaleSamp, \
                PhiShapeSamp, PhiInvScaleSamp, aCovSamp \
                    = updateAll(X1, WMeanSamp, hShapeSamp, hInvScaleSamp,
                                PhiShapeSamp, PhiInvScaleSamp, aCovSamp)
            elboSamp[i,j] = elbo4comp(SS, WMeanSamp, WCovSamp,
                                      hShapeSamp, hInvScaleSamp,
                                      PhiShapeSamp, PhiInvScaleSamp, aCovSamp)
        assert (np.all(np.logical_or(elboSamp[i,1:] - elboSamp[i,:-1] >= 0,
                np.abs(elboSamp[i,1:] - elboSamp[i,:-1]) < 1e-4)))
        if elboSamp[i,-1] > maxElboSamp:
            maxElboSamp = elboSamp[i,-1]
            CovSamp = calcGaussCov(WMeanSamp, WCovSamp,
                                   PhiShapeSamp, PhiInvScaleSamp)


    # old + rand example
    elboRand = np.zeros((nTrial,nIter))
    maxElboRand = -np.inf
    for i in xrange(nTrial):
        WMeanInit, hShapeInit, hInvScaleInit, \
        PhiShapeInit, PhiInvScaleInit, aCovInit = \
            randExampleInit(X1, seed=i)
        WCovInit = EPS * np.ones((D,C))
        elboRand[i,0] = elbo4comp(SS, WMeanInit, WCovInit,
                                  hShapeInit, hInvScaleInit,
                                  PhiShapeInit, PhiInvScaleInit, aCovInit)
        for j in xrange(1, nIter):
            if j == 1:
                WMeanRand, WCovRand, hShapeRand, hInvScaleRand, \
                PhiShapeRand, PhiInvScaleRand, aCovRand \
                    = updateAll(X1, WMeanInit, hShapeInit, hInvScaleInit,
                                     PhiShapeInit, PhiInvScaleInit, aCovInit)
            else:
                WMeanRand, WCovRand, hShapeRand, hInvScaleRand, \
                PhiShapeRand, PhiInvScaleRand, aCovRand \
                    = updateAll(X1, WMeanRand, hShapeRand, hInvScaleRand,
                                PhiShapeRand, PhiInvScaleRand, aCovRand)
            elboRand[i,j] = elbo4comp(SS, WMeanRand, WCovRand,
                                      hShapeRand, hInvScaleRand,
                                      PhiShapeRand, PhiInvScaleRand, aCovRand)
        assert (np.all(np.logical_or(elboRand[i,1:] - elboRand[i,:-1] >= 0,
                np.abs(elboRand[i,1:] - elboRand[i,:-1]) < 1e-4)))
        if elboRand[i,-1] > maxElboRand:
            maxElboRand = elboRand[i,-1]
            CovRand = calcGaussCov(WMeanRand, WCovRand,
                                   PhiShapeRand, PhiInvScaleRand)

    # plot elbo
    plt.figure(0)
    for i in xrange(nTrial):
        b0 = plt.plot(np.arange(nIter-1), elboRand[i,1:], 'y', linewidth=2)
    for i in xrange(nTrial):
        b1 = plt.plot(np.arange(nIter-1), elboSamp[i,1:], 'r', linewidth=2)
    b2 = plt.plot(np.arange(nIter-1), elboEig[1:], 'b', linewidth=2)
    plt.legend([b0[0], b1[0], b2[0]],
               ['Rand example + Old', 'Sampling + Old', 'Eigen decomp'], loc=4)
    plt.ylabel('Elbo', fontsize=25)
    plt.xlabel('iteration', fontsize=25)
    # plt.savefig('DeadLeavesD25_elbo.pdf', bbox_inches='tight')


    # plot covariance matrix
    if D == 2:
        plt.figure(1)
        from bnpy.viz.GaussViz import plotGauss2DContour
        plt.plot(X1[:,0], X1[:,1], 'k.')
        plotGauss2DContour(np.zeros(D), CovEig, color='b')
        plotGauss2DContour(np.zeros(D), CovSamp, color='r')
        plotGauss2DContour(np.zeros(D), CovRand, color='y')
        plt.xlim((-1, 1))
        plt.ylim((-1, 1))
        # plt.savefig('StarCovarK5.pdf', bbox_inches='tight')
    else:
        plt.figure(1)
        plt.imshow(CovEig, interpolation='nearest', cmap='hot', clim=[-.25, 1])
        plt.title('Best Eigen decomp', fontsize=25)
        # plt.savefig('DeadLeavesD25_eig.pdf', bbox_inches='tight')
        plt.figure(2)
        plt.imshow(CovSamp, interpolation='nearest', cmap='hot', clim=[-.25, 1])
        plt.title('Best Sampling', fontsize=25)
        # plt.savefig('DeadLeavesD25_old.pdf', bbox_inches='tight')
        plt.figure(3)
        plt.imshow(CovRand, interpolation='nearest', cmap='hot', clim=[-.25, 1])
        plt.title('Best Rand example', fontsize=25)
    plt.show()
    print 'hehe'