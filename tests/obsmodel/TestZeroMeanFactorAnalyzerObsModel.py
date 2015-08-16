import numpy as np
from numpy.linalg import inv, solve, det, slogdet, eig
from scipy.special import psi, gammaln
import bnpy
from bnpy.util import LOGTWOPI
from bnpy.suffstats import ParamBag, SuffStatBag
import matplotlib.pylab as plt
from bnpy.viz.GaussViz import plotGauss2DContour

def get_data(nObsTotal=1e3):
    import scipy.linalg
    import numpy as np
    from bnpy.util.RandUtil import rotateCovMat
    from bnpy.data import XData
    D = 2
    # Create basic 2D cov matrix with major axis much longer than minor one
    V = 1.0 / 16.0
    n = input("Choose a number between 1 ~ 6 to get different covariance shapes: ")
    if n == 1:
        a, b = 1, 0
    elif n == 2:
        a, b = 1, 1
    elif n == 3:
        a, b = 1, 2
    elif n == 4:
        a, b = 1, 3
    elif n == 5:
        a, b = 100, 0
    elif n == 6:
        a, b = 20, 0
    else:
        raise ValueError('Please choose from 1 ~ 6!')
    SigmaBase = np.asarray([[V, 0], [0, a * V / 100.0]])
    # Create several Sigmas by rotating this basic covariance matrix
    Sigma = rotateCovMat(SigmaBase, b * np.pi / 4.0)
    cholSigma = scipy.linalg.cholesky(Sigma)
    PRNG = np.random.RandomState(0)
    N = nObsTotal
    X = np.dot(cholSigma.T, PRNG.randn(D, N)).T
    TrueZ = np.ones(N)
    Data = XData(X=X, TrueZ=TrueZ)
    Data.name = 'Diagnal 2D'
    Data.summary = '1 true cluster in 2D space'
    return Data, Sigma

def get_SS(Data):
    SS = SuffStatBag(D=D, C=C)
    SS.setField('N', Data.X.shape[0])
    SS.setField('xxT', np.dot(Data.X.T, Data.X), dims=('D','D'))
    return SS

if __name__ == '__main__':
    plt.close()
    D = 2
    C = 1
    Data, trueSigma = get_data(nObsTotal=1e3)
    SS = get_SS(Data)

    # set Prior
    Prior = ParamBag(D=D, C=C)
    Prior.setField('f', 0.001)
    Prior.setField('g', 0.001)
    Prior.setField('s', 0.001)
    Prior.setField('t', 0.001)

    # initialization
    eigVal, eigVec = eig(SS.xxT / SS.N)
    idx = np.argsort(eigVal)[::-1]
    eigVal = eigVal[idx]
    eigVec = eigVec[:, idx]
    PhiShape = Prior.s + .5 * SS.N
    sigma2 = np.sum(eigVal[C:]) / (D - C)
    PhiInvScale = sigma2 * (PhiShape) * np.ones(D)
    WMean = np.dot(eigVec[:,:C], np.diag(np.sqrt(eigVal[:C] - sigma2)))
    assert np.all(PhiInvScale) > 0
    hShape = Prior.f + .5 * D
    hInvScale = Prior.g * np.ones(C)
    E_W_WT = np.zeros((D, C, C))
    for d in xrange(D):
        E_W_WT[d] = np.outer(WMean[d], WMean[d])
    E_WT_Phi_W = np.sum((PhiShape / PhiInvScale)[:, np.newaxis, np.newaxis]
                            * E_W_WT, axis=0)
    aCov = inv(np.eye(C) + E_WT_Phi_W)

    nIter = 200
    j = 0
    k = 1
    fig0 = plt.figure(0)
    elbo = np.zeros(nIter)

    for i in xrange(nIter):
        # get xaT and aaT
        LU = np.dot(aCov, np.dot(WMean.T, np.diag(PhiShape / PhiInvScale)))
        aaT = SS.N * aCov + np.dot(LU, np.inner(SS.xxT, LU))
        xaT = np.inner(SS.xxT, LU)

        # update W
        SigmaInvWW = np.diag(hShape / hInvScale) \
                            + (PhiShape / PhiInvScale)[:,np.newaxis,np.newaxis] \
                            * np.tile(aaT, (D,1,1))
        WCov = inv(SigmaInvWW)
        for d in xrange(D):
            WMean[d] = np.dot(WCov[d], PhiShape / PhiInvScale[d] * xaT[d])

        # update H
        hShape = Prior.f + .5 * D
        hInvScale = np.zeros(C)
        for c in xrange(C):
            hInvScale[c] = Prior.g + .5 * np.sum(WMean[:,c]**2 + WCov[:,c,c])

        # update Phi
        PhiShape = Prior.s + .5 * SS.N
        PhiInvScale = Prior.t * np.ones(D)
        for d in xrange(D):
            PhiInvScale[d] += .5 * (SS.xxT[d,d] - 2 * np.dot(xaT[d], WMean[d])
                                    + np.einsum('ij,ji', np.outer(WMean[d], WMean[d])
                                                + WCov[d], aaT))

        # update aCov
        E_W_WT = np.zeros((D, C, C))
        for d in xrange(D):
            E_W_WT[d] = np.outer(WMean[d], WMean[d]) + WCov[d]
        E_WT_Phi_W = np.sum((PhiShape / PhiInvScale)[:, np.newaxis, np.newaxis]
                            * E_W_WT, axis=0)
        aCov = inv(np.eye(C) + E_WT_Phi_W)

        # calculate elbo
        # terms related with Phi
        for d in xrange(D):
            elbo[i] += - PhiShape * np.log(PhiInvScale[d]) \
                       + Prior.s * np.log(Prior.t) \
                       + gammaln(PhiShape) - gammaln(Prior.s) \
                       - (PhiShape - Prior.s) * \
                         (psi(PhiShape) - np.log(PhiInvScale[d])) \
                       + PhiShape * (1 - Prior.t / PhiInvScale[d])

        # terms related with h
        for c in xrange(C):
            elbo[i] += - hShape * np.log(hInvScale[c]) \
                       + Prior.f * np.log(Prior.g) \
                       + gammaln(hShape) - gammaln(Prior.f) \
                       - (hShape - Prior.f) * (psi(hShape) - np.log(hInvScale[c])) \
                       + hShape * (1 - Prior.g / hInvScale[c])

        # terms related with W
        logdetWCov = np.zeros(D)
        for d in xrange(D):
            logdetWCov[d] = np.prod(slogdet(WCov[d]))
        E_W_WT = np.zeros((D, C, C))
        for d in xrange(D):
            E_W_WT[d] = np.outer(WMean[d], WMean[d]) + WCov[d]
        E_WT_Phi_W = np.sum((PhiShape / PhiInvScale)[:, np.newaxis, np.newaxis]
                             * E_W_WT, axis=0)
        elbo[i] += .5 * np.sum(logdetWCov) \
                   + .5 * D * (C + np.sum(psi(hShape) - np.log(hInvScale))) \
                   - .5 * np.sum(np.dot(np.diagonal(E_W_WT,axis1=1,axis2=2), hShape / hInvScale))

        # terms related with a
        LU = np.dot(aCov, np.dot(WMean.T, np.diag(PhiShape / PhiInvScale)))
        aaT = SS.N * aCov + np.dot(LU, np.inner(SS.xxT, LU))
        xaT = np.inner(SS.xxT, LU)
        elbo[i] += .5 * (np.prod(slogdet(aCov)) + C) * SS.N - .5 * np.trace(aaT)

        # terms related with x
        elbo[i] += - .5 * (D * LOGTWOPI -
                           np.sum(psi(PhiShape) - np.log(PhiInvScale))) * SS.N \
                   - .5 * np.dot(PhiShape / PhiInvScale, np.diag(SS.xxT)) \
                   + np.dot(PhiShape / PhiInvScale, np.einsum('ij, ij->i', xaT, WMean)) \
                   - .5 * np.einsum('ij,ji', E_WT_Phi_W, aaT)
        # check the increase of elbo
        if i > 0:
            assert elbo[i] > elbo[i-1]
        # plot the covariance
        if (i < 10 and i%3 == 0) or (i+1) in [20, 50, 100, 200]:
            if j == 4:
                j = 1
                k = k + 1
            else:
                j = j + 1
            from matplotlib import gridspec
            gs = gridspec.GridSpec(2, 4)
            ax = fig0.add_subplot(gs[k-1,j-1])
            plotGauss2DContour(np.zeros(D), trueSigma, color='r')
            sigma = np.inner(WMean, WMean) \
                    + np.sum(np.diagonal(WCov),axis=1) \
                    + np.diag(PhiInvScale / (PhiShape - 1))
            plotGauss2DContour(np.zeros(D), sigma, color='b')
            plt.axis([-.6, .6, -.6, .6])
            ax.set_xticks([])
            ax.set_yticks([])
            plt.xlabel(str(i+1),fontsize=15)
            print 'Iteration ' + str(i+1) + ', elbo :' + str(elbo[i])
    # from matplotlib.backends.backend_pdf import PdfPages
    # pp = PdfPages('/Users/Geng/Downloads/cov.pdf')
    # pp.savefig()
    # pp.close()

    # plt.figure(1)
    # plt.plot(np.arange(nIter), elbo, linewidth=1.5)
    # plt.axis([0, 200, 0, 1.1 * elbo[-1]])

    # pp = PdfPages('/Users/Geng/Downloads/elbo.pdf')
    # pp.savefig()
    # pp.close()

    plt.show()



# L = Post.aCov[k]
# LU = np.dot(L, np.dot(Post.WMean[k].T, np.diag(Post.PhiShape[k] / Post.PhiInvScale[k])))
# aaT = SS.N[k]*L + np.dot(LU, np.inner(SS.xxT[k],LU))
# xaT = np.inner(SS.xxT[k], LU)
#
# elbo = 0.
# for d in xrange(D):
#     elbo += - (Post.PhiShape[k] - Prior.s) * (psi(Post.PhiShape[k]) - np.log(Post.PhiInvScale[k,d])) \
#                     + Post.PhiShape[k] * (1 - Prior.t / Post.PhiInvScale[k,d])
# elbo += .5 * (np.sum(psi(Post.PhiShape[k]) - np.log(Post.PhiInvScale[k]))) * SS.N[k] \
#              - .5 * np.dot(Post.PhiShape[k] / Post.PhiInvScale[k], np.diag(SS.xxT[k])) \
#              + np.dot(Post.PhiShape[k] / Post.PhiInvScale[k], np.einsum('ij, ij->i', xaT, Post.WMean[k])) \
#             - .5 * np.einsum('ij,ji', self.GetCached('E_WT_invPsi_W', k), aaT)
# print elbo
#
# elbo2 = 0.
# for c in xrange(C):
#     elbo2 += - (Post.hShape - Prior.f) * (psi(Post.hShape) - np.log(Post.hInvScale[k,c])) \
#             + Post.hShape * (1 - Prior.g / Post.hInvScale[k,c])
# elbo2 += .5 * D * (np.sum(psi(Post.hShape) - np.log(Post.hInvScale[k]))) \
#                 - .5 * np.sum(np.dot(np.diagonal(self.GetCached('E_WWT',k),axis1=1,axis2=2),
#                                       Post.hShape / Post.hInvScale[k]))
# print elbo2
#
#
#
# # L = aCov[k]
# # LU = np.dot(L, np.dot(WMean[k].T, np.diag(PhiShape[k] / PhiInvScale[k])))
# # aaT = SS.N[k]*L + np.dot(LU, np.inner(SS.xxT[k],LU))
# # xaT = np.inner(SS.xxT[k], LU)
# E_WWT = np.zeros((K,D,C,C))
# for k in xrange(K):
#     for d in xrange(D):
#         E_WWT[k,d] = np.outer(WMean[k,d], WMean[k,d]) + WCov[k,d]
#
# E_WT_invPsi_W = np.zeros((K,D,C,C))
# for k in xrange(K):
#     E_WT_invPsi_W[k] = np.sum((PhiShape[k] / PhiInvScale[k])[:, np.newaxis, np.newaxis]
#                           * E_WWT[k], axis=0)
#
# elbo = 0.
# for d in xrange(D):
#     elbo += - (PhiShape[k] - self.Prior.s) * (psi(PhiShape[k]) - np.log(PhiInvScale[k,d])) \
#                     + PhiShape[k] * (1 - self.Prior.t / PhiInvScale[k,d])
# elbo += .5 * (np.sum(psi(PhiShape[k]) - np.log(PhiInvScale[k]))) * SS.N[k] \
#              - .5 * np.dot(PhiShape[k] / PhiInvScale[k], np.diag(SS.xxT[k])) \
#              + np.dot(PhiShape[k] / PhiInvScale[k], np.einsum('ij, ij->i', xaT, WMean[k])) \
#             - .5 * np.einsum('ij,ji', E_WT_invPsi_W[k], aaT)
# print elbo
#
# elbo2 = 0.
# for c in xrange(C):
#     elbo2 += - (hShape - self.Prior.f) * (psi(hShape) - np.log(hInvScale[k,c])) \
#             + hShape * (1 - self.Prior.g / hInvScale[k,c])
# elbo2 += .5 * D * (np.sum(psi(hShape) - np.log(hInvScale[k]))) \
#                 - .5 * np.sum(np.dot(np.diagonal(E_WWT[k],axis1=1,axis2=2),
#                                       hShape / hInvScale[k]))
# print elbo2