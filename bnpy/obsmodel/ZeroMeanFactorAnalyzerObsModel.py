import numpy as np
from bnpy.suffstats import ParamBag, SuffStatBag
from bnpy.obsmodel.AbstractObsModel import AbstractObsModel
from numpy.linalg import inv, solve, det, slogdet
from scipy.linalg import cholesky, solve_triangular
from scipy.special import psi, gammaln
from bnpy.util import dotATA
from bnpy.util import LOGTWOPI


class ZeroMeanFactorAnalyzerObsModel(AbstractObsModel):

    def __init__(self, inferType='VB', Data=None, C = None, **PriorArgs):
        self.D = Data.dim
        self.C = C
        self.K = 0
        self.inferType = inferType
        self.createPrior(Data, **PriorArgs)
        self.Cache = dict()

    def createPrior(self, Data, f = None, g = None, s=None, t = None):
        K = self.K
        D = self.D
        C = self.C
        self.Prior = ParamBag(K=K, D=D, C=C)
        self.Prior.setField('f', f)
        self.Prior.setField('g', g)
        self.Prior.setField('s', s)
        self.Prior.setField('t', t)

    ######################################################### I/O Utils
    #########################################################   for humans
    def get_name(self):
        return 'ZeroMeanFactorAnalyzer'

    def get_info_string(self):
        return 'Zero-mean factor analyzer.'

    def get_info_string_prior(self):
        msg = 'Gamma shape and invScale param on precision of factor loading matrices: f, g\n'
        if self.D > 2:
            sfx = ' ...'
        else:
            sfx = ''
        msg += 'f  = %s%s\n' % (str(self.Prior.f), sfx)
        msg += 'g  = %s%s' % (str(self.Prior.g), sfx)
        msg += '\nGamma shape and invScale param on precision of diagonal noise matrices: s, t\n'
        msg += 's  = %s%s\n' % (str(self.Prior.s), sfx)
        msg += 't  = %s%s' % (str(self.Prior.t), sfx)
        msg = msg.replace('\n', '\n  ')
        return msg

    ########################################################### Suff Stats
    ###########################################################
    # @profile
    def calcSummaryStats(self, Data, SS, LP):
        X = Data.X
        resp = LP['resp']

        K = resp.shape[1]
        D = self.D
        C = self.C

        if SS is None:
            SS = SuffStatBag(K=K, D=D, C=C)

        # Expected count for each k
        #  Usually computed by allocmodel. But just in case...
        if not hasattr(SS, 'N'):
            SS.setField('N', np.sum(resp, axis=0), dims='K')
        elif not hasattr(SS.kwargs,'C'):
            SS.kwargs['C'] = C
            setattr(SS._Fields, 'C', C)

        # Expected outer-product for each k
        sqrtResp = np.sqrt(resp)
        xxT = np.zeros( (K, D, D) )
        for k in xrange(K):
            xxT[k] = dotATA(sqrtResp[:,k][:,np.newaxis] * X)
        SS.setField('xxT', xxT, dims=('K','D','D'))

        return SS

  ########################################################### Local step
  ###########################################################
    # @profile
    def calc_local_params(self, Data, LP=None, **kwargs):
        if LP is None:
            LP = dict()
        if self.inferType == 'EM':
            raise NotImplementedError()
        else:
            LP['aMean'] = self.calcA_FromPost(Data)
            LP['E_log_soft_ev'] = self.calcLogSoftEvMatrix_FromPost(Data, LP)
        return LP
    # @profile
    def calcA_FromPost(self, Data):
        N = Data.nObs
        K = self.Post.K
        D = self.D
        C = self.C
        # calculate aMean
        aMean = np.zeros((N, K, C))
        for k in xrange(K):
            aCovk_WMeankT_invPsi = np.inner(self.Post.aCov[k], self.Post.WMean[k]) * \
                                   (self.Post.PhiShape[k] / self.Post.PhiInvScale[k])
            aMean[:,k] = np.inner(Data.X, aCovk_WMeankT_invPsi)
        return aMean
    # @profile
    def calcLogSoftEvMatrix_FromPost(self, Data, LP):
        N = Data.nObs
        K = self.Post.K
        L = np.zeros((N, K))
        DataX2 = Data.X**2
        for k in xrange(K):
            L[:,k] = .5 * np.einsum('ij,ji->i', LP['aMean'][:,k],
                                    np.inner(inv(self.Post.aCov[k]), LP['aMean'][:,k])) \
                     - .5 * np.inner(self.Post.PhiShape[k] / self.Post.PhiInvScale[k], DataX2) \
                     + .5 * np.sum(psi(self.Post.PhiShape[k]) - np.log(self.Post.PhiInvScale[k])) \
                     + .5 * np.prod(slogdet(self.Post.aCov[k]))
        return L

  ########################################################### Global step
  ###########################################################
    # @profile
    def updatePost(self, SS):
        self.ClearCache()
        if not hasattr(self, 'Post') or self.Post.K != SS.K:
            self.Post = ParamBag(K=SS.K, D=SS.D, C=SS.C)
        WMean, WCov, hShape, hInvScale, PhiShape, PhiInvScale, aCov = self.calcPostParams(SS)
        self.Post.setField('WMean', WMean, dims=('K','D','C'))
        self.Post.setField('WCov', WCov, dims=('K','D','C','C'))
        self.Post.setField('hShape', hShape)
        self.Post.setField('hInvScale', hInvScale, dims=('K','C'))
        self.Post.setField('PhiShape', PhiShape, dims=('K'))
        self.Post.setField('PhiInvScale', PhiInvScale, dims=('K','D'))
        self.Post.setField('aCov', aCov, dims=('K','C','C'))
        self.K = SS.K

    def calcPostParams(self, SS):
        if hasattr(self.Post,'hShape') and hasattr(self.Post,'hInvScale') \
           and hasattr(self.Post,'PhiShape') and hasattr(self.Post,'PhiInvScale')\
           and hasattr(self.Post, 'WMean') and hasattr(self.Post, 'aCov'):
            WMean = self.Post.WMean
            hShape = self.Post.hShape
            hInvScale = self.Post.hInvScale
            PhiShape = self.Post.PhiShape
            PhiInvScale = self.Post.PhiInvScale
            aCov = self.Post.aCov
        else:
            WMean = np.random.randn(SS.K, self.D, self.C)
            hShape = self.Prior.f
            hInvScale = self.Prior.g * np.ones((SS.K, self.C))
            PhiShape = self.Prior.s * np.ones(SS.K)
            PhiInvScale = self.Prior.t * np.ones((SS.K, self.D))
            aCov = np.tile(np.eye(self.C), (SS.K,1,1))
        for i in xrange(1):
            WMean, WCov = self.calcPostW(SS, WMean, hShape, hInvScale, PhiShape, PhiInvScale, aCov)
            aCov = self.calcPostACov(WMean, WCov, PhiShape, PhiInvScale)
            hShape, hInvScale = self.calcPostH(WMean, WCov)
            PhiShape, PhiInvScale = self.calcPostPhi(SS, WMean, WCov, PhiShape, PhiInvScale, aCov)
        return WMean, WCov, hShape, hInvScale, PhiShape, PhiInvScale, aCov

    def calcPostACov(self, WMean, WCov, PhiShape, PhiInvScale):
        C = self.C
        D = self.D
        K = WMean.shape[0]
        aCov = np.zeros((K, C, C))
        for k in xrange(K):
            E_W_WT = np.zeros((D, C, C))
            for d in xrange(D):
                E_W_WT[d] = np.outer(WMean[k][d], WMean[k][d]) + WCov[k][d]
            E_WT_Phi_W = np.sum((PhiShape[k] / PhiInvScale[k])[:, np.newaxis, np.newaxis]
                                * E_W_WT, axis=0)
            aCov[k] = inv(np.eye(C) + E_WT_Phi_W)
        return aCov

    # @profile
    def calcPostW(self, SS, WMean_old, hShape, hInvScale, PhiShape, PhiInvScale, aCov):
        K = SS.K
        D = self.D
        C = self.C
        WCov = np.zeros((K, D, C, C))
        WMean = np.zeros((K, D, C))
        for k in xrange(K):
            L = aCov[k]
            LU = np.dot(L, np.dot(WMean_old[k].T, np.diag(PhiShape[k] / PhiInvScale[k])))
            aaT = SS.N[k]*L + np.dot(LU, np.inner(SS.xxT[k],LU))
            xaT = np.inner(SS.xxT[k], LU)
            SigmaInvWW = np.diag(hShape / hInvScale[k]) \
                         + (PhiShape[k] / PhiInvScale[k])[:,np.newaxis,np.newaxis] \
                         * np.tile(aaT, (D,1,1))
            WCov[k] = inv(SigmaInvWW)
            for d in xrange(D):
                WMean[k, d] = np.dot(WCov[k,d], PhiShape[k] / PhiInvScale[k,d] * xaT[d])
        return WMean, WCov

    # @profile
    def calcPostH(self, WMean, WCov):
        C = self.C
        K = WMean.shape[0]
        hShape = self.Prior.f + .5 * self.D
        hInvScale = np.zeros((K, C))
        for k in xrange(K):
            for c in xrange(C):
                hInvScale[k,c] = self.Prior.g + .5 * np.sum(WMean[k,:,c]**2 + WCov[k,:,c,c])
        return hShape, hInvScale

    # @profile
    def calcPostPhi(self, SS, WMean, WCov, PhiShape_old, PhiInvScale_old, aCov):
        K = SS.K
        D = self.D
        PhiShape = self.Prior.s + .5 * SS.N
        PhiInvScale = self.Prior.t * np.ones((K, D))
        for k in xrange(K):
            L = aCov[k]
            LU = np.dot(L, np.dot(WMean[k].T, np.diag(PhiShape_old[k] / PhiInvScale_old[k])))
            aaT = SS.N[k]*L + np.dot(LU, np.inner(SS.xxT[k],LU))
            xaT = np.inner(SS.xxT[k], LU)
            for d in xrange(D):
                PhiInvScale[k,d] += .5 * (SS.xxT[k,d,d]
                                    - 2 * np.dot(xaT[d], WMean[k,d])
                                    + np.einsum('ij,ji', np.outer(WMean[k,d], WMean[k,d])
                                                + WCov[k,d], aaT))
        return PhiShape, PhiInvScale

    ########################################################### VB ELBO step
    ###########################################################
    # @profile
    def calcELBO_Memoized(self, SS, afterMStep=False):
        ''' Calculate obsModel's ELBO using sufficient statistics SS and Post.

            Args
            -------
            SS : bnpy SuffStatBag, contains fields for N, xxT
            afterMStep : boolean flag
                    if 1, elbo calculated assuming M-step just completed

            Returns
            -------
            obsELBO : scalar float, = E[ log p(Phi) + log p(h) + log p(W) + log p(a) + log p(x)
                                         - log q(Phi) - log q(h) - log q(W) - log q(a) ]
        '''
        elbo = np.zeros(SS.K)
        Post = self.Post
        Prior = self.Prior
        K = SS.K
        C = SS.C
        D = SS.D
        for k in xrange(K):
            # terms related with Phi
            for d in xrange(D):
                elbo[k] += - Post.PhiShape[k] * np.log(Post.PhiInvScale[k,d]) \
                           + Prior.s * np.log(Prior.t) \
                           + gammaln(Post.PhiShape[k]) - gammaln(Prior.s) \
                           - (Post.PhiShape[k] - Prior.s) * \
                             (psi(Post.PhiShape[k]) - np.log(Post.PhiInvScale[k,d])) \
                           + Post.PhiShape[k] * (1 - Prior.t / Post.PhiInvScale[k,d])

            # terms related with h
            for c in xrange(C):
                elbo[k] += - Post.hShape * np.log(Post.hInvScale[k,c]) \
                           + Prior.f * np.log(Prior.g) \
                           + gammaln(Post.hShape) - gammaln(Prior.f) \
                           - (Post.hShape - Prior.f) * (psi(Post.hShape) - np.log(Post.hInvScale[k,c])) \
                           + Post.hShape * (1 - Prior.g / Post.hInvScale[k,c])

            # terms related with W
            elbo[k] += .5 * np.sum(self.GetCached('logdetWCov', k)) \
                       + .5 * D * (C + np.sum(psi(Post.hShape) - np.log(Post.hInvScale[k]))) \
                       - .5 * np.sum(np.dot(np.diagonal(self.GetCached('E_WWT',k),axis1=1,axis2=2),
                                              Post.hShape / Post.hInvScale[k]))

            # terms related with a
            L = Post.aCov[k]
            LU = np.dot(L, np.dot(Post.WMean[k].T, np.diag(Post.PhiShape[k] / Post.PhiInvScale[k])))
            aaT = SS.N[k]*L + np.dot(LU, np.inner(SS.xxT[k],LU))
            xaT = np.inner(SS.xxT[k], LU)
            elbo[k] += .5 * (np.prod(slogdet(L)) + C) * SS.N[k] \
                       - .5 * np.trace(aaT)

            # terms related with x
            elbo[k] += - .5 * (D * LOGTWOPI -
                               np.sum(psi(Post.PhiShape[k]) - np.log(Post.PhiInvScale[k]))) * SS.N[k] \
                       - .5 * np.dot(Post.PhiShape[k] / Post.PhiInvScale[k], np.diag(SS.xxT[k])) \
                       + np.dot(Post.PhiShape[k] / Post.PhiInvScale[k],
                                np.einsum('ij, ij->i', xaT, Post.WMean[k])) \
                       - .5 * np.einsum('ij,ji', self.GetCached('E_WT_invPsi_W', k), aaT)
        return np.sum(elbo)

    def getDatasetScale(self, SS):
        ''' Get scale factor for dataset, indicating number of observed scalars.

            Used for normalizing the ELBO so it has reasonable range.

            Returns
            ---------
            s : scalar positive integer
        '''
        return SS.N.sum() * SS.D

    ########################################################### VB Expectations
    ###########################################################
    # @profile
    def _E_WWT(self, k=None):
        if k is None:
            raise NotImplementedError()
        else:
            C = self.C
            D = self.D
            result = np.zeros((D,C,C))
            for d in xrange(D):
                result[d] = np.outer(self.Post.WMean[k][d],self.Post.WMean[k][d]) + self.Post.WCov[k][d]
            return result

    def _E_WT_invPsi_W(self, k=None):
        if k is None:
            raise NotImplementedError()
        else:
            return np.sum((self.Post.PhiShape[k] / self.Post.PhiInvScale[k])[:, np.newaxis, np.newaxis]
                          * self.GetCached('E_WWT', k), axis=0)

    def _cholWCov(self, k):
        D = self.D
        C = self.C
        result = np.zeros((D,C,C))
        for d in xrange(D):
            result[d] = cholesky(self.Post.WCov[k,d], lower=True)
        return result

    def _logdetWCov(self, k):
        D = self.D
        cholWCov = self.GetCached('cholWCov', k)
        result = np.zeros(D)
        for d in xrange(D):
            result[d] = 2 * np.sum(np.log(np.diag(cholWCov[d])))
        return result


if __name__ == '__main__':
    import bnpy, profile
    profile.run("bnpy.run('D3C2K2_ZM', 'FiniteMixtureModel','ZeroMeanFactorAnalyzer', 'VB',nLap=500, K=2, printEvery=50)",sort=1)
    # hmodel, RInfo = bnpy.run('D3C2K2_ZM', 'FiniteMixtureModel',
    #                         'ZeroMeanFactorAnalyzer', 'VB',
    #                         nLap=500, K=2, printEvery=50)