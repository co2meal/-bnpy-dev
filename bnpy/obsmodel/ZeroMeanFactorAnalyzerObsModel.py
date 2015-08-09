import numpy as np
from bnpy.suffstats import ParamBag, SuffStatBag
from bnpy.obsmodel.AbstractObsModel import AbstractObsModel
from numpy.linalg import inv, solve, det, slogdet, eig
from scipy.linalg import cholesky, solve_triangular
from scipy.special import psi, gammaln
from bnpy.util import dotATA
from bnpy.util import LOGTWOPI


class ZeroMeanFactorAnalyzerObsModel(AbstractObsModel):

    def __init__(self, inferType='VB', Data=None, C = None, **PriorArgs):
        if Data is not None:
            self.D = Data.dim
        else:
            self.D = int(PriorArgs['D'])
        if C is None:
            C = 1
        else:
            C = int(C)
        self.C = C
        self.K = 0
        self.inferType = inferType
        self.createPrior(Data, **PriorArgs)
        self.Cache = dict()

    def createPrior(self, Data, f = None, g = None, s=None, t = None, **kwargs):
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
    def calcSummaryStats(self, Data, SS, LP, **kwargs):
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
    def updatePost(self, SS, **kwargs):
        self.ClearCache()
        WMean, WCov, hShape, hInvScale, PhiShape, PhiInvScale, aCov = self.calcPostParams(SS, **kwargs)
        self.Post = ParamBag(K=SS.K, D=SS.D, C=SS.C)
        self.Post.setField('WMean', WMean, dims=('K','D','C'))
        self.Post.setField('WCov', WCov, dims=('K','D','C','C'))
        self.Post.setField('hShape', hShape)
        self.Post.setField('hInvScale', hInvScale, dims=('K','C'))
        self.Post.setField('PhiShape', PhiShape, dims=('K'))
        self.Post.setField('PhiInvScale', PhiInvScale, dims=('K','D'))
        self.Post.setField('aCov', aCov, dims=('K','C','C'))
        self.K = SS.K
        assert self.K == self.Post.K


    def calcPostParams(self, SS,
                       newCompList = [],
                       nIter4NewComp = 5,
                       mergeCompA=None, mergeCompB=None,
                       isBirthCreate = False,
                       BirthCleanUpList = None,
                       nBirthExpand = None,
                       BirthRefineList = [],
                       kdel = None):
        C = SS.C
        D = SS.D
        K = SS.K
        if not hasattr(self, 'Post') or isBirthCreate:
            PRNG = np.random.RandomState(0)
            WMean = PRNG.randn(SS.K, self.D, self.C)
            hShape = self.Prior.f
            hInvScale = self.Prior.g * np.ones((SS.K, self.C))
            PhiShape = self.Prior.s * np.ones(SS.K)
            PhiInvScale = self.Prior.t * np.ones((SS.K, self.D))
            aCov = np.tile(np.eye(self.C), (SS.K,1,1))
        else:
            Post = self.Post.copy()
            if mergeCompA is not None and mergeCompB is not None:
                assert SS.K + 1 == self.Post.K
                assert mergeCompA < mergeCompB
                Post.removeComp(mergeCompB)
                newCompList = np.array([mergeCompA])
            elif BirthCleanUpList is not None:
                assert SS.K + len(BirthCleanUpList) == Post.K
                BirthCleanUpList = sorted(BirthCleanUpList, reverse=True)
                for k in BirthCleanUpList:
                    self.Post.removeComp(k)
            elif nBirthExpand is not None:
                assert SS.K == nBirthExpand + Post.K
                newCompList = np.arange(Post.K, SS.K)
            elif kdel is not None:
                kdel = sorted(kdel, reverse=True)
                for k in kdel:
                    Post.removeComp(k)
            elif Post.K != SS.K:
                raise NotImplementedError

            WMean = Post.WMean
            PhiShape = Post.PhiShape
            PhiInvScale = Post.PhiInvScale
            hShape = Post.hShape
            hInvScale = Post.hInvScale
            aCov = Post.aCov

            for k in newCompList:
                if np.allclose(SS.xxT[k],0) and np.allclose(SS.N[k],0):
                    PRNG = np.random.RandomState(0)
                    eigVal = PRNG.randn(D)
                    eigVec = PRNG.randn(D, D)
                else:
                    eigVal, eigVec = eig(SS.xxT[k] / SS.N[k])
                    idx = np.argsort(eigVal)[::-1]
                    eigVal = eigVal[idx]
                    eigVec = eigVec[:, idx]
                PhiShape[k] = self.Prior.s + .5 * SS.N[k]
                sigma2 = np.sum(eigVal[C:]) / (D - C)
                PhiInvScale[k] = self.Prior.t + sigma2 * (PhiShape[k] - 1)
                assert np.all(PhiInvScale[k]) > 0
                hInvScale[k] = self.Prior.g * np.ones(C)
                WMean[k] = np.dot(eigVec[:,:C], np.diag(np.sqrt(eigVal[:C] - sigma2)))
                E_W_WT = np.zeros((D, C, C))
                for d in xrange(D):
                    E_W_WT[d] = np.outer(WMean[k,d], WMean[k,d])
                E_WT_Phi_W = np.sum((PhiShape[k] / PhiInvScale[k])[:, np.newaxis, np.newaxis]
                                     * E_W_WT, axis=0)
                aCov[k] = inv(np.eye(C) + E_WT_Phi_W)

        xaT, aaT = self.get_xaT_aaT(SS, WMean, PhiShape, PhiInvScale, aCov)
        WMean, WCov = self.calcPostW(SS, xaT, aaT, hShape, hInvScale, PhiShape, PhiInvScale)
        hShape, hInvScale = self.calcPostH(WMean, WCov)
        PhiShape, PhiInvScale = self.calcPostPhi(SS, xaT, aaT, WMean, WCov)
        aCov = self.calcPostACov(WMean, WCov, PhiShape, PhiInvScale)

        if len(newCompList) > 0:
            idx = newCompList
            for i in xrange(nIter4NewComp - 1):
                xaT, aaT = self.get_xaT_aaT(SS, WMean, PhiShape, PhiInvScale, aCov)
                WMean[newCompList], WCov[newCompList] = \
                    self.calcPostW(SS, xaT, aaT, hShape, hInvScale, PhiShape, PhiInvScale, idx=idx)
                hShape, hInvScale[newCompList] = \
                    self.calcPostH(WMean, WCov, idx=idx)
                PhiShape[newCompList], PhiInvScale[newCompList] = \
                    self.calcPostPhi(SS, xaT, aaT, WMean, WCov, idx=idx)
                aCov[newCompList] = self.calcPostACov(WMean, WCov, PhiShape, PhiInvScale, idx=idx)
        return WMean, WCov, hShape, hInvScale, PhiShape, PhiInvScale, aCov

    @staticmethod
    def get_xaT_aaT(SS, WMean, PhiShape, PhiInvScale, aCov):
        K = SS.K
        D = SS.D
        C = SS.C
        xaT = np.zeros((K,D,C))
        aaT = np.zeros((K,C,C))
        for k in xrange(K):
            LU = np.dot(aCov[k], np.dot(WMean[k].T, np.diag(PhiShape[k] / PhiInvScale[k])))
            aaT[k] = SS.N[k] * aCov[k] + np.dot(LU, np.inner(SS.xxT[k], LU))
            xaT[k] = np.inner(SS.xxT[k], LU)
        return xaT, aaT

    # @profile
    def calcPostW(self, SS, xaT, aaT, hShape, hInvScale, PhiShape, PhiInvScale, idx=None):
        if idx is None:
            K = SS.K
            idx = xrange(K)
        else:
            K = len(idx)
        C = self.C
        D = self.D
        WCov = np.zeros((K, D, C, C))
        WMean = np.zeros((K, D, C))
        for i, k in enumerate(idx):
            SigmaInvWW = np.diag(hShape / hInvScale[k]) \
                         + (PhiShape[k] / PhiInvScale[k])[:,np.newaxis,np.newaxis] \
                         * np.tile(aaT[k], (D,1,1))
            WCov[i] = inv(SigmaInvWW)
            for d in xrange(D):
                WMean[i, d] = np.dot(WCov[i,d], PhiShape[k] / PhiInvScale[k,d] * xaT[k,d])
        return WMean, WCov

    # @profile
    def calcPostH(self, WMean, WCov, idx=None):
        if idx is None:
            K = WMean.shape[0]
            idx = xrange(K)
        else:
            K = len(idx)
        C = self.C
        hShape = self.Prior.f + .5 * self.D
        hInvScale = np.zeros((K, C))
        for i, k in enumerate(idx):
            for c in xrange(C):
                hInvScale[i,c] = self.Prior.g + .5 * np.sum(WMean[k,:,c]**2 + WCov[k,:,c,c])
        return hShape, hInvScale

    # @profile
    def calcPostPhi(self, SS, xaT, aaT, WMean, WCov, idx=None):
        if idx is None:
            K = SS.K
            idx = xrange(K)
        else:
            K = len(idx)
        D = self.D
        PhiShape = self.Prior.s * np.ones(K)
        PhiInvScale = self.Prior.t * np.ones((K, D))
        for i, k in enumerate(idx):
            PhiShape[i] += .5 * SS.N[k]
            for d in xrange(D):
                PhiInvScale[i,d] += .5 * (SS.xxT[k,d,d]
                                    - 2 * np.dot(xaT[k,d], WMean[k,d])
                                    + np.einsum('ij,ji', np.outer(WMean[k,d], WMean[k,d])
                                                + WCov[k,d], aaT[k]))
        return PhiShape, PhiInvScale

    def calcPostACov(self, WMean, WCov, PhiShape, PhiInvScale, idx=None):
        if idx is None:
            K = WMean.shape[0]
            idx = xrange(K)
        else:
            K = len(idx)
        C = self.C
        D = self.D
        aCov = np.zeros((K, C, C))
        for i, k in enumerate(idx):
            E_W_WT = np.zeros((D, C, C))
            for d in xrange(D):
                E_W_WT[d] = np.outer(WMean[k][d], WMean[k][d]) + WCov[k][d]
            E_WT_Phi_W = np.sum((PhiShape[k] / PhiInvScale[k])[:, np.newaxis, np.newaxis]
                                * E_W_WT, axis=0)
            aCov[i] = inv(np.eye(C) + E_WT_Phi_W)
        return aCov

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

    ########################################################### Merge
    ###########################################################
    def elbo4comp(self, SS, WMean, WCov, hShape, hInvScale, PhiShape, PhiInvScale,  aCov):
        elbo = 0.0
        C = self.C
        D = self.D
        Prior = self.Prior
        for d in xrange(D):
            elbo += - PhiShape * np.log(PhiInvScale[d]) \
                    + Prior.s * np.log(Prior.t) \
                    + gammaln(PhiShape) - gammaln(Prior.s) \
                    - (PhiShape - Prior.s) * \
                      (psi(PhiShape) - np.log(PhiInvScale[d])) \
                    + PhiShape * (1 - Prior.t / PhiInvScale[d])
        for c in xrange(C):
            elbo += - hShape * np.log(hInvScale[c]) \
                    + Prior.f * np.log(Prior.g) \
                    + gammaln(hShape) - gammaln(Prior.f)
        for d in xrange(D):
            elbo += .5 * (np.prod(slogdet(WCov[d])) + C)
        L = aCov
        LU = np.dot(L, np.dot(WMean.T, np.diag(PhiShape / PhiInvScale)))
        aaT = SS.N * L + np.dot(LU, np.inner(SS.xxT, LU))
        xaT = np.inner(SS.xxT, LU)
        E_WWT = np.zeros((D,C,C))
        for d in xrange(D):
            E_WWT[d] = np.outer(WMean[d], WMean[d]) + WCov[d]
        E_WT_invPsi_W = np.sum((PhiShape / PhiInvScale)[:, np.newaxis, np.newaxis]
                               * E_WWT, axis=0)
        elbo += .5 * (np.prod(slogdet(L)) + C) * SS.N \
                - .5 * np.trace(aaT)
        elbo += -.5 * D * SS.N * LOGTWOPI \
                + .5 * (np.sum(psi(PhiShape) - np.log(PhiInvScale))) * SS.N \
                - .5 * np.dot(PhiShape / PhiInvScale, np.diag(SS.xxT)) \
                + np.dot(PhiShape / PhiInvScale, np.einsum('ij, ij->i', xaT, WMean)) \
                - .5 * np.einsum('ij,ji', E_WT_invPsi_W, aaT)
        return elbo

    def calcHardMergeGap(self, SS, kA, kB):
        ''' Calculate change in ELBO after a hard merge applied to this model

            Returns
            ---------
            gap : scalar real, indicates change in ELBO after merge of kA, kB
        '''
        C = self.C
        D = self.D
        Prior = self.Prior
        elboA = self.elbo4comp(SS.getComp(kA), self.Post.WMean[kA], self.Post.WCov[kA],
                          self.Post.hShape, self.Post.hInvScale[kA],
                          self.Post.PhiShape[kA], self.Post.PhiInvScale[kA], self.Post.aCov[kA])
        elboB = self.elbo4comp(SS.getComp(kB), self.Post.WMean[kB], self.Post.WCov[kB],
                          self.Post.hShape, self.Post.hInvScale[kB],
                          self.Post.PhiShape[kB], self.Post.PhiInvScale[kB], self.Post.aCov[kB])
        SS_AB = SuffStatBag(K=1, D=D, C=C)
        SS_AB.setField('N', SS.N[kA] + SS.N[kB], dims='')
        SS_AB.setField('xxT', SS.xxT[kA] + SS.xxT[kB], dims=('D','D'))
        WMean, WCov, hShape, hInvScale, PhiShape, PhiInvScale, aCov = \
             self.calcPostParamsForComp(SS, kA=kA, kB=kB)
        elboAB = self.elbo4comp(SS_AB, WMean, WCov, hShape, hInvScale, PhiShape, PhiInvScale, aCov)
        return - elboA - elboB + elboAB

    def calcHardMergeGap_AllPairs(self, SS):
        ''' Calculate change in ELBO for all candidate hard merge pairs

        Returns
        ---------
        Gap : 2D array, size K x K, upper-triangular entries non-zero
              Gap[j,k] : scalar change in ELBO after merge of k into j
        '''
        Post = self.Post
        Prior = self.Prior
        e = np.zeros(SS.K)
        for k in xrange(SS.K):
            WMean = Post.WMean[k]
            WCov = Post.WCov[k]
            hShape = Post.hShape
            hInvScale = Post.hInvScale[k]
            PhiShape = Post.PhiShape[k]
            PhiInvScale = Post.PhiInvScale[k]
            aCov = Post.aCov[k]
            e[k] = self.elbo4comp(SS.getComp(k), WMean, WCov, hShape, hInvScale, PhiShape, PhiInvScale, aCov)
        Gap = np.zeros((SS.K, SS.K))
        for j in xrange(SS.K):
            for k in xrange(j + 1, SS.K):
                WMean, WCov, hShape, hInvScale, PhiShape, PhiInvScale, aCov \
                    = self.calcPostParamsForComp(SS, j, k)
                ejk = self.elbo4comp(SS.getComp(j) + SS.getComp(k), WMean, WCov,
                                     hShape, hInvScale, PhiShape, PhiInvScale, aCov)
                Gap[j, k] = - e[j] - e[k] + ejk
        return Gap

    def calcHardMergeGap_SpecificPairs(self, SS, PairList):
        ''' Calc change in ELBO for specific list of candidate hard merge pairs

        Returns
        ---------
        Gaps : 1D array, size L
              Gap[j] : scalar change in ELBO after merge of pair in PairList[j]
        '''
        Gaps = np.zeros(len(PairList))
        for ii, (kA, kB) in enumerate(PairList):
            Gaps[ii] = self.calcHardMergeGap(SS, kA, kB)
        return Gaps

    def calcPostParamsForComp(self, SS, kA=None, kB=None, nIter4NewComp=5):
        D = self.D
        C = self.C
        if kB is None:
            nIter = 1
            SN = SS.N[kA]
            SxxT = SS.xxT[kA]
        elif kB == -1 and SS.K == 1:
            nIter = nIter4NewComp
            SN = SS.N[kA]
            SxxT = SS.xxT[kA]
        else:
            nIter = nIter4NewComp
            SN = SS.N[kA] + SS.N[kB]
            SxxT = SS.xxT[kA] + SS.xxT[kB]
            if np.allclose(SxxT,0) and np.allclose(SN,0):
                PRNG = np.random.RandomState(0)
                eigVal = PRNG.randn(D)
                eigVec = PRNG.randn(D, D)
            else:
                eigVal, eigVec = eig(SxxT / SN)
                idx = np.argsort(eigVal)[::-1]
                eigVal = eigVal[idx]
                eigVec = eigVec[:, idx]
            PhiShape = self.Prior.s + .5 * SN
            sigma2 = np.sum(eigVal[C:]) / (D - C)
            PhiInvScale = self.Prior.t + sigma2 * (PhiShape - 1) * np.ones(D)
            WMean = np.dot(eigVec[:,:C], np.diag(np.sqrt(eigVal[:C] - sigma2)))
            assert np.all(PhiInvScale) > 0
            hShape = self.Prior.f
            hInvScale = self.Prior.g * np.ones(C)
            E_W_WT = np.zeros((D, C, C))
            for d in xrange(D):
                E_W_WT[d] = np.outer(WMean[d], WMean[d])
            E_WT_Phi_W = np.sum((PhiShape / PhiInvScale)[:, np.newaxis, np.newaxis]
                                 * E_W_WT, axis=0)
            aCov = inv(np.eye(C) + E_WT_Phi_W)

        for i in xrange(nIter):
            # get xaT, aaT
            LU = np.dot(aCov, np.dot(WMean.T, np.diag(PhiShape / PhiInvScale)))
            aaT = SN * aCov + np.dot(LU, np.inner(SxxT, LU))
            xaT = np.inner(SxxT, LU)
            # calc W
            SigmaInvWW = np.diag(hShape / hInvScale) \
                        + (PhiShape / PhiInvScale)[:,np.newaxis,np.newaxis] \
                        * np.tile(aaT, (D,1,1))
            WCov = inv(SigmaInvWW)
            for d in xrange(D):
                WMean[d] = np.dot(WCov[d], PhiShape / PhiInvScale[d] * xaT[d])
            # calc H
            hInvScale = np.zeros(C)
            for c in xrange(C):
                hInvScale[c] = self.Prior.g + .5 * np.sum(WMean[:,c]**2 + WCov[:,c,c])
            # calc Phi
            PhiShape = self.Prior.s + .5 * SN
            PhiInvScale = self.Prior.t * np.ones(D)
            for d in xrange(D):
                PhiInvScale[d] += .5 * (SxxT[d,d] - 2 * np.dot(xaT[d], WMean[d])
                              + np.einsum('ij,ji', np.outer(WMean[d], WMean[d]) + WCov[d], aaT))
            # calc aCov
            E_W_WT = np.zeros((D, C, C))
            for d in xrange(D):
                E_W_WT[d] = np.outer(WMean[d], WMean[d]) + WCov[d]
            E_WT_Phi_W = np.sum((PhiShape / PhiInvScale)[:, np.newaxis, np.newaxis]
                                * E_W_WT, axis=0)
            aCov = inv(np.eye(C) + E_WT_Phi_W)
        return WMean, WCov, hShape, hInvScale, PhiShape, PhiInvScale, aCov

    def setPostFactors(self, aCov=None,
                       WMean=None, WCov=None, PhiShape=None, PhiInvScale=None,
                       hShape=None, hInvScale=None, **kwargs):
        self.Post = ParamBag(K=WMean.shape[0], D=WMean.shape[1], C=WMean.shape[2])
        self.Post.setField('WMean', WMean, dims=('K','D','C'))
        self.Post.setField('WCov', WCov, dims=('K','D','C','C'))
        self.Post.setField('hShape', hShape)
        if hInvScale.ndim == 1:
            hInvScale = hInvScale[:, np.newaxis]
        self.Post.setField('hInvScale', hInvScale, dims=('K','C'))
        self.Post.setField('PhiShape', PhiShape, dims=('K'))
        self.Post.setField('PhiInvScale', PhiInvScale, dims=('K','D'))
        self.Post.setField('aCov', aCov, dims=('K','C','C'))

if __name__ == '__main__':
    import bnpy, profile
    import StarCovarK5
    Data = StarCovarK5.get_data(nObsTotal=10000)
    import matplotlib.pylab as plt
    hmodel, RInfo = bnpy.run(Data, 'DPMixtureModel',
                            'ZeroMeanFactorAnalyzer', 'moVB',
                            C=1, nLap=1000, K=20, printEvery=10,
                            moves='merge', mergeStartLap=10)
    # hmodel = bnpy.load_model('/Users/Geng/Documents/Brown/research/patch/FAPY/StarCovarK5/defaultjob/1')
    # from bnpy.viz import PlotELBO
    # PlotELBO.plotJobsThatMatchKeywords('StarCovarK5/defaultjob');

    from bnpy.viz.GaussViz import plotGauss2DContour, Colors
    import matplotlib.pylab as plt
    for k in xrange(hmodel.obsModel.K):
        plt.figure(k)
        sigma = np.inner(hmodel.obsModel.Post.WMean[k], hmodel.obsModel.Post.WMean[k]) \
                + np.sum(np.diagonal(hmodel.obsModel.Post.WCov[k]),axis=1) \
                + np.diag(hmodel.obsModel.Post.PhiInvScale[k] /
                          (hmodel.obsModel.Post.PhiShape[k] - 1))
        plotGauss2DContour(np.zeros(hmodel.obsModel.D), sigma)
        plt.axis([-1, 1, -1, 1])
    plt.show()

    # LP = hmodel.calc_local_params(Data)
    # SS = hmodel.get_global_suff_stats(Data, LP, doPrecompEntropy=True,
    #                                   doPrecompMergeEntropy=True, mPairIDs=[(3,5)])
    # print SS.N
    # ELBO = hmodel.calc_evidence(Data=Data, SS=SS, LP=LP)
    # print 'Original elbo is: ' + str(ELBO)
    #
    # from bnpy.mergemove import MergeMove
    # MergeMove.run_many_merge_moves(hmodel, SS, ELBO, [(3,5)])


    # from IPython import embed; embed()
    # SS.mergeComps(3,5)
    # hmodel.update_global_params(SS, mergeCompA=3, mergeCompB=5)
    # print 'Merged elbo is: ' + str(hmodel.calc_evidence(Data=Data, SS=SS))

    # propModel = hmodel.copy()
    # propSS = SS.copy()
    # kdel = [6,1]
    # for k in kdel:
    #     propSS.removeComp(k)
    # propModel.update_global_params(propSS, kdel=kdel)
    # print propModel.calc_evidence(Data, propSS)


    # hmodel, RInfo = bnpy.run('D2C1K2_ZM', 'DPMixtureModel',
    #                         'ZeroMeanFactorAnalyzer', 'moVB',
    #                         C=1, nLap=500, nObsTotal=10000, K=10,
    #                         moves='merge', mergeStartLap=1)

    # hmodel, RInfo = bnpy.run('D3C2K2_ZM', 'DPMixtureModel',
    #                         'ZeroMeanFactorAnalyzer', 'moVB',
    #                         C=2, nLap=500, nObsTotal=10000, K=20, moves='merge')