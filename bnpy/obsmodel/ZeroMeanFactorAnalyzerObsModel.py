import numpy as np
from bnpy.suffstats import ParamBag, SuffStatBag
from bnpy.obsmodel.AbstractObsModel import AbstractObsModel
from numpy.linalg import inv, solve, det, slogdet, eig, LinAlgError
from scipy.linalg import eigh
from scipy.special import psi, gammaln
from bnpy.util import dotATA
from bnpy.util import LOGTWOPI, EPS
from bnpy.data import XData
from IPython import embed

class ZeroMeanFactorAnalyzerObsModel(AbstractObsModel):

    def __init__(self, inferType='VB', Data=None, C = None,
                 WCovType = None, nPostUpdate=None, calcXxT=None,
                 **PriorArgs):
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
        WCovType = str(WCovType).lower()
        if WCovType == 'diag' or WCovType == 'full':
            self.WCovType = WCovType
        elif WCovType == 'none': # early result, full WCov
            self.WCovType = 'full'
        else:
            raise NameError('Unrecognized WCov type: %s.' % WCovType)
        self.nPostUpdate = nPostUpdate
        self.calcXxT = calcXxT
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
        msg = '\nGamma shape and invScale param on precision of factor loading matrices: f, g\n'
        sfx = ' '
        msg += 'f = %s,%s' % (str(self.Prior.f), sfx)
        msg += 'g = %s' % (str(self.Prior.g))
        msg += '\nGamma shape and invScale param on precision of diagonal noise matrices: s, t\n'
        msg += 's = %s,%s' % (str(self.Prior.s), sfx)
        msg += 't = %s' % (str(self.Prior.t))
        msg += '\nLatent space dimension C = %s, WCovType = %s' % (str(self.C), self.WCovType)
        msg = msg.replace('\n', '\n  ')
        return msg

    ########################################################### Suff Stats
    ###########################################################
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

        sqrtResp = np.sqrt(resp)
        if self.calcXxT:
            # Expected outer-product of x for each k
            xxT = np.zeros( (K, D, D) )
            for k in xrange(K):
                xxT[k] = dotATA(sqrtResp[:,k][:,np.newaxis] * X)
            SS.setField('xxT', xxT, dims=('K','D','D'))
        else:
            aMean = LP['aMean']
            # xaT and aaT
            xaT = np.zeros( (K, D, C) )
            aaT = np.zeros( (K, C, C) )
            x2 = X**2
            for k in xrange(K):
                xaT[k] = np.dot(X.T, resp[:,k][:,np.newaxis] * aMean[k])
                aaT[k] = dotATA(sqrtResp[:,k][:,np.newaxis] * aMean[k])
            diagXxT = np.dot(resp.T, x2)
            SS.setField('xaT', xaT, dims=('K','D','C'))
            SS.setField('aaT', aaT, dims=('K','C','C'))
            SS.setField('diagXxT', diagXxT, dims=('K','D'))
        return SS

  ########################################################### Local step
  ###########################################################
    def calc_local_params(self, Data, LP=None, **kwargs):
        if LP is None:
            LP = dict()
        if self.inferType == 'EM':
            raise NotImplementedError()
        else:
            LP['aMean'] = self.calcAMean_FromPost(Data)
            LP['E_log_soft_ev'] = self.calcLogSoftEvMatrix_FromPost(Data, LP)
        return LP

    def calcAMean_FromPost(self, Data):
        '''  Compute the posterior mean of a_k for each data point
        See Eq. in the writeup

        Returns
        --------
        aMean : 3D array, size K x N x C
        '''
        N = Data.nObs
        K = self.Post.K
        D = self.D
        C = self.C
        # calculate aMean
        aMean = np.zeros((K, N, C))
        for k in xrange(K):
            aCovk_WMeankT_invPsi = np.inner(self.Post.aCov[k], self.Post.WMean[k]) * \
                                   (self.Post.PhiShape[k] / self.Post.PhiInvScale[k])
            aMean[k] = np.inner(Data.X, aCovk_WMeankT_invPsi)
        return aMean

    def calcLogSoftEvMatrix_FromPost(self, Data, LP):
        N = Data.nObs
        K = self.Post.K
        L = np.zeros((N, K))
        DataX2 = Data.X**2
        for k in xrange(K):
            L[:,k] = .5 * np.einsum('ij,ji->i', LP['aMean'][k],
                                    np.inner(inv(self.Post.aCov[k]), LP['aMean'][k])) \
                     - .5 * np.inner(self.Post.PhiShape[k] / self.Post.PhiInvScale[k], DataX2) \
                     + .5 * np.sum(psi(self.Post.PhiShape[k]) - np.log(self.Post.PhiInvScale[k])) \
                     + .5 * np.prod(slogdet(self.Post.aCov[k]))
            if not np.all(np.isfinite(L[:,k])):
                embed()
        return L

  ########################################################### Global step
  ###########################################################
    def updatePost(self, SS, **kwargs):
        self.ClearCache()
        WMean, WCov, hShape, hInvScale, PhiShape, PhiInvScale, aCov = self.calcPostParams(SS, **kwargs)
        self.Post = ParamBag(K=SS.K, D=SS.D, C=SS.C)
        self.Post.setField('WMean', WMean, dims=('K','D','C'))
        if self.WCovType == 'diag':
            self.Post.setField('WCov', WCov, dims=('K','D','C'))
        elif self.WCovType == 'full':
            self.Post.setField('WCov', WCov, dims=('K','D','C','C'))
        self.Post.setField('hShape', hShape)
        self.Post.setField('hInvScale', hInvScale, dims=('K','C'))
        self.Post.setField('PhiShape', PhiShape, dims=('K'))
        self.Post.setField('PhiInvScale', PhiInvScale, dims=('K','D'))
        self.Post.setField('aCov', aCov, dims=('K','C','C'))
        self.K = SS.K
        assert self.K == self.Post.K

    def calcPostParams(self, SS):
        C, D, K = (SS.C, SS.D, SS.K)
        WMean = np.zeros((K,D,C))
        if self.WCovType == 'diag':
            WCov = np.zeros((K,D,C))
        elif self.WCovType == 'full':
            WCov = np.zeros((K,D,C,C))
        hShape = 0
        hInvScale = np.zeros((K,C))
        PhiShape = np.zeros(K)
        PhiInvScale = np.zeros((K,D))
        aCov = np.zeros((K,C,C))

        # initialization
        for k in xrange(K):
            if hasattr(self, 'Post'):
                if self.calcXxT:  # PPCA
                    WMean[k], hShape, hInvScale[k], PhiShape[k], PhiInvScale[k], aCov[k] = self.initPostParams(SS.getComp(k))
                else: # Old values
                    Post = self.Post
                    WMean[k], hShape, hInvScale[k], PhiShape[k], PhiInvScale[k], aCov[k] = \
                        (Post.WMean[k], Post.hShape, Post.hInvScale[k], Post.PhiShape[k], Post.PhiInvScale[k], Post.aCov[k])
            else: # Sampling from prior
                Prior = self.Prior
                PRNG = np.random.RandomState(k)
                h = PRNG.gamma(Prior.f, 1./Prior.g, C)
                hShape = Prior.f
                hInvScale[k] = Prior.g
                sigma = 1./np.sqrt(h)
                for c in xrange(C):
                    WMean[k,:,c] = PRNG.normal(0., sigma[c], D)
                Phi = PRNG.gamma(Prior.s, 1./Prior.t, D)
                PhiShape[k] = Prior.s
                PhiInvScale[k] = Prior.t
                E_WT_Phi_W = np.sum(Phi[:, np.newaxis, np.newaxis] * np.einsum('ij,ik->ijk', WMean[k], WMean[k]), axis=0)
                aCov[k] = inv(np.eye(C) + E_WT_Phi_W)

        # global update
        for i in xrange(self.nPostUpdate):
            for k in xrange(K):
                xaT, aaT = self.get_xaT_aaT(SS.getComp(k), WMean[k], PhiShape[k], PhiInvScale[k], aCov[k])
                WMean[k], WCov[k] = self.calcPostW(xaT, aaT, hShape, hInvScale[k], PhiShape[k], PhiInvScale[k])
                E_WT_W = self.calcE_WT_W(WMean[k], WCov[k])
                hShape, hInvScale[k] = self.calcPostH(WMean[k], WCov[k])
                PhiShape[k], PhiInvScale[k] = self.calcPostPhi(SS.getComp(k), xaT, aaT, WMean[k], E_WT_W)
                aCov[k] = self.calcPostACov(E_WT_W, PhiShape[k], PhiInvScale[k])
        return WMean, WCov, hShape, hInvScale, PhiShape, PhiInvScale, aCov

    def initPostParams(self, SS):
        ''' Initialize post parameters given SS
        See Eq. in the writeup

        Returns
        --------
        WMean, hShape, hInvScale, PhiShape, PhiInvScale, aCov
        '''
        C, D = self.C, self.D
        # eigen decomp on empirical cov
        if np.allclose(SS.N, 0):
            # generate a positive definite matrix randomly
            PRNG = np.random.RandomState(0)
            empCov = PRNG.normal(size=(D,C))
            empCov = np.dot(empCov, empCov.T)
        else:
            empCov = SS.xxT / SS.N
        eigVal, eigVec = eigh(empCov, eigvals=(D-C,D-1))
        sigma2 = (np.trace(empCov) - np.sum(eigVal)) / (D-C)
        if sigma2 <= EPS or not np.all(eigVal - sigma2 >= EPS):
            assert np.allclose(sigma2, 0)
            assert np.allclose(eigVal[eigVal<EPS], 0)
            assert np.allclose((eigVal-sigma2)[eigVal-sigma2<EPS], 0)
            sigma2 = EPS
            eigVal[eigVal<EPS] = EPS
        # WMean
        WMean = np.dot(eigVec[:,:C], np.diag(np.sqrt(eigVal - sigma2)))
        # h
        hShape = self.Prior.f + .5 * D
        hInvScale = self.Prior.g * np.ones(C) + .5 * np.sum(WMean**2, axis=0)
        # Phi
        PhiShape = self.Prior.s + .5 * SS.N
        PhiInvScale = sigma2 * PhiShape * np.ones(D)
        assert np.all(PhiInvScale > 0)
        # aCov
        E_WT_Phi_W = np.sum((PhiShape / PhiInvScale)[:, np.newaxis, np.newaxis]
                             * np.einsum('ij,ik->ijk', WMean, WMean), axis=0)
        aCov = inv(np.eye(C) + E_WT_Phi_W)
        return WMean, hShape, hInvScale, PhiShape, PhiInvScale, aCov

    @staticmethod
    def get_xaT_aaT(SS, WMean, PhiShape, PhiInvScale, aCov):
        ''' Construct xaT and aaT given SS and post parameters
        See Eq. in the writeup

        Returns
        --------
        xaT : 3D array, size K x D x C
        aaT : 3D array, size K x C x C
        '''
        if hasattr(SS,'xaT') and hasattr(SS, 'aaT'):
            xaT = SS.xaT
            aaT = SS.aaT + SS.N * aCov
        else:
            LU = np.inner(aCov, (WMean * PhiShape / PhiInvScale[:, np.newaxis]))
            xaT = np.inner(SS.xxT, LU)
            aaT = SS.N * aCov + np.dot(LU, xaT)
        return xaT, aaT

    def calcPostW(self, xaT, aaT, hShape, hInvScale, PhiShape, PhiInvScale):
        ''' Compute posterior mean and covariance for each W_{kd}
        See Eq. in the writeup

        Returns
        --------
        WMean : 3D array, size K x D x C
        WCov: if WCovType is 'diag', 3D array, size K x D x C (by default)
              if WCovTpye is 'full', 4D array, size K x D x C x C
        '''
        C = self.C
        D = self.D
        scaled_xaT = PhiShape / PhiInvScale[:,np.newaxis] * xaT
        if self.WCovType == 'diag':
            WCov = 1. / (hShape / hInvScale
                         + (PhiShape / PhiInvScale)[:,np.newaxis]
                         * np.tile(np.diag(aaT),(D,1)))
            WMean = WCov * scaled_xaT
        elif self.WCovType == 'full':
            SigmaInvWW = np.diag(hShape / hInvScale) \
                         + (PhiShape / PhiInvScale)[:,np.newaxis,np.newaxis] \
                         * np.tile(aaT, (D,1,1))
            WCov = inv(SigmaInvWW)
            WMean = np.einsum('ijk,i...k->ij', WCov, scaled_xaT)
        return WMean, WCov

    def calcE_WT_W(self, WMean, WCov):
        C, D = self.C, self.D
        E_WT_W = np.einsum('ij,ik->ijk', WMean, WMean)
        if self.WCovType == 'diag':
            diagIdx = np.arange(C*D) / C * C**2 + np.tile(np.ravel_multi_index(np.diag_indices(C), (C,C)), D)
            diagIdx = np.unravel_index(diagIdx, E_WT_W.shape)
            E_WT_W[diagIdx] += WCov.flatten()
        elif self.WCovType == 'full':
            E_WT_W += WCov
        return E_WT_W

    def calcPostH(self, WMean, WCov):
        ''' Compute posterior shape and inverse-scale parameters for each h_c
        See Eq. in the writeup

        Returns
        --------
        hShape : scalar
        hInvScale: 2D array, size K x C
        '''
        C = self.C
        hShape = self.Prior.f + .5 * self.D
        hInvScale = self.Prior.g * np.ones(C)
        if self.WCovType == 'diag':
            hInvScale += .5 * np.sum(WMean**2 + WCov, axis=0)
        elif self.WCovType == 'full':
            hInvScale += .5 * np.sum(WMean**2 + np.diagonal(WCov,axis1=1,axis2=2), axis=0)
        return hShape, hInvScale

    def calcPostPhi(self, SS, xaT, aaT, WMean, E_WT_W):
        ''' Compute posterior shape and inverse-scale parameters for each Phi_d
        See Eq. in the writeup

        Returns
        --------
        PhiShape : 1D array, size K
        PhiInvScale: 2D array, size K x D
        '''
        D = self.D
        PhiShape = self.Prior.s + .5 * SS.N
        PhiInvScale = self.Prior.t * np.ones(D)
        if self.calcXxT:
            PhiInvScale += 0.5 * np.diag(SS.xxT)
        else:
            PhiInvScale += 0.5 * SS.diagXxT
        PhiInvScale += .5 * (- 2 * np.einsum('ij,ij->i', xaT, WMean)
                             + np.einsum('ijk,...kj->i', E_WT_W, aaT))
        assert np.all(PhiInvScale > 0)
        return PhiShape, PhiInvScale

    def calcPostACov(self, E_WT_W, PhiShape, PhiInvScale):
        ''' Compute posterior covariance for each a_k
        See Eq. in the writeup

        Returns
        --------
        aCov: 3D array, size K x C x C
        '''
        C, D = self.C, self.D
        E_WT_Phi_W = np.sum((PhiShape / PhiInvScale)[:, np.newaxis, np.newaxis] * E_WT_W, axis=0)
        aCov = inv(np.eye(C) + E_WT_Phi_W)
        return aCov

    ########################################################### VB ELBO step
    ###########################################################
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
        elbo = 0.0
        Post = self.Post
        for k in xrange(SS.K):
            elbo += self.elbo4Comp(SS.getComp(k), Post.WMean[k], Post.WCov[k],
                                   Post.hShape, Post.hInvScale[k],
                                   Post.PhiShape[k], Post.PhiInvScale[k], Post.aCov[k])
        return elbo

    def elbo4Comp(self, SS, WMean, WCov, hShape, hInvScale, PhiShape, PhiInvScale, aCov):
        elbo = 0.0
        C, D, Prior = self.C, self.D, self.Prior

        # terms related with Phi
        elbo += - PhiShape * np.sum(np.log(PhiInvScale)) \
                + D * Prior.s * np.log(Prior.t) \
                + D * (gammaln(PhiShape) - gammaln(Prior.s)) \
                - (PhiShape - Prior.s) * \
                  (D * psi(PhiShape) - np.sum(np.log(PhiInvScale))) \
                + PhiShape * np.sum(1 - Prior.t / PhiInvScale)

        # terms related with h
        elbo += - hShape * np.sum(np.log(hInvScale)) \
                + C * Prior.f * np.log(Prior.g) \
                + C * (gammaln(hShape) - gammaln(Prior.f)) \
                - (hShape - Prior.f) * \
                  (C * psi(hShape) - np.sum(np.log(hInvScale))) \
                + hShape * np.sum(1 - Prior.g / hInvScale)

        # terms related with W
        E_WT_W = self.calcE_WT_W(WMean, WCov)
        if self.WCovType == 'diag':
            sumLogDetWCov = np.sum(np.sum(np.log(WCov), axis=1))
        elif self.WCovType == 'full':
            sumLogDetWCov = np.dot(slogdet(WCov)[0], slogdet(WCov)[1])
        elbo += .5 * sumLogDetWCov + .5 * D * (C + np.sum(psi(hShape) - np.log(hInvScale))) \
                - .5 * np.dot(np.sum(np.diagonal(E_WT_W, axis1=1, axis2=2), axis=0), hShape / hInvScale)

        # terms related with a
        if self.calcXxT:
            xaT, aaT = self.get_xaT_aaT(SS, WMean, PhiShape, PhiInvScale, aCov)
        else:
            xaT = SS.xaT
            aaT = SS.aaT + SS.N * aCov
        elbo += .5 * (np.prod(slogdet(aCov)) + C) * SS.N - .5 * np.trace(aaT)

        # terms related with x
        E_WT_Phi_W = np.sum((PhiShape / PhiInvScale)[:, np.newaxis, np.newaxis] * E_WT_W, axis=0)
        elbo += - .5 * (D * LOGTWOPI - np.sum(psi(PhiShape) - np.log(PhiInvScale))) * SS.N \
                + np.dot(PhiShape / PhiInvScale, np.einsum('ij, ij->i', xaT, WMean)) \
                - .5 * np.einsum('ij,ji', E_WT_Phi_W, aaT)
        if self.calcXxT:
            elbo += - .5 * np.dot(PhiShape / PhiInvScale, np.diag(SS.xxT))
        else:
            elbo += - .5 * np.dot(PhiShape / PhiInvScale, SS.diagXxT)
        return elbo

    def getDatasetScale(self, SS):
        ''' Get scale factor for dataset, indicating number of observed scalars.

            Used for normalizing the ELBO so it has reasonable range.

            Returns
            ---------
            s : scalar positive integer
        '''
        return SS.N.sum() * SS.D


    ########################################################### Merge
    ###########################################################
    def calcHardMergeGap(self, SS, kA, kB):
        ''' Calculate change in ELBO after a hard merge applied to this model

            Returns
            ---------
            gap : scalar real, indicates change in ELBO after merge of kA, kB
        '''
        if not self.calcXxT:
            raise NotImplementedError('Merge move is not allowed for aaT and xaT!')
        C = self.C
        D = self.D
        elboA = self.elbo4Comp(SS.getComp(kA), self.Post.WMean[kA], self.Post.WCov[kA],
                          self.Post.hShape, self.Post.hInvScale[kA],
                          self.Post.PhiShape[kA], self.Post.PhiInvScale[kA], self.Post.aCov[kA])
        elboB = self.elbo4Comp(SS.getComp(kB), self.Post.WMean[kB], self.Post.WCov[kB],
                          self.Post.hShape, self.Post.hInvScale[kB],
                          self.Post.PhiShape[kB], self.Post.PhiInvScale[kB], self.Post.aCov[kB])
        # WMean, WCov, hShape, hInvScale, PhiShape, PhiInvScale, aCov = \
        #     self.calcPostParams4Comp(SS, kA=kA)
        # elboA = self.elbo4Comp(SS.getComp(kA), WMean, WCov,
        #                   hShape, hInvScale, PhiShape, PhiInvScale, aCov)
        # WMean, WCov, hShape, hInvScale, PhiShape, PhiInvScale, aCov = \
        #     self.calcPostParams4Comp(SS, kA=kB)
        # elboB = self.elbo4Comp(SS.getComp(kB), WMean, WCov,
        #                   hShape, hInvScale, PhiShape, PhiInvScale, aCov)
        SS_AB = SuffStatBag(K=1, D=D, C=C)
        SS_AB.setField('N', SS.N[kA] + SS.N[kB], dims='')
        SS_AB.setField('xxT', SS.xxT[kA] + SS.xxT[kB], dims=('D','D'))
        WMean, WCov, hShape, hInvScale, PhiShape, PhiInvScale, aCov = \
             self.calcPostParams4Comp(SS, kA=kA, kB=kB)
        elboAB = self.elbo4Comp(SS_AB, WMean, WCov, hShape, hInvScale, PhiShape, PhiInvScale, aCov)
        return - elboA - elboB + elboAB

    def calcHardMergeGap_AllPairs(self, SS):
        ''' Calculate change in ELBO for all candidate hard merge pairs

        Returns
        ---------
        Gap : 2D array, size K x K, upper-triangular entries non-zero
              Gap[j,k] : scalar change in ELBO after merge of k into j
        '''
        if not self.calcXxT:
            raise NotImplementedError('Merge move is not allowed for aaT and xaT!')
        e = np.zeros(SS.K)
        for k in xrange(SS.K):
            Post = self.Post
            WMean, WCov, hShape, hInvScale = Post.WMean[k], Post.WCov[k], Post.hShape, Post.hInvScale[k]
            PhiShape, PhiInvScale, aCov = Post.PhiShape[k], Post.PhiInvScale[k], Post.aCov[k]
            # WMean, WCov, hShape, hInvScale, PhiShape, PhiInvScale, aCov \
            #     = self.calcPostParams4Comp(SS, k)
            e[k] = self.elbo4Comp(SS.getComp(k), WMean, WCov, hShape, hInvScale, PhiShape, PhiInvScale, aCov)
        Gap = np.zeros((SS.K, SS.K))
        for j in xrange(SS.K):
            for k in xrange(j + 1, SS.K):
                WMean, WCov, hShape, hInvScale, PhiShape, PhiInvScale, aCov \
                    = self.calcPostParams4Comp(SS, j, k)
                ejk = self.elbo4Comp(SS.getComp(j) + SS.getComp(k), WMean, WCov,
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
        if not self.calcXxT:
            raise NotImplementedError('Merge move is not allowed for aaT and xaT!')
        Gaps = np.zeros(len(PairList))
        for ii, (kA, kB) in enumerate(PairList):
            Gaps[ii] = self.calcHardMergeGap(SS, kA, kB)
        return Gaps

    def calcPostParams4Comp(self, SS, kA=None, kB=None):
        if not self.calcXxT:
            raise NotImplementedError('Merge move is not allowed for aaT and xaT!')

        C, D = self.C, self.D
        if kB is None:
            thisSS = SS.getComp(kA)
        else:
            thisSS = SS.getComp(kA) + SS.getComp(kB)

        WMean, hShape, hInvScale, PhiShape, PhiInvScale, aCov = self.initPostParams(thisSS)

        for i in xrange(self.nPostUpdate):
            xaT, aaT = self.get_xaT_aaT(thisSS, WMean, PhiShape, PhiInvScale, aCov)
            WMean, WCov = self.calcPostW(xaT, aaT, hShape, hInvScale, PhiShape, PhiInvScale)
            E_WT_W = self.calcE_WT_W(WMean, WCov)
            hShape, hInvScale = self.calcPostH(WMean, WCov)
            PhiShape, PhiInvScale = self.calcPostPhi(thisSS, xaT, aaT, WMean, E_WT_W)
            aCov = self.calcPostACov(E_WT_W, PhiShape, PhiInvScale)

        return WMean, WCov, hShape, hInvScale, PhiShape, PhiInvScale, aCov


    ########################################################### Other
    ###########################################################
    def setPostFactors(self, aCov=None,
                       WMean=None, WCov=None, PhiShape=None, PhiInvScale=None,
                       hShape=None, hInvScale=None, **kwargs):
        K, D, C = WMean.shape
        self.C = C
        self.K = K
        self.Post = ParamBag(K=K, D=D, C=C)
        self.Post.setField('WMean', WMean, dims=('K','D','C'))
        if WCov.ndim == 4:
            self.WCovType = 'full'
            self.Post.setField('WCov', WCov, dims=('K','D','C','C'))
        elif WCov.ndim == 3:
            self.WCovType = 'diag'
            self.Post.setField('WCov', WCov, dims=('K','D','C'))
        self.Post.setField('hShape', float(hShape))
        if hInvScale.ndim == 1:
            hInvScale = hInvScale[:, np.newaxis]
        self.Post.setField('hInvScale', hInvScale, dims=('K','C'))
        if PhiShape.ndim == 2 and PhiShape.shape[0] is 1:
            PhiShape = np.reshape(PhiShape,PhiShape.shape[1])
        self.Post.setField('PhiShape', PhiShape, dims=('K'))
        self.Post.setField('PhiInvScale', PhiInvScale, dims=('K','D'))
        self.Post.setField('aCov', aCov, dims=('K','C','C'))

    def getGaussCov4Comp(self, k=None):
        Post = self.Post
        cov = np.inner(Post.WMean[k], Post.WMean[k]) \
              + np.diag(Post.PhiInvScale[k] / (Post.PhiShape[k] - 1))
        if self.WCovType == 'diag':
            cov += np.sum(Post.WCov[k], axis=1)
        elif self.WCovType == 'full':
            cov += np.sum(np.diagonal(Post.WCov[k], axis1=1, axis2=2), axis=1)
        return cov

    def sampleFromComp(self, k=None, N=1, seed=0):
        PRNG = np.random.RandomState(seed)
        cov = self.getGaussCov4Comp(k=k)
        result = PRNG.multivariate_normal(np.zeros(self.D), cov, N)
        return result


if __name__ == '__main__':
    import bnpy
    import StarCovarK5
    Data = StarCovarK5.get_data(nObsTotal=5000)
    # import DeadLeavesD25
    # Data = DeadLeavesD25.get_data()
    hmodel, RInfo = bnpy.run('DeadLeavesD25', 'DPMixtureModel', 'ZeroMeanFactorAnalyzer', 'moVB',
                             C=2, nLap=50, K=1, WCovType='diag',
                             calcXxT=1, nTask=5, jobname='K1', nPostUpdate=2,
                             moves='birth,merge,delete,shuffle')
    print hmodel.allocModel.get_active_comp_probs()

    # import matplotlib.pylab as plt
    # for k in xrange(hmodel.obsModel.K):
    #     plt.figure(k)
    #     sigma = hmodel.obsModel.getGaussCov4Comp(k)
    #     plt.imshow(sigma, interpolation='nearest', cmap='hot', clim=[-.25, 1])
    # plt.show()

    # import matplotlib.pylab as plt
    # hmodel, RInfo = bnpy.run('/Users/Geng/Documents/Brown/research/patch/HDP_patches/BerkSeg500/Patches_Size8x8_Stride4',
    #                          'DPMixtureModel', 'ZeroMeanFactorAnalyzer', 'moVB',
    #                         C=30, nLap=10, K=10, Kmax=300,
    #                         jobname='FA/C30', datasetName='Half', WCovType='full')#,
                            # moves='birth,merge,delete',
                            # birthPerLap = 1, targetMinSize=1000, targetMaxSize=10000)

    # print hmodel.allocModel.get_active_comp_probs()

    # hmodel, RInfo = bnpy.run(Data,
    #                          'DPMixtureModel', 'ZeroMeanGauss', 'moVB',
    #                         nLap=200, K=1, Kmax=300,
    #                         moves='birth,merge,delete',
    #                         sF=1e-5, nTask=10,
    #                         jobname='Gauss', datasetName='DeadLeavesD25',
    #                         birthPerLap = 2, targetMinSize=100, targetMaxSize=1000)

    # from bnpy.viz.GaussViz import plotCovMatFromHModel
    # plotCovMatFromHModel(hmodel)
    # plt.show()

    from bnpy.viz.PlotTrace import plotJobsThatMatch
    # plotJobsThatMatch('/Users/Geng/Documents/Brown/research/patch/FAPY/StarCovarK5/', yvar='K')
    # plt.show()
    # plt.figure(2)
    # plotJobsThatMatch('/Users/Geng/Documents/Brown/research/patch/FAPY/DeadLeavesD25/*xxT*')
    # plotJobsThatMatch('/Users/Geng/Downloads/hehe/*', loc = 'lower right',
    #                   savefilename = '/Users/Geng/Downloads/hehe/elbo.pdf')

    # hmodel = bnpy.load_model('/Users/Geng/Documents/Brown/research/patch/FAPY/StarCovarK5/defaultjob/1')
    # from bnpy.viz import PlotELBO
    # PlotELBO.plotJobsThatMatchKeywords('StarCovarK5/defaultjob');

    # from bnpy.viz.GaussViz import plotGauss2DContour, Colors
    # import matplotlib.pylab as plt
    # for k in xrange(hmodel.obsModel.K):
        # plt.figure(k)
        # sigma = hmodel.obsModel.getGaussCov4Comp(k)
        # plt.imshow(sigma, interpolation='nearest', cmap='hot', clim=[-.25, 1])
        # plotGauss2DContour(np.zeros(hmodel.obsModel.D), sigma)
    # plt.show()

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
