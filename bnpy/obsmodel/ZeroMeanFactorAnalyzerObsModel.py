import numpy as np
from bnpy.suffstats import ParamBag, SuffStatBag
from bnpy.obsmodel.AbstractObsModel import AbstractObsModel
from numpy.linalg import inv, solve
from scipy.linalg import cholesky, solve_triangular
from scipy.special import psi, gammaln
from bnpy.util import dotATA
from bnpy.util import LOGTWOPI


class ZeroMeanFactorAnalyzerObsModel(AbstractObsModel):

    def __init__(self, inferType='VB', Data=None, C = None, **PriorArgs):
        self.D = Data.dim
        self.C = Data.TrueParams['a'].shape[1]
        self.C = C
        self.K = 0
        self.inferType = inferType
        self.createPrior(Data, **PriorArgs)
        self.Cache = dict()

    def createPrior(self, Data, Psi = None, f = None, g = None):
        K = self.K
        D = self.D
        C = self.C
        if Psi is None:
            Psi = Data.TrueParams['Psi']
        self.Prior = ParamBag(K=K, D=D, C=C)
        self.Prior.setField('Psi', Psi, dims=('D','D'))
        self.Prior.setField('f', f)
        self.Prior.setField('g', g)

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
        msg = msg.replace('\n', '\n  ')
        return msg

    ########################################################### Suff Stats
    ###########################################################
    # @profile
    def calcSummaryStats(self, Data, SS, LP):
        X = Data.X
        resp = LP['resp']
        aMean = LP['aMean']
        aCov = LP['aCov']

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

        # Expected low-dim mean for each k
        # Ea = np.zeros((K,C))
        # for k in xrange(K):
        #     Ea[k] = dotATB(resp[:,k][:,np.newaxis], aMean[:,k])
        # SS.setField('a', Ea, dims=('K','C'))

        # Expected low-dim outer-product for each k
        EaaT = np.zeros((K, C, C))
        sqrtResp = np.sqrt(resp)
        for k in xrange(K):
            EaaT[k] = dotATA(sqrtResp[:,k][:,np.newaxis] * aMean[:,k]) + \
                      SS.N[k] * aCov[k]
        SS.setField('aaT', EaaT, dims=('K','C','C'))

        # Expected high-low product for each k
        xaT = np.zeros((K, D, C))
        for k in xrange(K):
            xaT[k] = np.dot(X.T, resp[:,k][:,np.newaxis] * aMean[:,k])
        SS.setField('xaT', xaT, dims=('K','D','C'))

        # Expected high-dim mean for each k
        # SS.setField('x', dotATB(resp, X), dims=('K','D'))

        # Expected high-dim outer-product for each k
        diagXxT = np.zeros((K, D))
        for k in xrange(K):
            diagXxT[k] = np.einsum('ij,ij->j', resp[:,k][:,np.newaxis] * X, X)
        SS.setField('diagXxT', diagXxT, dims=('K','D'))

        # aCov is not instance-dependent; pass it to SS then self.Post to calc ELBO
        SS.setField('aCov', aCov, dims=('K','C','C'))
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
            LP['aMean'], LP['aCov'] = self.calcA_FromPost(Data)
            LP['E_log_soft_ev'] = self.calcLogSoftEvMatrix_FromPost(Data, LP)
        return LP
    # @profile
    def calcA_FromPost(self, Data):
        N = Data.nObs
        K = self.Post.K
        D = self.D
        C = self.C
        # calculate aCov
        aCov = np.zeros((K, C, C))
        for k in xrange(K):
            aCov[k] = inv(np.eye(C) + self.GetCached('E_WT_invPsi_W', k))
        assert hasattr(self, 'Post')
        self.Post.setField('aCov', aCov, dims=('K','C','C'))
        # calculate aMean
        aMean = np.zeros((N, K, C))
        for k in xrange(K):
            aCovk_WMeankT_invPsi = np.inner(aCov[k], self.Post.WMean[k]) * 1./np.diag(self.Prior.Psi)
            aMean[:,k] = np.inner(Data.X, aCovk_WMeankT_invPsi)
        return aMean, aCov
    # @profile
    def calcLogSoftEvMatrix_FromPost(self, Data, LP):
        N = Data.nObs
        K = self.Post.K
        L = np.zeros((N, K))
        for k in xrange(K):
            Q = solve_triangular(self.GetCached('cholACov', k), LP['aMean'][:,k].T, lower=True)
            L[:,k] = .5 * np.einsum('ij,ij->j', Q, Q) + .5 * self.GetCached('logdetACov', k)
        return L

  ########################################################### Global step
  ###########################################################
    # @profile
    def updatePost(self, SS):
        self.ClearCache()
        if not hasattr(self, 'Post') or self.Post.K != SS.K:
            self.Post = ParamBag(K=SS.K, D=SS.D, C=SS.C)
        WMean, WCov, hShape, hInvScale, aCov = self.calcPostParams(SS)
        self.Post.setField('WMean', WMean, dims=('K','D','C'))
        self.Post.setField('WCov', WCov, dims=('K','D','C','C'))
        self.Post.setField('hShape', hShape)
        self.Post.setField('hInvScale', hInvScale, dims=('K','C'))
        self.Post.setField('aCov', aCov, dims=('K','C','C'))
        self.K = SS.K

    def calcPostParams(self, SS):
        if hasattr(self.Post,'hShape') and hasattr(self.Post,'hInvScale'):
            hShape = self.Post.hShape
            hInvScale = self.Post.hInvScale
        else:
            hShape = self.Prior.f
            hInvScale = self.Prior.g * np.ones((SS.K, self.C))
        for i in xrange(3):
            WMean, WCov = self.calcPostW(SS, hShape, hInvScale)
            hShape, hInvScale = self.calcPostH(WMean, WCov)
        return WMean, WCov, hShape, hInvScale, SS.aCov

    # @profile
    def calcPostW(self, SS, hShape, hInvScale):
        K = SS.K
        D = self.D
        C = self.C
        SigmaInvWW = np.zeros((K, D, C, C))
        for k in xrange(K):
            SigmaInvWW[k] = np.diag(hShape / hInvScale[k]) + \
                            1./ np.diag(self.Prior.Psi)[:,np.newaxis,np.newaxis] * \
                            np.tile(SS.aaT[k], (D,1,1))
        WCov = inv(SigmaInvWW)
        WBar = np.zeros((K, D, C))
        for k in xrange(K):
            for d in xrange(D):
                WBar[k, d] = np.dot(WCov[k,d],
                                    1./np.diag(self.Prior.Psi)[d] * SS.xaT[k][d])
        WMean = WBar
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

    ########################################################### VB ELBO step
    ###########################################################
    # @profile
    def calcELBO_Memoized(self, SS, afterMStep=False):
        ''' Calculate obsModel's ELBO using sufficient statistics SS and Post.

            Args
            -------
            SS : bnpy SuffStatBag, contains fields for N, a, aaT, xaT
            afterMStep : boolean flag
                    if 1, elbo calculated assuming M-step just completed

            Returns
            -------
            obsELBO : scalar float, = E[ log p(h) + log p(W) + log p(a) + log p(x) - log q(h) - log q(W) - log q(a)]
        '''
        elbo = np.zeros(SS.K)
        Post = self.Post
        Prior = self.Prior
        K = SS.K
        C = SS.C
        D = SS.D
        for k in xrange(K):
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
            elbo[k] += .5 * (self.GetCached('logdetACov', k) + C) * SS.N[k] \
                       - .5 * np.trace(SS.aaT[k])

            # terms related with x
            elbo[k] += - .5 * (D * LOGTWOPI + np.sum(np.log(np.diag(Prior.Psi)))) * SS.N[k] \
                       - .5 * np.dot(1./np.diag(Prior.Psi), SS.diagXxT[k]) \
                       + np.dot(1./np.diag(Prior.Psi),
                                np.einsum('ij, ij->i', SS.xaT[k], Post.WMean[k])) \
                       - .5 * np.einsum('ij,ij', SS.aaT[k],self.GetCached('E_WT_invPsi_W', k))
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
    def _E_WWT(self, k=None):
        if k is None:
            raise NotImplementedError()
        else:
            C = self.C
            D = self.D
            result = np.zeros((D,C,C))
            for d in xrange(D):
                result[d] = dotATA(self.Post.WMean[k][d]) + self.Post.WCov[k][d]
            return result

    def _E_WT_invPsi_W(self, k=None):
        if k is None:
            raise NotImplementedError()
        else:
            return np.sum(1./np.diag(self.Prior.Psi)[:, np.newaxis, np.newaxis] * \
                          self.GetCached('E_WWT', k), axis=0)

    def _cholACov(self, k):
        return cholesky(self.Post.aCov[k], lower=True)

    def _logdetACov(self, k):
        cholACov = self.GetCached('cholACov', k)
        return 2 * np.sum(np.log(np.diag(cholACov)))

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