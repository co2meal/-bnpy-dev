import numpy as np
from bnpy.suffstats import ParamBag, SuffStatBag
from bnpy.obsmodel.AbstractObsModel import AbstractObsModel
from numpy.linalg import inv, solve, det
from scipy.special import psi
from bnpy.util import dotATA, dotATB, dotABT


class ZeroMeanFactorAnalyzerObsModel(AbstractObsModel):

    def __init__(self, inferType='VB', Data=None, C = None, **PriorArgs):
        self.D = Data.dim
        self.C = C
        self.K = 0
        self.inferType = inferType
        self.createPrior(Data, **PriorArgs)

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
    def calcSummaryStats(self, Data, SS, LP):
        X = Data.X
        resp = LP['resp']
        aMean = LP['aMean']
        aCov = LP['aCov']

        K = resp.shape[1]
        D = self.D
        C = self.C
        N = Data.nObs

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
        Ea = np.zeros((K,C))
        for k in xrange(K):
            Ea[k] = dotATB(resp[:,k][:,np.newaxis], aMean[:,k])
        SS.setField('a', Ea, dims=('K','C'))

        # Expected low-dim outer-product for each k
        EaaT = np.zeros((K, C, C))
        for n in xrange(N):
            for k in xrange(K):
                EaaT[k] += resp[n][k] * (dotATA(aMean[n][k][np.newaxis,:]) + aCov[k])
        SS.setField('aaT', EaaT, dims=('K','C','C'))

        # Expected high-low product for each k
        xaT = np.zeros((K, D, C))
        for k in xrange(K):
            xaT[k] = dotATB(resp[:,k][:,np.newaxis] * X, aMean[:,k])
        SS.setField('xaT', xaT, dims=('K','D','C'))

        # Expected high-dim mean for each k
        SS.setField('x', dotATB(resp, X), dims=('K','D'))
        return SS


  ########################################################### Local step
  ###########################################################
    def calc_local_params(self, Data, LP=None, **kwargs):
        if LP is None:
            LP = dict()
        if self.inferType == 'EM':
            raise NotImplementedError()
        else:
            LP['aMean'], LP['aCov'] = self.calcA_FromPost(Data)
            LP['E_log_soft_ev'] = self.calcLogSoftEvMatrix_FromPost(Data, LP)
        return LP

    def calcA_FromPost(self, Data):
        N = Data.nObs
        K = self.Post.K
        D = self.D
        C = self.C
        aCov = np.zeros((K, C, C))
        for k in xrange(K):
            aCov[k] = np.eye(C)
            for d in xrange(D):
                aCov[k] += 1.0/self.Prior.Psi[d][d] * \
                           (dotATA(self.Post.WMean[k][d][np.newaxis,:]) +
                           self.Post.WCov[k][d])
            aCov[k] = inv(aCov[k])
        aMean = np.zeros((N, K, C))
        for n in xrange(N):
            for k in xrange(K):
                aMean[n][k] = np.dot(
                              dotABT(aCov[k], self.Post.WMean[k]),
                              solve(self.Prior.Psi, Data.X[n]))
        return aMean, aCov

    def calcLogSoftEvMatrix_FromPost(self, Data, LP):
        N = Data.nObs
        K = self.Post.K
        L = np.zeros((N, K))
        for n in xrange(N):
            for k in xrange(K):
                L[n,k] = .5 * np.dot( LP['aMean'][n,k],
                                      solve(LP['aCov'][k], LP['aMean'][n,k]) ) \
                         + .5 * det(LP['aCov'][k])
        return L

  ########################################################### Global step
  ###########################################################
    def updatePost(self, SS):
        if not hasattr(self, 'Post') or self.Post.K != SS.K:
            self.Post = ParamBag(K=SS.K, D=SS.D, C=SS.C)
        WMean, WCov, hShape, hInvScale = self.calcPostParams(SS)
        self.Post.setField('WMean', WMean, dims=('K','D','C'))
        self.Post.setField('WCov', WCov, dims=('K','D','C','C'))
        self.Post.setField('hShape', hShape)
        self.Post.setField('hInvScale', hInvScale, dims=('K','C'))
        self.K = SS.K

    def calcPostParams(self,SS):
        if hasattr(self.Post,'hshape') and hasattr(self.Post,'hInvScale'):
            hShape = self.Post.hshape
            hInvScale = self.Post.hInvScale
        else:
            hShape = self.Prior.f
            hInvScale = self.Prior.g * np.ones((SS.K, self.C))
        for i in xrange(5):
            WMean, WCov = self.calcPostW(SS, hShape, hInvScale)
            hShape, hInvScale = self.calcPostH(WMean, WCov)
        return WMean, WCov, hShape, hInvScale

    def calcPostW(self, SS, hShape, hInvScale):
        K = SS.K
        D = self.D
        C = self.C
        SigmaInvWW = np.zeros((K, D, C, C))
        for k in xrange(K):
            for d in xrange(D):
                hkMean = hShape / hInvScale[k]
                SigmaInvWW[k, d] = np.diag(hkMean) + \
                                       1.0/(self.Prior.Psi[d][d]) * SS.aaT[k]
        WCov = np.zeros((K, D, C, C))
        for k in xrange(K):
            for d in xrange(D):
                WCov[k, d] = inv(SigmaInvWW[k,d])
        WBar = np.zeros((K, D, C))
        for k in xrange(K):
            for d in xrange(D):
                WBar[k, d] = np.dot(WCov[k,d],
                                      1.0/self.Prior.Psi[d][d] *
                                      SS.xaT[k][d])
        WMean = WBar
        return WMean, WCov

    def calcPostH(self, WMean, WCov):
        K = WMean.shape[0]
        hShape = self.Prior.f + self.D/2.0
        hInvScale = np.zeros((K, self.C))
        for k in xrange(K):
            for c in xrange(self.C):
                hInvScale[k,c] = self.Prior.g
                for d in xrange(self.D):
                    hInvScale[k,c] += .5 * (WMean[k,d,c]**2 + WCov[k,d,c,c])
        return hShape, hInvScale

    ########################################################### VB ELBO step
    ###########################################################
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



if __name__ == '__main__':
    import bnpy
    hmodel, RInfo = bnpy.run('D3C2K2_ZM', 'FiniteMixtureModel',
                             'ZeroMeanFactorAnalyzer', 'VB', nLap=50, K=2)
    # hmodel, RInfo = bnpy.run('AsteriskK8', 'FiniteMixtureModel', 'Gauss', 'VB', nLap=50, K=8)