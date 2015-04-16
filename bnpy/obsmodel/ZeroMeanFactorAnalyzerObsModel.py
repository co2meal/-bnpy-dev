import numpy as np
from bnpy.suffstats import ParamBag, SuffStatBag
from bnpy.obsmodel.AbstractObsModel import AbstractObsModel
from numpy.linalg import inv, solve, det
from scipy.special import psi
from bnpy.util import dotATA, dotATB, dotABT


class ZeroMeanFactorAnalyzerObsModel(AbstractObsModel):

    def __init__(self, inferType='VB', gt_local=True, gt_global=False, Data=None, **PriorArgs):
        self.D = Data.dim
        self.E = Data.TrueParams['Lam'].shape[2]
        self.K = Data.TrueParams['K']
        self.inferType = inferType
        self.createPrior(Data, **PriorArgs)
        self.gt_local = gt_local
        self.gt_global = gt_global

    def createPrior(self, Data, Psi = None, Nu = None, MuStar = None, NuStar = None):
        K = self.K
        D = self.D
        E = self.E
        Psi = Data.TrueParams['Psi']
        a = .001
        b = .001
        eta = 1.0
        self.Prior = ParamBag(K=K, D=D, E=E)
        self.Prior.setField('Psi', Psi, dims=('D','D'))
        self.Prior.setField('a', a)
        self.Prior.setField('b', b)
        self.Prior.setField('eta', eta)

    ########################################################### Suff Stats
    ###########################################################
    def calcSummaryStats(self, Data, LP):
        resp = LP['resp']
        K = resp.shape[1]
        D = self.D
        E = self.E
        N = Data.nObs

        X = Data.X
        Y = LP['Y']['Mean']

        SS = SuffStatBag(K=K, D=Data.dim, E=E)

        # Expected count for each k
        #  Usually computed by allocmodel. But just in case...
        if not hasattr(SS, 'N'):
            SS.setField('N', np.sum(resp, axis=0), dims='K')

        # Expected low-dim mean for each k
        EY = np.zeros((K,E))
        for k in xrange(K):
            EY[k] = dotATB(resp[:,k][:,np.newaxis], Y[:,k])
        SS.setField('y', EY, dims=('K','E'))

        # Expected low-dim outer-product for each k
        EYyT = np.zeros((K, E, E))
        for n in xrange(N):
            for k in xrange(K):
                EYyT[k] += resp[n][k] * ( dotATA(Y[n][k][np.newaxis,:]) + LP['Y']['Cov'][k] )
        SS.setField('yyT', EYyT, dims=('K','E','E'))

        # Expected high-low product for each k
        xyT = np.zeros((K, D, E))
        for k in xrange(K):
            xyT[k] = dotATB(resp[:,k][:,np.newaxis] * X, Y[:,k])
        SS.setField('xyT', xyT, dims=('K','D','E'))

        # Expected high-dim mean for each k
        SS.setField('x', dotATB(resp, X), dims=('K','D'))
        return SS


  ########################################################### Local step
  ###########################################################
    def calcLocalParams(self, Data):
        N = Data.nObs
        K = self.K
        D = self.D
        E = self.E
        def calcResp(Data, LP):
            resp = np.zeros((N, K))
            if self.gt_local:
                for n in xrange(N):
                    resp[n][Data.TrueParams['Z'][n]] = 1.0
            else:
                for n in xrange(N):
                    for k in xrange(K):
                        resp[n,k] = .5 * np.dot( LP['Y']['Mean'][n,k],
                                                 solve(LP['Y']['Cov'][k], LP['Y']['Mean'][n,k]) ) \
                                    + .5 * det(LP['Y']['Cov'][k]) #\
                                    #+ psi(self.Prior.eta / K) - psi(self.Prior.eta)
                    resp[n] -= max(resp[n])
                    resp[n] = np.exp(resp[n]) / sum(np.exp(resp[n]))
            return resp
        def calcPostY(Data):
            if self.gt_local:
                return dict(Mean = np.tile(Data.TrueParams['Y'][:, np.newaxis,:], [1,K,1]),
                            Cov = np.zeros((K, E, E)))
            else:

                if not hasattr(self,'Post'):
                    PRNG = np.random.RandomState(0)
                    self.Post = ParamBag(K=K, D=D, E=E)
                    self.Post.setField('lamMean',
                                       Data.TrueParams['Lam']+10*PRNG.randn(K,D,E), dims=('K','D','E'))
                    self.Post.setField('lamCov',
                                       np.zeros((K, D, E, E))+10*PRNG.randn(K,D,E,E), dims=('K','D','E','E'))

                result = dict()
                result['Cov'] = np.zeros((K, E, E))
                for k in xrange(K):
                    result['Cov'][k] = np.eye(E)
                    for d in xrange(D):
                        result['Cov'][k] += 1.0/self.Prior.Psi[d][d] * \
                                            (dotATA(self.Post.lamMean[k][d][np.newaxis,:]) +
                                             self.Post.lamCov[k][d])
                    result['Cov'][k] = inv(result['Cov'][k])
                result['Mean'] = np.zeros((N, K, E))
                for n in xrange(N):
                    for k in xrange(K):
                        result['Mean'][n][k] = np.dot(
                                                dotABT(result['Cov'][k], self.Post.lamMean[k]),
                                                solve(self.Prior.Psi, Data.X[n]))
                return result
        LP = dict()
        LP['Y'] = calcPostY(Data)
        LP['resp'] = calcResp(Data, LP)
        return LP


  ########################################################### Global step
  ###########################################################

    def calcPostLam(self, SS):
        K = self.K
        D = self.D
        E = self.E
        SigmaInvLamLam = np.zeros((K, D, E, E))
        for k in xrange(K):
            for d in xrange(D):
                if hasattr(self, 'Post') and hasattr(self.Post, 'nuShape'):
                    NukMean = self.Post.nuShape / self.Post.nuInvScale[k]
                else:
                    NukMean = self.Prior.a / self.Prior.b * np.ones(E)
                SigmaInvLamLam[k, d] = np.diag(NukMean) + \
                                       1.0/(self.Prior.Psi[d][d]) * SS.yyT[k]
        lamCov = np.zeros((K, D, E, E))
        for k in xrange(K):
            for d in xrange(D):
                lamCov[k, d] = inv(SigmaInvLamLam[k,d])
        LamBar = np.zeros((K, D, E))
        for k in xrange(K):
            for d in xrange(D):
                LamBar[k, d] = np.dot(lamCov[k,d],
                                      1.0/self.Prior.Psi[d][d] *
                                      SS.xyT[k][d])
        lamMean = LamBar
        if not hasattr(self, 'Post') or self.Post.K != SS.K:
            self.Post = ParamBag(K=K, D=D, E=E)
        self.Post.setField('lamMean',lamMean,dims=('K','D','E'))
        self.Post.setField('lamCov',lamCov,dims=('K','D','E','E'))

    def calcPostNu(self):
        nuShape = self.Prior.a + self.D/2.0
        nuInvScale = np.zeros((self.K, self.E))
        for k in xrange(self.K):
            for e in xrange(self.E):
                nuInvScale[k,e] = self.Prior.b
                for d in xrange(self.D):
                    nuInvScale[k,e] += .5 * (self.Post.lamMean[k,d,e]**2
                                             + self.Post.lamCov[k,d,e,e])
        self.Post.setField('nuShape',nuShape)
        self.Post.setField('nuInvScale', nuInvScale,dims=('K','E'))

    def calcGlobalParams(self, SS):
        K = self.K
        D = self.D
        E = self.E
        if self.gt_global and not hasattr(self, 'Post'):
            self.Post = ParamBag(K=K, D=D, E=E)
            self.Post.setField('lamMean', Data.TrueParams['Lam'], dims=('K','D','E'))
            self.Post.setField('lamCov', np.zeros((K, D, E, E)), dims=('K','D','E','E'))
            self.Post.setField('nuShape', np.inf)
            self.Post.setField('nuInvScale', np.inf)
        else:
            for i in xrange(5):
                self.calcPostLam(SS)
                self.calcPostNu()


if __name__ == '__main__':

    import D3E2K2_ZM
    Data = D3E2K2_ZM.get_data()

    obsmod = ZeroMeanFactorAnalyzerObsModel(Data=Data, gt_local=False, gt_global=False)

    for i in xrange(10):
        LP = obsmod.calcLocalParams(Data)
        SS = obsmod.calcSummaryStats(Data, LP)
        obsmod.calcGlobalParams(SS)