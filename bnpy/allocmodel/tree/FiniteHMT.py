import numpy as np
import matplotlib.pyplot as plt

from bnpy.allocmodel import AllocModel
from bnpy.suffstats import SuffStatBag
from bnpy.allocmodel.tree import HMTUtil
from bnpy.allocmodel.tree import HMTViterbi

class FiniteHMT(AllocModel):

    ######################################################### Constructors
    #########################################################

    def __init__(self, inferType, priorDict=dict()):
        self.inferType = inferType
        self.K = 0
        self.initPi = None
        self.transPi = None
        self.maxBranch = 4
        self.set_prior(**priorDict)
        #self.image = 0

    def set_prior(self, alpha0=1.0, **kwargs):
        self.alpha0 = alpha0

    ######################################################### Local Params
    #########################################################

    def calc_local_params(self, Data, LP, **kwargs):
        lpr = LP['E_log_soft_ev']
        if self.inferType.count('VB') > 0:
            print 'inferType VB not yet supported for FiniteHMT'
        elif self.inferType == 'EM' > 0:
            encoding = HMTViterbi.ViterbiAlg(self.initPi, self.transPi, lpr)
            #plt.scatter(Data.X[:,0], Data.X[:,1], c=encoding, alpha=.7)
            #plt.show()
            #if self.image % 5 == 0:
            #    plt.savefig('/home/mterzihan/Desktop/denoising/trial/%d.png' % self.image)
            #self.image = self.image+1
            resp, respPair, logMargPrSeq = HMTUtil.SumProductAlg_QuadTree(self.initPi, self.transPi, lpr)
            LP.update({'resp':resp})
            LP.update({'respPair':respPair})
            LP.update({'evidence':logMargPrSeq})

        return LP

    ######################################################### Suff Stats
    #########################################################

    def get_global_suff_stats( self, Data, LP , **kwargs):   
        resp = LP['resp']
        respPair = LP['respPair']
        
        FirstStateCount = resp[0,:]
        N = np.sum(resp, axis = 0)
        SS = SuffStatBag(K = self.K , D = Data.dim)
        for b in xrange(self.maxBranch):
            PairCounts = np.sum(respPair[Data.mask[b],:,:], axis = 0)
            SS.setField('PairCounts'+str(b), PairCounts, dims=('K','K'))
        SS.setField('FirstStateCount', FirstStateCount, dims=('K'))
        SS.setField('N', N, dims=('K'))

        return SS

    ######################################################### Global Params
    #########################################################
    def update_global_params_EM( self, SS, **kwargs ):
        self.K = SS.K

        self.initPi = (SS.FirstStateCount) / (SS.FirstStateCount.sum())
        
        for b in xrange(self.maxBranch):
            PairCounts = getattr(SS._Fields, 'PairCounts'+str(b))
            normFactor = np.sum(PairCounts, axis = 1)
            self.transPi[b,:,:] = PairCounts / normFactor[:,np.newaxis]

    def init_global_params(self, Data, K=0, **kwargs):
        self.K = K
        branchNo = 4
        if self.inferType == 'EM':
            self.initPi = 1.0/K * np.ones(K)
            self.transPi = np.empty((branchNo, K, K))
            for b in xrange(branchNo):
                self.transPi[b,:,:] = np.ones(K)[:,np.newaxis]/K * np.ones((K,K))
        else:
            print 'inferType other than EM are not yet supported for FiniteHMT'

    def set_global_params(self, trueParams=None, hmodel=None, K=None, initPi=None, transPi=None, maxBranch=None,**kwargs):
        if hmodel is not None:
            self.K = hmodel.allocModel.K
            self.initPi = hmodel.allocModel.initPi
            self.transPi = hmodel.allocModel.transPi
            if maxBranch is None:
                self.maxBranch = 4
            else:
                self.maxBranch = maxBranch
        elif trueParams is not None:
            self.initPi = trueParams[initPi]
            self.transPi = trueParams[transPi]
            self.mu = trueParams[mu]
            self.Sigma = trueParams[Sigma]
        else:
            self.K = K
            self.initPi = initPi
            self.transPi = transPi
            self.maxBranch = maxBranch

    def calc_evidence(self, Data, SS, LP):
        if self.inferType == 'EM':
            return LP['evidence']

    def from_dict(self, myDict):
        self.inferType = myDict['inferType']
        self.K = myDict['K']
        if self.inferType == 'VB':
            print 'VB is not supported yet for FiniteHMT'
        elif self.inferType == 'EM':
            self.initPit = myDict['initPi']
            self.transPi = myDict['transPi']

    def to_dict(self):
        if self.inferType == 'EM':
            return dict(initPi = self.initPi, transPi = self.transPi)

    def get_prior_dict(self):
        return dict(alpha0=self.alpha0, K=self.K)