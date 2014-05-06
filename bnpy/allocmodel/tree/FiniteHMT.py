import numpy as np

from bnpy.allocmodel import AllocModel
from bnpy.suffstats import SuffStatBag
from bnpy.allocmodel.tree import HMTUtil

class FiniteHMT(AllocModel):

    ######################################################### Constructors
    #########################################################

    def __init__(self, inferType):
        self.inferType = inferType
        self.K = 0
        self.initPi = None
        self.transPi = None
        self.initAlpha = 0.0
        self.maxBranch = 0

    def set_prior(self, initAlpha):
        self.initAlpha = initAlpha

    ######################################################### Local Params
    #########################################################

    def calc_local_params(self, Data, LP, **kwargs):
        lpr = LP['E_log_soft_ev']
        if self.inferType.count('VB') > 0:
            print 'inferType VB yet not supported for FiniteHMT'
        elif self.inferType == 'EM' > 0:
            resp, respPair, logMargPrSeq = HMTUtil.SumProductAlg_QuadTree(self.initPi, self.transPi, lpr)
            LP.update({'resp':resp})
            LP.update({'respPair':respPair})
            LP.update({'evidence':logMargPrSeq})

        return LP

    ######################################################### Suff Stats
    #########################################################

    def get_global_suff_stats( self, Data, SS, LP ):   
        resp = LP['resp']
        respPair = LP['respPair']
        
        FirstStateCount = resp[0,:]
        N = np.sum(resp, axis = 1)
        for b in xrange(self.maxBranch):
            PairCounts = np.sum(respPair[Data.mask[b],:,:], axis = 0)
            SS.setField('PairCounts'+str(b), PairCounts, dims=('K','K'))

        SS = SuffStatBag(K = self.K , D = Data.dim)
        SS.setField('FirstStateCount', FirstStateCount, dims=('K'))
        SS.setField('N', N, dims=('K'))

        return SS

    ######################################################### Global Params
    #########################################################
    def update_global_params_EM( self, SS, **kwargs ):
        self.K = SS.K

        if (self.initPi is None) or (self.transPi is None):
            self.initPi = np.ones(self.K)
            self.transPi = np.ones((self.maxBranch, self.K, self.K))

        self.initPi = (SS.FirstStateCount + self.initAlpha) / (SS.FirstStateCount.sum() + self.K * self.initAlpha)

        for b in xrange(self.maxBranch):
            PairCounts = getattr(SS._Fields, 'PairCounts'+str(b))
            normFactor = np.sum(PairCounts, axis = 1)
            self.transPi[b,:] = psiSums / normFactor[:,np.newaxis]

    def set_global_params(self, hmodel=None, K=None, initPi=None, transPi=None, maxBranch=None,**kwargs):
        if hmodel is not None:
            self.K = hmodel.allocModel.K
            self.initPi = hmodel.allocModel.initPi
            self.transPi = hmodel.allocModel.transPi
            if maxBranch is None:
                self.maxBranch = 4
            else:
                self.maxBranch = maxBranch
        else:
            self.K = K
            self.initPi = initPi
            self.transPi = transPi
            self.maxBranch = maxBranch

    def calc_evidence(self, Data, SS, LP):
        if self.inferType == 'EM':
            return LP['evidence']

    def to_dict(self):
        if self.inferType == 'EM':
            return dict(initPi = self.initPi, transPi = self.transPi)