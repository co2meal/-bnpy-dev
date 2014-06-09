
import numpy as np

from bnpy.allocmodel import AllocModel
from bnpy.suffstats import SuffStatBag
from bnpy.allocmodel.seq import HMMUtil

class FiniteHMM(AllocModel):


 ######################################################### Constructors
 #########################################################
    
    def __init__(self, inferType, priorDict):
        self.inferType = inferType

        self.K = 0 #Number of states
        self.initPi = None #Starting distribution
        self.transPi = None #Transition matrix
        self.initAlpha = 0.0
        self.transAlpha = .1 #TODO : This needs to do something!
        self.paramDims = dict(initPi = ('K'), transPi = ('K', 'K'))



    #TODO: actually set up priors in config/allocmodel.conf
    def set_prior(self, initAlpha):
        self.initAlpha = initAlpha #Dirichlet parameter for initPi
        self.transAlphas = transAlphas #Array of dirichlet parameters for 
                                          #transPi


  ######################################################### Local Params
  #########################################################


    def calc_local_params(self, Data, LP, **kwargs):
        '''
        Args
        -------
        Data : bnpy data object

        Returns
        -------
        LP : A dictionary with updated keys 'resp' and 'respPair' (see the 
             documentation for mathematical definitions of resp and respPair).
             Note that respPair[0,:,:] is undefined.


        Runs the forward backward algorithm (from HMMUtil) to calculate resp
        and respPair and adds them to the LP dict
        '''

        lpr = LP['E_log_soft_ev']

        if self.inferType.count('VB') > 0:
            print 'inferType VB yet not supported for FiniteHMM'
        elif self.inferType == 'EM' > 0:
            
            #Initialize the global params if they already haven't been
            if self.initPi is None:
                self.initPi = np.ones(self.K)
                self.initPi /= self.K
            if self.transPi is None:
                self.transPi = np.ones((self.K, self.K))
                for k in xrange(self.K):
                    self.transPi[k,:] /= self.K

            #TODO : is logMargPrSeq actually the "evidence"?
            resp, respPair, logMargPrSeq = \
                HMMUtil.FwdBwdAlg(self.initPi, self.transPi, lpr)

            LP.update({'resp':resp})
            LP.update({'respPair':respPair})
            LP.update({'evidence':logMargPrSeq})

            #TODO : is this what belongs in resp?
            LP.update({'resp':resp})

            return LP
 

 ######################################################### Suff Stats
 #########################################################
    
    def get_global_suff_stats(self, Data, LP, doPrecompEntropy=None, **kwargs):
        '''
        Creates a SuffStatBag that has parameters needed for global parameter
        updates

        Args
        -------
        Data : bnpy data object
        LP : Dictionary containing the local parameters. Expected to contain:
            resp : Data.nObs x K array
            respPair : Data.nObs x K x K array (from the def. of respPair, note 
                       respPair[0,:,:] is undefined)

        Returns
        -------
        SS : A SuffStatBag with fields
            firstStateResp : A vector of length K with entry i being 
                             resp(z_{1k}) = resp[0,:]
            respPairSums : A K x K matrix where respPairSums[i,j] = 
                           sum_{n=2}^K respPair(z_{n-1,j}, z_{nk})
            N : A vector of length K with entry k being
                sum_{n=1}^Data.nobs resp(z_{nk})
            
            The first two of these are used by FiniteHMM.update_global_params,
            and the third is used by ObsModel.update_global_params.

        (see the documentation for information about resp and respPair)
        '''
        
        if doPrecompEntropy is not None:
            print '*********\n FiniteHMM.get_global_suff_stats() currently ',\
                'doesn\'t support doPrecompEntropy \n ********'

        #This method is called before calc_local_params() during initialization,
            #in which case resp and respPair won't exist
        if ('resp' not in LP) or ('respPair' not in LP):
            self.K = LP['resp'].shape[1]
            resp = np.ones((Data.nObs, self.K)) / self.K
            respPair = np.ones((Data.nObs, self.K, self.K)) / (self.K * self.K)
        else:
            resp = LP['resp']
            respPair = LP['respPair']
        
        firstStateResp = resp[0,:]
        respPairSums = np.sum(respPair[1:Data.nObs,:,:], axis = 0)
        N = np.sum(resp, axis = 0)

        SS = SuffStatBag(K = self.K , D = Data.dim)
        SS.setField('firstStateResp', firstStateResp, dims=('K'))
        SS.setField('respPairSums', respPairSums, dims=('K','K'))
        SS.setField('N', N, dims=('K'))

        return SS


 ######################################################### Global Params
 #########################################################

    def update_global_params_EM(self, SS, **kwargs):
        '''
        Args
        -------
        SS : A SuffStatBag that is expected to have the fields firstStateResp 
             and respPairSums, as described in FiniteHMM.get_global_suff_stats()

        Returns
        -------
        Nothing, this method just updates self.initPi and self.transPi
        '''

        self.K = SS.K

        #TODO : get these to be properly initialized
        # (or is this how it should be?)
        if (self.initPi is None) or (self.transPi is None):
            self.initPi = np.ones(self.K)
            self.transPi = np.ones((self.K, self.K))

        self.initPi = (SS.firstStateResp + self.initAlpha) \
            / (SS.firstStateResp.sum() + self.K * self.initAlpha)

        normFactor = np.sum(SS.respPairSums, axis = 1)
        for i in xrange(SS.K):
            self.transPi[i,:] = SS.respPairSums[i,:] / normFactor[i]


    def set_global_params(self, hmodel=None, K=None, initPi=None, transPi=None,
                          **kwargs):
        if hmodel is not None:
            self.K = hmodel.allocModel.K
            self.initPi = hmodel.allocModel.initPi
            self.transPi = hmodel.allocModel.transPi
        else:
            self.K = K
            self.initPi = initPi
            self.transPi = transPi



    def calc_evidence(self, Data, SS, LP):
        if self.inferType == 'EM':
            return LP['evidence']

    def to_dict(self):
        if self.inferType == 'EM':
            return dict(initPi = self.initPi, transPi = self.transPi)
    def get_prior_dict(self):
        return dict(initAlpha = self.initAlpha, transAlpha = self.transAlpha, \
                        K = self.K)


