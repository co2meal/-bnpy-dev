

import numpy as np

from bnpy.allocmodel import AllocModel
from bnpy.suffstats import SuffStatBag
from bnpy.allocmodel.seq import HMMUtil

class FiniteHMM(AllocModel):


 ######################################################### Constructors
 #########################################################
    
    def __init__(self, inferType, priorDict):
        self.inferType = inferType

        self.K = 4 #Number of states  TODO : get this to not be set here...
        self.initPi = None #Starting distribution
        self.transPi = None #Transition matrix
        self.alpha0 = 1

        print 'heyo'


    #TODO: actually set up priors in config/allocmodel.conf
    def set_prior(self, alpha0):
        self.alpha0 = alpha0


  ######################################################### Local Params
  #########################################################


    def calc_local_params(self, Data, LP, **kwargs):
        '''
        Args
        -------
        Data : bnpy data object

        Returns
        -------
        LP : A dictionary with updated keys 'gamma' and 'psi' (see the 
             documentation for mathematical definitions of gamma and psi).
             Note that psi[0,:,:] is undefined.



        Runs the forward backward algorithm (from HMMUtil) to calculate gamma
        and psi and adds them to the LP dict
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
            #print self.initPi
            #print 'lpr = ', lpr[0:3,:]
            gamma, psi, logMargPrSeq = \
                HMMUtil.FwdBwdAlg(self.initPi, self.transPi, lpr)
            #blah = raw_input()
            LP.update({'gamma':gamma})
            LP.update({'psi':psi})
            LP.update({'evidence':logMargPrSeq})

            #TODO : is this what belongs in resp?
            LP.update({'resp':gamma})

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
            gamma : Data.nObs x K array
            psi   : Data.nObs x K x K array (from the def. of psi, note 
                    psi[0,:,:] is undefined)

        Returns
        -------
        SS : A SuffStatBag with fields
            gamma1 : A vector of length K with entry i being gamma(z_{1k})
            psiSums : A K x K matrix where psiSums[i,j] = 
                      sum_{n=2}^K psi(z_{n-1,j}, z_{nk}) 
            N : A vector of length K with entry k being
                sum_{n=1}^Data.nobs gamma(z_{nk})
            
            The first two of these are used by FiniteHMM.update_global_params,
            and the third is used by ObsModel.update_global_params.

        (see the documentation for information about psi and gamma)
        '''

        if doPrecompEntropy is not None:
            print '*********\n FiniteHMM.get_global_suff_stats() currently ',\
                'doesn\'t support doPrecompEntropy \n ********'

        #This method is called before calc_local_params() during initialization,
            #in which case gamma and psi won't exist
        if ('gamma' not in LP) or ('psi' not in LP):
            gamma = np.ones((Data.nObs, self.K)) / self.K
            psi = np.ones((Data.nObs, self.K, self.K)) / (self.K * self.K)
            LP.update({'gamma':gamma})
            LP.update({'psi':psi})
            
        gamma = LP['gamma']
        psi = LP['psi']
        
        gamma1 = gamma[0,:]
        #print gamma[0:5, :]
        #print np.shape(gamma)
        psiSums = np.sum(psi[1:Data.nObs,:,:], axis = 0)
        N = np.sum(gamma, axis = 0)

        SS = SuffStatBag(K = self.K , D = Data.dim)
        SS.setField('gamma1', gamma1, dims=('K'))
        SS.setField('psiSums', psiSums, dims=('K','K'))
        SS.setField('N', N, dims=('K'))

        return SS


 ######################################################### Global Params
 #########################################################

    def update_global_params_EM(self, SS, **kwargs):
        '''
        Args
        -------
        SS : A SuffStatBag that is expected to have the fields gamma1 and
             psiSums, as described in FiniteHMM.get_global_suff_stats()

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

        self.initPi = SS.gamma1 / SS.gamma1.sum()

        normFactor = np.sum(SS.psiSums, axis = 1)
        for i in xrange(SS.K):
            self.transPi[i,:] = SS.psiSums[i,:] / normFactor[i]


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
        return dict(alpha0 = self.alpha0, K = self.K)


