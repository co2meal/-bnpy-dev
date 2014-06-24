
import numpy as np

from bnpy.allocmodel import AllocModel
from bnpy.suffstats import SuffStatBag
from bnpy.allocmodel.seq import HMMUtil
from bnpy.util import digamma, gammaln, EPS

class FiniteHMM(AllocModel):


 ######################################################### Constructors
 #########################################################
    
    def __init__(self, inferType, priorDict):
        self.inferType = inferType

        self.K = 0 #Number of states
        self.initPi = None #Starting distribution
        self.transPi = None #Transition matrix

        #Priors
        self.initAlpha = .1
        self.transAlpha = .1

        #Variational parameters
        self.initTheta = None
        self.transTheta = None


    #TODO: actually set up priors in config/allocmodel.conf
    def set_prior(self, initAlpha = .1, transAlpha = .1, **kwargs):
        self.initAlpha = initAlpha #Dirichlet parameter for initPi
        self.transAlpha = transAlpha #Array of dirichlet parameters for 
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

        #First calculate the parameters that will be fed into the fwd-bkwd 
        #  algorithm, which will differ for EM and VB
        if self.inferType.count('VB') > 0:
            #Calculating exp(E_q[log transPi]) and exp(E_q[log initPi])
            expELogTrans = np.exp(digamma(self.transTheta) - 
                                  digamma(np.sum(self.transTheta, axis = 1)))
            expELogInit = np.exp(digamma(self.initTheta) - 
                                 digamma(np.sum(self.initTheta)))

            #Normalize so that FwdBwdAlg() will give accurate logMargPrSeq
            expELogInit = expELogInit / np.sum(expELogInit)
            for k in xrange(self.K):
                expELogTrans[k, :] = expELogTrans[k, :] \
                    / np.sum(expELogTrans[k,:])

            LP.update({'gamma0':expELogInit, 'gamma':expELogTrans})

            initParam = expELogInit
            transParam = expELogTrans

#            resp, respPair, logMargPrSeq = \
#                HMMUtil.FwdBwdAlg(expELogInit, expELogTrans, lpr)

        elif self.inferType == 'EM' > 0:
            #Initialize the global params if they already haven't been
            if self.initPi is None:
                self.initPi = np.ones(self.K)
                self.initPi /= self.K
            if self.transPi is None:
                self.transPi = np.ones((self.K, self.K))
                for k in xrange(self.K):
                    self.transPi[k,:] /= self.K

            initParam = self.initPi
            transParam = self.transPi

        #Now run the forward backward algorithm on each sequence
        resp = None
        respPair = None
        logMargPr = np.empty(Data.nSeqs)
        for n in xrange(Data.nSeqs):
            seqResp, seqRespPair, seqLogMargPr = \
                HMMUtil.FwdBwdAlg(initParam, transParam, \
                                      lpr[Data.seqInds[n]:Data.seqInds[n+1]])
            
            if resp is None:
                resp = np.vstack(seqResp)
                respPair = seqRespPair
            else:
                resp = np.vstack((resp, seqResp))
                respPair = np.append(respPair, seqRespPair, axis = 0)
            logMargPr[n] = seqLogMargPr
    
        LP.update({'evidence':np.sum(logMargPr)})        
        LP.update({'resp':resp})
        LP.update({'respPair':respPair})
        
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
            self.K = np.shape(LP['resp'])[1]
            resp = np.ones((Data.nObs, self.K)) / self.K
            respPair = np.ones((Data.nObs, self.K, self.K)) / (self.K * self.K)
        else:
            resp = LP['resp']
            respPair = LP['respPair']

        inds = Data.seqInds[:-1]

        respPairSums = np.sum(respPair, axis = 0)
        firstStateResp = np.sum(resp[inds], axis = 0)
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
        
    def update_global_params_VB(self, SS, **kwargs):
        self.initTheta = self.initAlpha + SS.firstStateResp
        self.transTheta = self.transAlpha + SS.respPairSums
        self.K = SS.K
            

    def set_global_params(self, hmodel=None, K=None, initPi=None, transPi=None,
                          **kwargs):
        if hmodel is not None:
            self.K = hmodel.allocModel.K
            if self.inferType == 'EM':
                self.initPi = hmodel.allocModel.initPi
                self.transPi = hmodel.allocModel.transPi
            elif self.inferType == 'VB':
                self.initTheta = hmodel.allocModel.initTheta
                self.transTheta = hmodel.allocModel.transTheta
        else:
            self.K = K
            if self.inferType == 'EM':
                self.initPi = initPi
                self.transPi = transPi
            elif self.inferType == 'VB':
                self.initTheta = initTheta
                self.transTheta = transTheta



    def calc_evidence(self, Data, SS, LP):
        if self.inferType == 'EM':
            return LP['evidence']
        elif self.inferType.count('VB') > 0:
            initHat = np.sum(self.initTheta)
            transHat = np.sum(self.transTheta, axis = 1)

            initDigamma = digamma(self.initTheta) - \
                                 digamma(initHat)
            transDigamma = digamma(self.transTheta) - \
                                  digamma(transHat)
            #return np.random.randn() * 50000
#            return (np.sum((self.initAlpha - self.initTheta + LP['resp'][0,:])
#                           * initDigamma) +
#                    np.sum((self.transAlpha - self.transTheta + 
#                            SS.respPairSums * transDigamma)) -
#                    np.sum(SS.firstStateResp* np.log(SS.firstStateResp + EPS)) -
#                    np.sum(LP['respPair'] * np.log(LP['respPair'] + EPS)) +
#                    gammaln(self.K * self.initAlpha)  -
#                    self.K * gammaln(self.initAlpha) +
#                    self.K * gammaln(self.K * self.transAlpha) -
#                    self.K**2 * gammaln(self.transAlpha) -
#                    np.sum(gammaln(self.initTheta)) +
#                    gammaln(initHat) -
#                    np.sum(gammaln(self.transTheta)) +
#                    np.sum(gammaln(transHat)))
            return self.elbo_pi0(LP) + self.elbo_pi(SS) + self.elbo_z(LP, SS)
        else:
            raise NotImplementedError('Unrecognized inferType '+self.inferType)

    def elbo_pi0(self, LP):
        #The normaliziation constant in front of p(pi_0 | alpha_0) does not 
        #  change through iterations, so it does not need to be calculated to
        #  assess convergence.
        #normP = gammaln(self.K * self.initAlpha) - \
        #    self.K * gammaln(self.initAlpha)
        normP = 0
        normQ = gammaln(np.sum(self.initTheta)) - \
            np.sum(gammaln(self.initTheta))

        theMeat = np.sum((- self.initTheta) * 
                         (digamma(self.initTheta) - 
                          digamma(np.sum(self.initTheta))))

        return normP - normQ + theMeat

    def elbo_pi(self, SS):
#        normP = self.K * gammaln(self.K * self.transAlpha) - \
#            self.K**2 * gammaln(self.transAlpha)
        normP = 0
        normQ = np.sum(gammaln(np.sum(self.transTheta, axis = 1)) - \
                           np.sum(gammaln(self.transTheta), axis = 1))

        theMeat = np.sum((- self.transTheta) *
                         (digamma(self.transTheta) - 
                              digamma(np.sum(self.transTheta, axis = 1))))
        return normP - normQ + theMeat
 
    def elbo_z(self, LP, SS):
        z_1 = np.sum(LP['resp'][0,:] * (digamma(self.initTheta) - 
                                        digamma(np.sum(self.initTheta)) -
                                        np.log(LP['resp'][0,:] + EPS)))
        restZ = -np.sum(LP['respPair'][1:,:,:] * \
              np.log(LP['respPair'][1:,:,:] + EPS)) + \
              np.sum(digamma(self.transTheta)*SS.respPairSums) - \
              np.sum(digamma(np.sum(self.transTheta, axis = 1))*SS.respPairSums)

        return z_1 + restZ
        

                    

    def to_dict(self):
        if self.inferType == 'EM':
            return dict(initPi = self.initPi, transPi = self.transPi)
        elif self.inferType == 'VB':
            return dict(initTheta = self.initTheta, 
                        transTheta = self.transTheta)
    def get_prior_dict(self):
        return dict(initAlpha = self.initAlpha, transAlpha = self.transAlpha, \
                        K = self.K)
