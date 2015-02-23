'''
FiniteHMM.py

Hidden Markov model (HMM) with fixed, finite number of hidden states.
'''

import numpy as np

import HMMUtil
from bnpy.allocmodel import AllocModel
from bnpy.suffstats import SuffStatBag
from bnpy.util import digamma, gammaln, as2D

def log_pdf_dirichlet(PiMat, alphavec):
  ''' Return scalar log probability for Dir(PiMat | alphavec)
  '''
  PiMat = as2D(PiMat + 1e-100)
  J, K = PiMat.shape
  if type(alphavec) == float:
    alphavec = alphavec * np.ones(K)
  elif alphavec.ndim == 0:
    alphavec = alphavec * np.ones(K)
  assert alphavec.size == K
  cDir = gammaln(np.sum(alphavec)) - np.sum(gammaln(alphavec))  
  return K * cDir + np.sum(np.dot(np.log(PiMat), alphavec-1.0))


class FiniteHMM(AllocModel):

 ######################################################### Constructors
 #########################################################
    def __init__(self, inferType, priorDict):
        self.inferType = inferType
        self.set_prior(**priorDict)

        # Variational parameters
        self.K = 0 # Number of states
        self.initPi = None # Initial state transition distribution
        self.transPi = None # Transition matrix
        self.initTheta = None
        self.transTheta = None


    def set_prior(self, initAlpha = .1, transAlpha = .1, hmmKappa = 0.0,
                  **kwargs):
        ''' Set hyperparameters that control state transition probs
        '''
        self.initAlpha = initAlpha 
        self.transAlpha = transAlpha
        self.kappa = hmmKappa

    def get_active_comp_probs(self):
      ''' Get vector of appearance probabilities for each active state
      '''
      if self.inferType == 'EM':
        return self.transPi.mean(axis=0)
      else:
        EPiMat = self.transTheta / self.transTheta.sum(axis=1)[:,np.newaxis]
        return EPiMat.mean(axis=0)

    def get_init_prob_vector(self):
      ''' Get vector of initial probabilities for all K active states
      '''
      if self.inferType == 'EM':
        pi0 = self.initPi
      else:
        pi0 = np.exp(digamma(self.initTheta)
                           - digamma(np.sum(self.initTheta)))
      return pi0

    def get_trans_prob_matrix(self):
      ''' Get matrix of transition probabilities for all K active states
      '''
      if self.inferType == 'EM':
        EPiMat = self.transPi
      else:
        digammasumVec = digamma(np.sum(self.transTheta, axis = 1))              
        EPiMat = np.exp(digamma(self.transTheta) 
                        - digammasumVec[:,np.newaxis])

      return EPiMat

  ######################################################### Local Params
  #########################################################
    def calc_local_params(self, Data, LP, **kwargs):
        ''' Local update step

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
        logSoftEv = LP['E_log_soft_ev']
        K = logSoftEv.shape[1]

        # First calculate input params for forward-backward alg,
        # These calculations are different for EM and VB
        if self.inferType.count('VB') > 0:
            # Row-wise subtraction
            digammasumVec = digamma(np.sum(self.transTheta, axis = 1))        
            expELogTrans = np.exp(digamma(self.transTheta) 
                                  - digammasumVec[:,np.newaxis])
            expELogInit = np.exp(digamma(self.initTheta)
                                 - digamma(np.sum(self.initTheta)))
            initParam = expELogInit
            transParam = expELogTrans
        elif self.inferType == 'EM' > 0:
            initParam = self.initPi
            transParam = self.transPi
        else:
            raise ValueError('Unrecognized inferType')

        # Run forward-backward algorithm on each sequence
        logMargPr = np.empty(Data.nDoc)
        resp = np.empty((Data.nObs, K))
        respPair = np.zeros((Data.nObs, K, K))
        for n in xrange(Data.nDoc):
            start = Data.doc_range[n]
            stop = Data.doc_range[n+1]
            logSoftEv_n = logSoftEv[start:stop]
            seqResp, seqRespPair, seqLogMargPr = \
                     HMMUtil.FwdBwdAlg(initParam, transParam, logSoftEv_n)
            
            resp[start:stop] = seqResp
            respPair[start:stop] = seqRespPair
            logMargPr[n] = seqLogMargPr

        LP['resp'] = resp
        LP['respPair'] = respPair
        if self.inferType == 'EM':
          LP['evidence'] = np.sum(logMargPr)
        return LP
 

    def initLPFromResp(self, Data, LP, deleteCompID=None):
        ''' Initial complete local params for this model given responsibilities.
        '''
        resp = LP['resp']
        N, K = resp.shape
        respPair = np.zeros((N, K, K))

        # Loop over each sequence,
        # and define pair-wise responsibilities via an outer-product
        for n in xrange(Data.nDoc):
          start = Data.doc_range[n]
          stop = Data.doc_range[n+1]
          R = resp[start:stop]
          respPair[start+1:stop] = R[:-1, :, np.newaxis] \
                                   * R[1:, np.newaxis, :] 
        LP['respPair'] = respPair
        return LP


 ######################################################### Suff Stats
 #########################################################
    
    def get_global_suff_stats(self, Data, LP, doPrecompEntropy=None, **kwargs):
        ''' Create sufficient stats needed for global param updates

        Args
        -------
        Data : bnpy data object
        LP : Dictionary containing the local parameters. Expected to contain:
            resp : Data.nObs x K array
            respPair : Data.nObs x K x K array (from the def. of respPair, note 
                       respPair[0,:,:] is undefined)

        Returns
        -------
        SS : SuffStatBag with fields
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
        resp = LP['resp']
        respPair = LP['respPair']
        K = resp.shape[1]
        startLocIDs = Data.doc_range[:-1]

        firstStateResp = np.sum(resp[startLocIDs], axis = 0)
        N = np.sum(resp, axis = 0)
        respPairSums = np.sum(respPair, axis = 0)

        SS = SuffStatBag(K=K, D=Data.dim)
        SS.setField('firstStateResp', firstStateResp, dims=('K'))
        SS.setField('respPairSums', respPairSums, dims=('K','K'))
        SS.setField('N', N, dims=('K'))

        if doPrecompEntropy is not None:
            entropy = self.elbo_entropy(Data, LP)
            SS.setELBOTerm('Elogqz', entropy, dims=None)
        return SS

    def forceSSInBounds(self, SS):
      ''' Force SS.respPairSums and firstStateResp to be >= 0.  This avoids
          numerical issues in moVB (where SS "chunks" are added and subtracted)
          such as:
            x = 10
            x += 1e-15
            x -= 10
            x -= 1e-15
          resulting in x < 0.

          Returns
          -------
          Nothing.  SS is updated in-place.
      '''
      np.maximum(SS.N, 0, out=SS.N)
      np.maximum(SS.respPairSums, 0, out=SS.respPairSums)
      np.maximum(SS.firstStateResp, 0, out=SS.firstStateResp)



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
        if self.initAlpha <= 1.0:
          self.initPi = SS.firstStateResp
        else:
          self.initPi = SS.firstStateResp + self.initAlpha - 1.0
        self.initPi /= self.initPi.sum()

        if self.transAlpha <= 1.0:
          self.transPi = SS.respPairSums
        else:
          self.transPi = SS.respPairSums + self.transAlpha - 1.0
        self.transPi /= self.transPi.sum(axis=1)[:,np.newaxis]
        

    def update_global_params_VB(self, SS, **kwargs):
        self.initTheta = self.initAlpha + SS.firstStateResp
        self.transTheta = self.transAlpha + SS.respPairSums + \
                          self.kappa * np.eye(self.K)
        self.K = SS.K

    def update_global_params_soVB(self, SS, rho, **kwargs):
        initNew = self.initAlpha + SS.firstStateResp
        transNew = self.transAlpha + SS.respPairSums + \
                   self.kappa * np.eye(self.K)      
        self.initTheta = rho * initNew + (1 - rho) * self.initTheta
        self.transTheta = rho * transNew + (1 - rho) * self.transTheta
        self.K = SS.K

    def init_global_params(self, Data, K=0, **kwargs):
        ''' Default initialization of global parameters when 

            Not used for local-first initializations, such as
            * contigBlocksLP
            * randexamples
            * kmeansplusplus
        '''
        self.K = K
        if self.inferType == 'EM':
            self.initPi = 1.0 / K * np.ones(K)
            self.transPi = 1.0 / K * np.ones((K,K))
        else:
            self.initTheta = self.initAlpha + np.ones(K)
            self.transTheta = self.transAlpha + np.ones((K,K)) + \
                              self.kappa * np.eye(self.K)
                    

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



    def calc_evidence(self, Data, SS, LP, todict = False, **kwargs):
        if self.inferType == 'EM':
            if self.initAlpha < 1.0:
              logprior_init = 0
            else:
              logprior_init = log_pdf_dirichlet(self.initPi, self.initAlpha)
            if self.transAlpha < 1.0:
              logprior_trans = 0
            else:
              logprior_trans = log_pdf_dirichlet(self.transPi, self.transAlpha)

            return LP['evidence'] + logprior_init + logprior_trans

        elif self.inferType.count('VB') > 0:
            if SS.hasELBOTerm('Elogqz'):
                entropy = SS.getELBOTerm('Elogqz')
            else:
                entropy = self.elbo_entropy(Data, LP)
            # For stochastic (soVB), we need to scale up the entropy
            # Only used when --doMemoELBO is set to 0 (not recommended)
            if SS.hasAmpFactor():
                entropy *= SS.ampF
            return entropy + self.elbo_alloc()
        else:
            emsg = 'Unrecognized inferType: ' + self.inferType
            raise NotImplementedError(emsg)

    def elbo_entropy(self, Data, LP):
        return HMMUtil.calcEntropyFromResp(LP['resp'], LP['respPair'], Data)

    def elbo_alloc(self):
        normPinit = gammaln(self.K * self.initAlpha) \
                    - self.K * gammaln(self.initAlpha)
        
        normQinit = gammaln(np.sum(self.initTheta)) \
                    - np.sum(gammaln(self.initTheta))

        normPtrans = self.K * gammaln(self.K*self.transAlpha + self.kappa) - \
                     self.K*(self.K-1) * gammaln(self.transAlpha) - \
                     self.K * gammaln(self.transAlpha + self.kappa)
        
        normQtrans = np.sum(gammaln(np.sum(self.transTheta, axis=1))) \
                      - np.sum(gammaln(self.transTheta))


        return normPinit + normPtrans - normQinit - normQtrans

        
  ######################################################### IO Utils
  #########################################################   for machines
    def to_dict(self):
        if self.inferType == 'EM':
            return dict(initPi=self.initPi,
                        transPi=self.transPi)
        elif self.inferType.count('VB') > 0:
            return dict(initTheta=self.initTheta, 
                        transTheta=self.transTheta)

    def from_dict(self, myDict):
        self.inferType = myDict['inferType']
        self.K = myDict['K']
        if self.inferType.count('VB') > 0:
            self.initTheta = myDict['initTheta']
            self.transTheta = myDict['transTheta']
        elif self.inferType == 'EM':
            self.initPi = myDict['initPi']
            self.transPi = myDict['transPi']

    def get_prior_dict(self):
        return dict(initAlpha=self.initAlpha,
                    transAlpha=self.transAlpha,
                    kappa=self.kappa,
                    K=self.K)
