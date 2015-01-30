'''
HDPHMM.py

Hidden Markov model (HMM) with hierarchical Dirichlet process (HDP) prior.
'''

import numpy as np
import logging
Log = logging.getLogger('bnpy')

import HMMUtil
from bnpy.allocmodel import AllocModel
from bnpy.suffstats import SuffStatBag
from bnpy.util import digamma, gammaln, EPS
from bnpy.util import StickBreakUtil
from bnpy.allocmodel.topics import OptimizerRhoOmega

from bnpy.allocmodel.topics.HDPTopicModel import c_Beta, c_Dir

class HDPHMM(AllocModel):

    def __init__(self, inferType, priorDict = dict()):
        if inferType == 'EM':
            raise ValueError('EM is not supported for HDPHMM')

        self.set_prior(**priorDict)
        self.inferType = inferType
        self.K = 0

        # Variational free parameters
        self.rho = None # rho/omega define Beta distr. on global stick lengths
        self.omega = None
        self.transTheta = None # Kx(K+1) Dirichlet params for transition matrix
        self.initTheta = None # K+1 Dirichlet parameters for the initial state

    def set_prior(self, gamma = 10, alpha = 0.5, hmmKappa = 0.0, **kwargs):
        self.gamma = gamma
        self.alpha = alpha
        self.kappa = hmmKappa

    def get_active_comp_probs(self):
        ''' Return K vector of appearance probabilities for each of the K comps
        '''
        return StickBreakUtil.rho2beta_active(self.rho)

    def get_init_prob_vector(self):
      ''' Get vector of initial probabilities for all K active states
      '''
      expELogPi0 = digamma(self.initTheta) - digamma(np.sum(self.initTheta))
      np.exp(expELogPi0, out = expELogPi0)
      return expELogPi0[0:self.K]


    def get_trans_prob_matrix(self):
      ''' Get matrix of transition probabilities for all K active states
      '''
      digammaSumVec = digamma(np.sum(self.transTheta, axis = 1))
      expELogPi = digamma(self.transTheta) - digammaSumVec[:, np.newaxis]
      np.exp(expELogPi, out = expELogPi)
      return expELogPi[0:self.K, 0:self.K]
      


  ######################################################### Local Params
  #########################################################

    def calc_local_params(self, Data, LP, 
                                MergePrepInfo=None,
                                limitMemoryLP=0,
                                **kwargs):
        ''' Calculate local parameters for each data item and each component.   
        This is part of the E-step.
        
        Args
        -------
        Data : bnpy data object with Data.nObs observations
        LP : local param dict with fields
              E_log_soft_ev : Data.nObs x K array where
                  E_log_soft_ev[n,k] = log p(data obs n | comp k)
        
        Returns
        -------
        LP : A dictionary with updated keys 'resp' and 'respPair' (see the 
             documentation for mathematical definitions of resp and respPair).
             Note that respPair[0,:,:] is undefined.
        '''
        if MergePrepInfo is None or 'mPairIDs' not in MergePrepInfo:
          mPairIDs = np.zeros((0,2))
          M = 0
        else:
          mPairIDs = MergePrepInfo['mPairIDs']
          M = len(mPairIDs)

        # Calculate trans matrix arg for fwd/bwd alg
        digammasumVec = digamma(np.sum(self.transTheta, axis = 1))
        expELogPi = digamma(self.transTheta[:, :self.K]) \
                            - digammasumVec[:, np.newaxis]
        np.exp(expELogPi, out = expELogPi)

        # Calculate start state log prob vector for fwd/bwd alg
        ELogPi0 = digamma(self.initTheta[:self.K]) \
                            - digamma(np.sum(self.initTheta))

        # Unpack soft evidence matrix for fwd/bwd alg
        lpr = LP['E_log_soft_ev']

        if limitMemoryLP:
          # Set init probs are uniform,
          # because we'll update the first step's logSoftEv to include log pi_0
          expELogPi0 = np.ones(self.K)

          logMargPr = np.empty(Data.nDoc)
          resp = np.empty((Data.nObs, self.K))
          Htable = np.empty((Data.nDoc, self.K, self.K))
          TransCount = np.empty((Data.nDoc, self.K, self.K))
          mHtable = np.zeros((2*M, self.K))

          # Run forward backward algorithm on each sequence
          for n in xrange(Data.nDoc):
            start = Data.doc_range[n]
            stop = Data.doc_range[n+1]
            logSoftEv_n = lpr[start:stop]
            logSoftEv_n[0] += ELogPi0 # adding in start state log probs

            resp_n, lp_n, TransCount_n, Htable_n, mHtable_n = \
                      HMMUtil.FwdBwdAlg_LimitMemory(
                                        expELogPi0,
                                        expELogPi, 
                                        logSoftEv_n,
                                        mPairIDs)
            resp[start:stop] = resp_n
            logMargPr[n] = lp_n
            TransCount[n] = TransCount_n
            Htable[n] = Htable_n
            mHtable += mHtable_n

          LP['resp'] = resp
          LP['evidence'] = np.sum(logMargPr)
          LP['TransCount'] = TransCount
          LP['Htable'] = Htable
          LP['mHtable'] = mHtable
        else:
          # Set init probs are uniform,
          # because we'll update the first step's logSoftEv to include log pi_0
          expELogPi0 = np.ones(self.K)

          # Run the forward backward algorithm on each sequence
          logMargPr = np.empty(Data.nDoc)
          resp = np.empty((Data.nObs, self.K))
          respPair = np.empty((Data.nObs, self.K, self.K))
          respPair[0].fill(0)

          for n in xrange(Data.nDoc):
            start = Data.doc_range[n]
            stop = Data.doc_range[n+1]
            logSoftEv_n = lpr[start:stop]
            logSoftEv_n[0] += ELogPi0 # adding in start state log probs

            seqResp, seqRespPair, seqLogMargPr = \
                      HMMUtil.FwdBwdAlg(expELogPi0,
                                        expELogPi, 
                                        logSoftEv_n)
            resp[start:stop] = seqResp
            respPair[start:stop] = seqRespPair
            logMargPr[n] = seqLogMargPr

          LP['evidence'] = np.sum(logMargPr)
          LP['resp'] = resp
          LP['respPair'] = respPair

        return LP

    def initLPFromResp(self, Data, LP):
        ''' Initialize all local params needed for global update from given resp

            Returns
            --------
            LP : same dict as input, with some additional fields
        '''
        K = LP['resp'].shape[1]
        LP['TransCount'] = np.zeros((Data.nDoc, K, K))
        for n in xrange(Data.nDoc):
          start = Data.doc_range[n]
          stop = Data.doc_range[n+1]
          for t in xrange(start+1, stop):
            respPair_t = np.outer(LP['resp'][t-1,:], LP['resp'][t,:])
            LP['TransCount'][n] += respPair_t
        return LP

    def selectSubsetLP(self, Data, LP, relIDs):
        ''' Create local parameter dict for subset of sequences in Data

            Returns
            -------
            subsetLP : local params dict
        '''
        relIDs = np.asarray(relIDs, dtype=np.int32)
        if relIDs.size == Data.nDoc:
          return dict(**LP)
        T_all = np.sum(Data.doc_range[relIDs+1] - Data.doc_range[relIDs]) 
        K = LP['resp'].shape[1]
        resp = np.zeros((T_all, K))
        if 'respPair' in LP:
          respPair = np.zeros((T_all, K, K))
        else:
          TransCount = np.zeros((len(relIDs), K, K))
          Htable = np.zeros((len(relIDs), K, K))
        nstart = 0
        for ii, n in enumerate(relIDs):
          start = Data.doc_range[n]
          stop = Data.doc_range[n+1]
          nstop = nstart + stop - start
          resp[nstart:nstop] = LP['resp'][start:stop]
          if 'respPair' in LP:
            respPair[nstart:nstop] = LP['respPair'][start:stop]
          else:
            TransCount[ii] = LP['TransCount'][n]
            Htable[ii] = LP['Htable'][n]
          nstart = nstop
        if 'respPair' in LP:
          subsetLP = dict(resp=resp, respPair=respPair)
        else:
          subsetLP = dict(resp=resp, TransCount=TransCount, Htable=Htable)
        return subsetLP

  ######################################################### Sufficient Stats
  #########################################################
    def getSummaryFieldNames(self):
      return ['StartStateCount', 'TransStateCount']

    def getSummaryFieldDims(self):
      return [('K'), ('K', 'K')]

    def get_global_suff_stats(self, Data, LP,
                                    doPrecompEntropy=False, 
                                    doPrecompMergeEntropy=False, 
                                    mergePairSelection=None,
                                    mPairIDs=None,
                                    trackDocUsage=0,
                                    **kwargs):

      if mPairIDs is None:
        M = 0
      else:
        M = len(mPairIDs)

      resp = LP['resp']
      K = resp.shape[1]
      startLocIDs = Data.doc_range[:-1]
      StartStateCount = np.sum(resp[startLocIDs], axis = 0)
      N = np.sum(resp, axis = 0)

      if 'TransCount' in LP:
        TransStateCount = np.sum(LP['TransCount'], axis=0)
      else:
        respPair = LP['respPair']
        TransStateCount = np.sum(respPair, axis = 0)
      
      SS = SuffStatBag(K=K, D=Data.dim, M=M)
      SS.setField('StartStateCount', StartStateCount, dims=('K'))
      SS.setField('TransStateCount', TransStateCount, dims=('K','K'))
      SS.setField('N', N, dims=('K'))
      
      if doPrecompEntropy or 'Htable' in LP:
        Hstart = HMMUtil.calc_Hstart(LP['resp'], Data=Data)
        if 'Htable' in LP:
          Htable = np.sum(LP['Htable'], axis=0)
        else:
          Htable = HMMUtil.calc_Htable(LP['respPair'])
        SS.setELBOTerm('Hstart', Hstart, dims=('K'))
        SS.setELBOTerm('Htable', Htable, dims=('K','K'))

      if doPrecompMergeEntropy:
        subHstart, subHtable = HMMUtil.PrecompMergeEntropy_SpecificPairs(
                                      LP, Data, mPairIDs)
        SS.setMergeTerm('Hstart', subHstart, dims=('M'))
        SS.setMergeTerm('Htable', subHtable, dims=('M', 2, 'K'))
        SS.mPairIDs = np.asarray(mPairIDs)

      if trackDocUsage:
        ## Track number of times a topic appears with "signif. mass" in a seq.
        DocUsage = np.zeros(K)
        for n in xrange(Data.nDoc):
          start = Data.doc_range[n]
          stop = Data.doc_range[n+1]
          DocUsage += np.sum(LP['resp'][start:stop], axis=0) > 0.01
        SS.setSelectionTerm('DocUsageCount', DocUsage, dims='K')
      return SS


    def forceSSInBounds(self, SS):
      ''' Force TransStateCount and StartStateCount to be >= 0.  

          This avoids numerical issues in memoized updates 
          where SS "chunks" are added and subtracted incrementally
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
      np.maximum(SS.TransStateCount, 0, out = SS.TransStateCount)
      np.maximum(SS.StartStateCount, 0, out = SS.StartStateCount)
      


  ######################################################### Global Params
  #########################################################
    
    def find_optimum_rhoOmega(self, **kwargs):
        ''' Performs numerical optimization of rho and omega for M-step update.

            Note that the optimizer forces rho to be in [EPS, 1-EPS] for
            the sake of numerical stability
        '''

        # Calculate expected log transition probability
        # using theta vectors for all K states plus initial state
        ELogPi = digamma(self.transTheta) \
                 - digamma(np.sum(self.transTheta, axis=1))[:, np.newaxis]
        sumELogPi = np.sum(ELogPi, axis = 0)
        sumELogPi += digamma(self.initTheta) \
                      - digamma(np.sum(self.initTheta))

        # Select initial rho, omega values for gradient descent
        if self.rho is not None and self.rho.size == self.K:
            initRho = self.rho
        else:
            initRho = None

        if self.omega is not None and self.omega.size == self.K:
            initOmega = self.omega
        else:
            initOmega = None

        # Do the optimization
        try:
            rho, omega, fofu, Info = \
              OptimizerRhoOmega.find_optimum_multiple_tries(
                     sumLogPi = sumELogPi,
                     nDoc = self.K+1, 
                     gamma = self.gamma, 
                     alpha = self.alpha,
                     kappa = self.kappa,
                     initrho = initRho, 
                     initomega = initOmega)
            self.OptimizerInfo = Info
            self.OptimizerInfo['fval'] = fofu

        except ValueError as error:
            if hasattr(self, 'rho') and self.rho.size == self.K:
                Log.error('***** Optim failed. Remain at cur val. '+str(error))
                rho = self.rho
                omega = self.omega
            else:
                Log.error('***** Optim failed. Set to prior. ' + str(error))
                omega = (self.gamma + 1 ) * np.ones(SS.K)
                rho = 1/float(1+self.gamma) * np.ones(SS.K)


        return rho, omega


 

    def update_global_params_EM(self, SS, **kwargs):
        raise ValueError('HDPHMM does not support EM')


    def update_global_params_VB(self, SS, 
                                      mergeCompA=None, mergeCompB=None, 
                                      **kwargs):
        ''' Update global parameters.
        '''
        self.K = SS.K

        # Special update case for merges:
        # Fast, heuristic update for new rho given original value
        if mergeCompA is not None:
          beta = OptimizerRhoOmega.rho2beta_active(self.rho)
          beta[mergeCompA] += beta[mergeCompB]
          beta = np.delete(beta, mergeCompB, axis=0)
          self.rho = OptimizerRhoOmega.beta2rho(beta, SS.K)
          omega = self.omega
          omega[mergeCompA] += omega[mergeCompB]
          self.omega = np.delete(omega, mergeCompB, axis=0)

        # Update theta with recently updated info from suff stats
        self.transTheta, self.initTheta = self._calcTheta(SS)

        # Update rho, omega through numerical optimization
        self.rho, self.omega = self.find_optimum_rhoOmega(**kwargs)

        #Pick hyperparameters alpha, gamma that optimize the ELBO
        #self.alpha, self.gamma = self._find_optimal_alpha_gamma()     

        # Update theta again to reflect the new rho, omega
        self.transTheta, self.initTheta = self._calcTheta(SS)

        

    def update_global_params_soVB(self, SS, rho, **kwargs):
        ''' Updates global parameters when learning with stochastic online VB.
            Note that the rho here is the learning rate parameter, not
            the global stick weight parameter rho
        '''
        self.K = SS.K

        # Update theta (1/2), incorporates recently updated suff stats
        transThetaStar, initThetaStar = self._calcTheta(SS)
        self.transTheta = rho*transThetaStar + (1 - rho)*self.transTheta
        self.initTheta = rho*initThetaStar + (1 - rho)*self.initTheta

        # Update rho/omega
        rhoStar, omegaStar = self.find_optimum_rhoOmega(**kwargs)
        g1 = (1-rho) * (self.rho * self.omega) + rho * (rhoStar*omegaStar)
        g0 = (1-rho) * ((1-self.rho)*self.omega) + rho * ((1-rhoStar)*omegaStar)
        self.rho = g1 / (g1+g0)
        self.omega = g1 + g0

        # TODO: update theta (2/2)?? incorporates recent rho/omega updates
        #transThetaStar, initThetaStar = self._calcTheta(SS)
        #self.transTheta = rho*transThetaStar + (1 - rho)*self.transTheta
        #self.initTheta = rho*initThetaStar + (1 - rho)*self.initTheta


    def _calcTheta(self, SS):
      ''' Update free parameters theta, representing transition distributions.

          Returns
          ---------
          transTheta : 2D array, size K x K+1
          initTheta : 1D array, size K
      '''
      K = SS.K
      if self.rho is None or self.rho.size != K:
        self.rho = OptimizerRhoOmega.create_initrho(K)

      # Calculate E_q[alpha * Beta_l] for l = 1, ..., K+1
      alphaEBeta = self.alpha * StickBreakUtil.rho2beta(self.rho)

      # transTheta_kl = M_kl + E_q[alpha * Beta_l] + kappa * 1_{k==l}
      transTheta = np.zeros((K, K + 1))
      transTheta += alphaEBeta[np.newaxis,:]
      transTheta[:K, :K] += SS.TransStateCount + self.kappa * np.eye(self.K)
 
      # initTheta_k = r_1k + E_q[alpha * Beta_l] (where r_1,>K = 0)
      initTheta = alphaEBeta
      initTheta[:K] += SS.StartStateCount

      return transTheta, initTheta
        

    def init_global_params(self, Data, K=0, **initArgs):
      ''' Initialize rho, omega, and theta to reasonable values.

          This is only called by "from scratch" init routines.
      '''
      self.K = K
      self.rho = OptimizerRhoOmega.create_initrho(K)
      self.omega = (1.0 + self.gamma) * np.ones(K)
      
      # To initialize theta, perform standard update given rho, omega
      # but with "empty" sufficient statistics.
      SS = SuffStatBag(K = self.K, D = Data.dim)
      SS.setField('StartStateCount', np.ones(K), dims = ('K'))
      SS.setField('TransStateCount', np.ones((K,K)), dims = ('K','K'))
      self.transTheta, self.initTheta = self._calcTheta(SS)


    def set_global_params(self, hmodel=None, 
                                rho=None, omega=None, 
                                initTheta=None, transTheta=None,
                                **kwargs):
      ''' Set rho, omega to provided values.
      '''
      if hmodel is not None:
        self.K = hmodel.allocModel.K
        if hasattr(hmodel.allocModel, 'rho'):
          self.rho = hmodel.allocModel.rho
          self.omega = hmodel.allocModel.omega
        else:
          raise AttributeError('Unrecognized hmodel. No field rho.')
        if hasattr(hmodel.allocModel, 'initTheta'):
          self.initTheta = hmodel.allocModel.initTheta
          self.transTheta = hmodel.allocModel.transTheta
        else:
          raise AttributeError('Unrecognized hmodel. No field initTheta.')
      elif rho is not None \
              and omega is not None \
              and initTheta is not None:
        self.rho = rho
        self.omega = omega
        self.initTheta = initTheta
        self.transTheta = transTheta
        self.K = omega.size
        assert self.K == self.initTheta.size - 1
      else:
        self._set_global_params_from_scratch(**kwargs)
      
    def _set_global_params_from_scratch(self, beta=None, 
                                        Data=None, nDoc=None, **kwargs):
      ''' Set rho, omega to values that reproduce provided appearance probs

          Args
          --------
          beta : 1D array, size K
                 beta[k] gives top-level probability for active comp k

      '''
      if nDoc is None:
        nDoc = Data.nDoc
      if nDoc is None:
        raise ValueError('Bad parameters. nDoc not specified.')
      if beta is not None:
        beta = beta / beta.sum()
      if beta is None:
        raise ValueError('Bad parameters. Vector beta not specified.')
      Ktmp = beta.size
      rem = np.minimum(0.05, 1./(Ktmp))
      beta = np.hstack([np.squeeze(beta), rem])
      beta = beta/np.sum(beta)
      self.K = beta.size - 1
      self.rho, self.omega = self._convert_beta2rhoomega(beta)
      assert self.rho.size == self.K
      assert self.omega.size == self.K

    def _convert_beta2rhoomega(self, beta, nDoc=10):
      ''' Find vectors rho, omega that are probable given beta

          Returns
          --------
          rho : 1D array, size K
          omega : 1D array, size K
      '''
      assert abs(np.sum(beta) - 1.0) < 0.001
      rho = OptimizerRhoOmega.beta2rho(beta, self.K)
      omega = (nDoc + self.gamma) * np.ones(rho.size)
      return rho, omega

    ####################################################### Objective
    #######################################################
    def calc_evidence(self, Data, SS, LP, todict = False, **kwargs):
        if SS.hasELBOTerm('Htable'):
            L_ent = SS.getELBOTerm('Htable').sum() \
                    + SS.getELBOTerm('Hstart').sum()
        else:
            L_ent = self.elbo_entropy(Data, LP)

        # For stochastic (soVB), we need to scale up the entropy
        # Only used when --doMemoELBO is set to 0 (not recommended)
        if self.inferType == 'soVB' and SS.hasAmpFactor():
            L_ent *= SS.ampF

        Lalloc = L_alloc_no_slack(self.initTheta, self.transTheta)
        Ltop = L_top(self.rho, self.omega, self.alpha, self.gamma, self.kappa)
        return L_ent + Lalloc + Ltop

    def elbo_entropy(self, Data, LP):
        ''' Calculates entropy of state seq. assignment var. distribution
        '''
        return HMMUtil.calcEntropyFromResp(LP['resp'], LP['respPair'], Data)

    def calcHardMergeGap(self, SS, kA, kB):
      ''' Calculate scalar improvement in ELBO for hard merge of comps kA, kB
          
          Does *not* include any entropy.

          Returns
          ---------
          L : scalar
      ''' 
      m_K = SS.K - 1
      m_SS = SuffStatBag(K=SS.K, D=0)
      m_SS.setField('StartStateCount', SS.StartStateCount.copy(), dims='K')
      m_SS.setField('TransStateCount', SS.TransStateCount.copy(),
                                       dims=('K','K'))
      m_SS.mergeComps(kA, kB)

      ## Create candidate beta vector
      m_beta = StickBreakUtil.rho2beta(self.rho)
      m_beta[kA] += m_beta[kB]
      m_beta = np.delete(m_beta, kB, axis=0)

      ## Create candidate rho and omega vectors
      m_rho = StickBreakUtil.beta2rho(m_beta, m_K)
      m_omega = np.delete(self.omega, kB)

      ## Create candidate initTheta
      m_initTheta = self.alpha * m_beta.copy()
      m_initTheta[:m_K] += m_SS.StartStateCount

      ## Create candidate transTheta
      m_transTheta = self.alpha * np.tile(m_beta, (m_K,1))
      m_transTheta[:, :m_K] += m_SS.TransStateCount

      ## Evaluate objective func. for both candidate and current model
      Ltop = L_top(self.rho, self.omega, self.alpha, self.gamma, self.kappa)
      m_Ltop = L_top(m_rho, m_omega, self.alpha, self.gamma, self.kappa)

      Lalloc = L_alloc_no_slack(self.initTheta, self.transTheta)
      m_Lalloc = L_alloc_no_slack(m_initTheta, m_transTheta)
      return (m_Ltop + m_Lalloc) - (Ltop + Lalloc)

    def calcHardMergeGap_AllPairs(self, SS):
      ''' Calc matrix of improvement in ELBO for all possible pairs of comps
      '''
      Gap = np.zeros((SS.K, SS.K))
      for kB in xrange(1, SS.K):
        for kA in xrange(0, kB):  
          Gap[kA, kB] = self.calcHardMergeGap(SS, kA, kB)
      return Gap

    def calcHardMergeGap_SpecificPairs(self, SS, PairList):
      ''' Calc matrix of improvement in ELBO for all possible pairs of comps
      '''
      Gaps = np.zeros(len(PairList))
      for ii, (kA, kB) in enumerate(PairList):
        Gaps[ii] = self.calcHardMergeGap(SS, kA, kB)
      return Gaps

  ######################################################### IO Utils
  #########################################################   for machines


    def to_dict(self):
        return dict(transTheta = self.transTheta, initTheta = self.initTheta,
                    omega = self.omega, rho = self.rho,
                    gamma = self.gamma, alpha = self.alpha)

    def from_dict(self, myDict):
        self.inferType = myDict['inferType']
        self.K = myDict['K']
        self.transTheta = myDict['transTheta']
        self.initTheta = myDict['initTheta']
        self.omega = myDict['omega']
        self.rho = myDict['rho']

    def get_prior_dict(self):
        return dict(gamma = self.gamma, alpha = self.alpha, K = self.K,
                    kappa = self.kappa)


########################################################### ELBO functions
########################################################### 

def L_top(rho, omega, alpha, gamma, kappa):
  ''' Evaluate the top-level term of the surrogate objective
  '''
  K = rho.size
  eta1 = rho * omega
  eta0 = (1-rho) * omega
  digamma_omega = digamma(omega)
  ElogU = digamma(eta1) - digamma_omega
  Elog1mU = digamma(eta0) - digamma_omega

  diff_cBeta = K * c_Beta(1.0, gamma) - c_Beta(eta1, eta0)

  tAlpha = (K+1) * K * np.log(alpha)

  if kappa > 0:
    coefU = K + 1.0 - eta1
    coef1mU = K * OptimizerRhoOmega.kvec(K) + 1.0 + gamma - eta0

    sumEBeta = np.sum(StickBreakUtil.rho2beta_active(rho))
    tBeta = sumEBeta * (np.log(alpha+kappa) - np.log(kappa))

    tKappa = K * (np.log(kappa) - np.log(alpha + kappa))
  else:
    coefU = (K+1) + 1.0 - eta1
    coef1mU = (K+1) * OptimizerRhoOmega.kvec(K) + gamma - eta0

    tBeta = 0
    tKappa = 0

  diff_logU = np.inner(coefU, ElogU) \
              + np.inner(coef1mU, Elog1mU)
  return tAlpha + tKappa + tBeta + diff_cBeta + diff_logU


def L_alloc_no_slack(initTheta, transTheta):
    diff_cDir = - c_Dir(transTheta) - c_Dir(initTheta)
    return diff_cDir

def L_alloc_with_slack(initTheta, transTheta, initM, transM, alphaEbeta):
    K = transM.shape[0]
    if initM.size == alphaEbeta.size - 1:
      initM = np.hstack([initM,0])
    if transM.shape[-1] == alphaEbeta.size - 1:
      transM = np.hstack([transM, np.zeros((K,1))])

    init_slack = np.inner(initM + alphaEbeta - initTheta,
                          digamma(initTheta) - digamma(initTheta.sum())
                          )

    digammaSum = np.sum(transTheta, axis=1)
    trans_slack = np.sum( (transM + alphaEbeta[np.newaxis,:] - transTheta)
                         * (digamma(transTheta) - digammaSum[:,np.newaxis])
                        )                  
    return L_alloc_no_slack(initTheta, transTheta) + init_slack + trans_slack
