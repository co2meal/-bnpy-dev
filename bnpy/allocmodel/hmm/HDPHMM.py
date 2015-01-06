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

class HDPHMM(AllocModel):

    def __init__(self, inferType, priorDict = dict()):
        if inferType == 'EM':
            raise ValueError('EM is not supported for HDPHMM')

        self.set_prior(**priorDict)
        self.inferType = inferType
        self.K = 0

        #Variational parameters
        self.rho = None #rho/omega are stick-breaking params for global stick
        self.omega = None
        self.transTheta = None #Kx(K+1) Dirichlet params for transition matrix
        self.initTheta = None #K+1 Dirichlet parameters for the initial state

    def set_prior(self, gamma = 5, alpha = 0.1, **kwargs):
        self.gamma = gamma
        self.alpha = alpha

    def get_active_comp_probs(self):
        ''' Return K vector of appearance probabilities for each of the K comps
        '''
        return StickBreakUtil.rho2beta_active(self.rho)

    def get_init_prob_vector(self):
      ''' Get vector of initial probabilities for all K active states
      '''
      expELogPi0 = digamma(self.initTheta) - digamma(np.sum(self.initTheta))
      np.exp(expELogPi0, out = expELogPi0)
      return expELogPi0


    def get_trans_prob_matrix(self):
      ''' Get matrix of transition probabilities for all K active states
      '''
      digammaSumVec = digamma(np.sum(self.transTheta, axis = 1))
      expELogPi = digamma(self.transTheta) - digammaSumVec[:, np.newaxis]
      np.exp(expELogPi, out = expELogPi)
      return expELogPi
      


  ######################################################### Local Params
  #########################################################

    def calc_local_params(self, Data, LP, **kwargs):
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

        lpr = LP['E_log_soft_ev']

        #Calculate arguments to the forward backward algorithm
        digammasumVec = digamma(np.sum(self.transTheta, axis = 1))
        expELogPi = digamma(self.transTheta) - digammasumVec[:, np.newaxis]
        np.exp(expELogPi, out = expELogPi)

        expELogPi0 = digamma(self.initTheta) - digamma(np.sum(self.initTheta))
        np.exp(expELogPi0, out = expELogPi0)

        #Run the forward backward algorithm on each sequence
        logMargPr = np.empty(Data.nDoc)
        resp = np.empty((Data.nObs, self.K))
        respPair = np.empty((Data.nObs, self.K, self.K))
        respPair[0].fill(0)

        for n in xrange(Data.nDoc):
          start = Data.doc_range[n]
          stop = Data.doc_range[n+1]
          logSoftEv_n = lpr[start:stop]
          seqResp, seqRespPair, seqLogMargPr = \
                      HMMUtil.FwdBwdAlg(expELogPi0[0:self.K],
                                        expELogPi[0:self.K, 0:self.K], 
                                        logSoftEv_n)
          resp[start:stop] = seqResp
          respPair[start:stop] = seqRespPair
          logMargPr[n] = seqLogMargPr

        LP['evidence'] = np.sum(logMargPr)
        LP['resp'] = resp
        LP['respPair'] = respPair
        return LP

    def initLPFromResp(self, Data, LP):
        shape = np.shape(LP['resp'])
        self.K = shape[1]
        respPair = np.zeros((shape[0], self.K, self.K))
        for t in xrange(1,shape[0]):
            respPair[t,:,:] = np.outer(LP['resp'][t-1,:], LP['resp'][t,:])
        LP.update({'respPair' : respPair})
        return LP

  ######################################################### Sufficient Stats
  #########################################################

    def get_global_suff_stats(self, Data, LP, doPrecompEntropy=None, **kwargs):

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
        SS.setELBOTerm('Elogqz', entropy, dims = (()))
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
      np.maximum(SS.respPairSums, 0, out = SS.respPairSums)
      np.maximum(SS.firstStateResp, 0, out = SS.firstStateResp)
      


  ######################################################### Global Params
  #########################################################
    
    def find_optimum_rhoOmega(self, **kwargs):
        ''' Performs numerical optimization of rho and omega needed in 
            M-step update of global parameters.

            Note that OptimizerHDPDir forces rho to be in [EPS, 1-EPS] for
            the sake of numerical stability
        '''

        #Argument required by optimizer
        ELogPi = digamma(self.transTheta) \
                 - digamma(np.sum(self.transTheta, axis=1))[:, np.newaxis]
        sumELogPi = np.sum(ELogPi, axis = 0)
        sumELogPi += digamma(self.initTheta) \
                      - digamma(np.sum(self.initTheta))

        #Select initial rho, omega values
        if (self.rho is not None) and (self.omega is not None):
            initRho = self.rho
            initOmega = self.omega
        else:
            initRho = None
            initOmega = None

        #Do the optimization
        try:
            rho, omega, fofu, Info = \
                  OptimizerRhoOmega.find_optimum_multiple_tries(
                                                         sumLogPi = sumELogPi,
                                                         nDoc = self.K+1, 
                                                         gamma = self.gamma, 
                                                         alpha = self.alpha, 
                                                         initrho = initRho, 
                                                         initomega = initOmega)

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


    def update_global_params_VB(self, SS, **kwargs):
        ''' Update variational free parameters to maximize objective.
        '''
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
        rhoNew, omegaNew = self.find_optimum_rhoOmega(SS, **kwargs)
        transThetaNew, initThetaNew = self._calcTheta(SS)

        self.transTheta = rho*transThetaNew + (1 - rho)*self.transTheta
        self.initTheta = rho*initThetaNew + (1 - rho)*self.initTheta
        self.rho = rho*rhoNew + (1 - rho)*self.rho
        self.omega = rho*omegaNew + (1 - rho)*self.omega


    def _calcTheta(self, SS):
      ''' Update free parameters theta, representing transition distributions.

          Returns
          ---------
          transTheta : 2D array, size K x K+1
          initTheta : 1D array, size K
      '''
      #Calculate E_q[alpha * Beta_l] for l = 1, ..., K+1
      alphaEBeta = self.alpha * StickBreakUtil.rho2beta(self.rho)

      #transTheta_kl = M_kl + E_q[alpha * Beta_l] (M_k,>K = 0)
      transTheta = np.zeros((self.K, self.K + 1))
      transTheta += alphaEBeta[np.newaxis,:]
      transTheta[0:self.K, 0:self.K] += SS.respPairSums
 
      #initTheta_k = r_1k + E_q[alpha * Beta_l] (r_1,>K = 0)
      initTheta = alphaEBeta
      initTheta[0:self.K] += SS.firstStateResp

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
      SS.setField('firstStateResp', np.zeros(K), dims = ('K'))
      SS.setField('respPairSums', np.zeros((K,K)), dims = ('K','K'))
      self.transTheta, self.initTheta = self._calcTheta(SS)


    ####################################################### Objective
    #######################################################
    def calc_evidence(self, Data, SS, LP, todict = False, **kwargs):
        if SS.hasELBOTerm('Elogqz'):
            entropy = SS.getELBOTerm('Elogqz')
        else:
            entropy = self.elbo_entropy(Data, LP)

        return entropy + self.elbo_alloc() + self.elbo_allocSlack(SS)



    def elbo_entropy(self, Data, LP):
        ''' Calculates entropy of state seq. assignment var. distribution
        '''
        return HMMUtil.calcEntropyFromResp(LP['resp'], LP['respPair'], Data)


 
    def E_logpU_logqU_plus_cDirAlphaBeta(self, SS):
        ''' Calculate E[ log p(u) - log q(u) ] 

            Includes surrogate bound on E[c_D(alpha beta)] 

            Returns
            ---------
            Elogstuff : real scalar
        '''
        ELogU = digamma(self.rho * self.omega) - digamma(self.omega)
        ELog1mU = digamma((1 - self.rho) * self.omega) - digamma(self.omega)
        

    def elbo_alloc(self):
        ''' Calculates allocation term of the variational objective
        '''
        #K + 1 - k for k = 1, ..., K
        thatOneTerm = [self.K - i for i in xrange(self.K)]

        ELogU = digamma(self.rho * self.omega) - digamma(self.omega)
        ELog1mU = digamma((1 - self.rho) * self.omega) - digamma(self.omega)
        
        #Includes norm. constant for pi0.  Note this is the term that requires
        #  the lower bound ... this is the version of the bound that allows for
        #  any alpha
        normPPi = (self.K+1)*(self.K * np.log(self.alpha) + \
                              np.sum(ELogU + thatOneTerm*ELog1mU))
        
        gamTheta = gammaln(self.transTheta)
        gamSumTheta = gammaln(np.sum(self.transTheta, axis = 1))
        normQPi = np.sum(gamSumTheta - np.sum(gamTheta, axis = 1))
        normQPi0 = gammaln(np.sum(self.initTheta)) - np.sum(gammaln(self.initTheta))

        return normPPi - normQPi - normQPi0


    def elbo_allocSlack(self, SS):
      '''
      Term that will be zero if ELBO is computed after the M-step
      '''
      return 0

      #Calculate E_q[alpha * Beta_l] for l = 1, ..., K+1
      EBeta = np.ones(self.K+1)
      EBeta[1:self.K] = np.cumprod(1 - self.rho[:-1])
      EBeta[0:self.K] *= self.rho
      EBeta[self.K] = 1 - np.sum(EBeta[0:self.K])
      EBeta *= self.alpha

      #Calculate E_q[log pi]
      ElogPi = digamma(self.transTheta) - \
               digamma(np.sum(self.transTheta, axis = 1))[:, np.newaxis]
      ElogPi0 = digamma(self.initTheta) - digamma(np.sum(self.initTheta))

      #sum = (E_q[alpha*beta_l] - transTheta_{kl} + M_{kl}) * E[log pi_{kl}]
      sum = EBeta[np.newaxis,:] - self.transTheta
      sum[0:self.K, 0:self.K] += SS.respPairSums
      sum *= ElogPi

      #sum0 = (E_q[alpha*beta_l] - transTheta_{0l} + N_{1l}) * E[log pi_{0l}]
      sum0 = EBeta - self.initTheta
      sum0[0:self.K] += SS.firstStateResp

      return np.sum(sum) + np.sum(sum0)
      
      
      


      
    
            

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
        return dict(gamma = self.gamma, alpha = self.alpha, K = self.K)


