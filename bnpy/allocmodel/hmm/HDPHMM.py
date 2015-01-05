import numpy as np
import scipy.optimize

from bnpy.allocmodel import AllocModel
from bnpy.suffstats import SuffStatBag
from bnpy.allocmodel.seq import HMMUtil
from bnpy.util import digamma, EPS
from scipy.special import gammaln
#from bnpy.allocmodel.admix import OptimizerForHDPStickBreak as OptimHDPSB
from bnpy.allocmodel.admix import OptimizerHDPDir as OptimHDPDir


import logging
Log = logging.getLogger('bnpy')

class HDPHMM(AllocModel):

    def __init__(self, inferType, priorDict = dict()):
        if inferType == 'EM':
            raise ValueError('EM is not supported for HDPHMM')
        
        self.inferType = inferType
        self.K = 0

        #Beta params for global stick
        self.rho = None
        self.omega = None

        #Dirichlet parameters for the transition matrix
        self.theta = None

        #Dirichlet parameters for the initial state distribution
        self.theta0 = None

        self.set_prior(**priorDict)

        self.estZ = dict()



    def set_prior(self, gamma = 5, alpha = 0.1, **kwargs):
        self.gamma = gamma
        self.alpha = alpha
        self.kappa = 100

    def get_active_comp_probs(self):
        ''' Return K vector of appearance probabilities for each of the K comps
        '''
        return OptimHDPDir._v2beta(self.rho)[:-1]


  ######################################################### Local Params
  #########################################################

    def calc_local_params(self, Data, LP, **kwargs):
        ''' Calculate local parameters for each data item and each component.   
        This is part of the E-step.
        
        Args
        -------
        Data : bnpy data object with Data.nObs observations
        LP : local param dict with fields
              E_log_soft_ev : Data.nObs x K array
                  E_log_soft_ev[n,k] = log p(data obs n | comp k)
        
        Returns
        -------
        LP : A dictionary with updated keys 'resp' and 'respPair' (see the 
             documentation for mathematical definitions of resp and respPair).
             Note that respPair[0,:,:] is undefined.
        '''
        lpr = LP['E_log_soft_ev']


        #Calculate arguments to the forward backward algorithm
        dirSums = digamma(np.sum(self.theta, axis = 1))
        expELogPi = digamma(self.theta) - dirSums[:, np.newaxis]
        np.exp(expELogPi, out = expELogPi)

        expELogPi0 = digamma(self.theta0) - digamma(np.sum(self.theta0))
        np.exp(expELogPi0, out = expELogPi0)

        #Run the forward backward algorithm on each sequence
        resp = None
        respPair = None

        for n in xrange(Data.nSeqs):

            seqResp, seqRespPair, _ = \
                HMMUtil.FwdBwdAlg(expELogPi0[0:self.K],
                                  expELogPi[0:self.K, 0:self.K], 
                                  lpr[Data.seqInds[n]:Data.seqInds[n+1]])

            est = HMMUtil.viterbi(lpr[Data.seqInds[n]:Data.seqInds[n+1]],
                                  expELogPi0[0:self.K],
                                  expELogPi[0:self.K, 0:self.K])
            if resp is None:
                resp = np.vstack(seqResp)
                respPair = seqRespPair
            else:
                resp = np.vstack((resp, seqResp))
                respPair = np.append(respPair, seqRespPair, axis = 0)

            self.estZ.update({'%d'%(Data.seqsUsed[n]) : est})
            
        LP.update({'resp' : resp})
        LP.update({'respPair' : respPair})
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
    
    def find_optimum_rhoOmega(self, SS, **kwargs):
        ''' Performs numerical optimization of rho and omega needed in 
            M-step update of global parameters
        '''
        ELogPi = digamma(self.theta) - \
                 digamma(np.sum(self.theta, axis=1))[:, np.newaxis]
        sumELogPi = np.sum(ELogPi, axis = 0)
        #print np.sum(self.theta, axis=1)

        #Add in the contribution of the initial state
        sumELogPi += (digamma(self.theta0) - digamma(np.sum(self.theta0)))

        if (self.rho is not None) and (self.omega is not None):
            initRho = self.rho
            initOmega = self.omega
        else:
            initRho = None
            initOmega = None
        try:

            rho, omega, fofu, Info = \
                  OptimHDPDir.find_optimum_multiple_tries(sumLogPi = sumELogPi,
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


        #Constrain eps <= rho <= 1 - eps to improve numerical stability of 
        #  forward backward algorithm and future optimization
        eps = 1e-8
        lower = np.ones(self.K) * eps
        upper = np.ones(self.K) * (1 - eps)
        rho = np.min(np.vstack((rho, upper)), axis = 0)
        rho = np.max(np.vstack((rho, lower)), axis = 0)

        return rho, omega


    def _find_optimal_alpha_gamma(self):
      eBeta = np.ones(self.K+1)
      eBeta[0:self.K] = self.rho
      eBeta[1:self.K] *= np.cumprod(1-self.rho[:-1])
      eBeta[self.K] = 1 - np.sum(eBeta[0:self.K])

      sumTheta = np.sum(self.theta, axis = 1)
      elogPi = digamma(self.theta) - digamma(sumTheta[:,np.newaxis])
      #print 'elogpi = ', elogPi
      elogPi = np.append(elogPi, (digamma(self.theta0) -
                                  digamma(np.sum(self.theta0)))[np.newaxis,:], axis = 0)

      alpha = -(self.K**2+self.K) / (np.sum(elogPi * eBeta[np.newaxis,:]))
    

      #Numerically optimize gamma
      digsum = np.sum(digamma((1 - self.rho)*self.omega) - digamma(self.omega))
      fprime = lambda gam: -(-self.K*digamma(gam) + self.K*digamma(1+gam) +
                            digsum)
      fprime = lambda gam: self.fooPrime(gam, digsum)

      f =  lambda gam: (self.foo(gam, digsum))

      gamma, _, _ = scipy.optimize.fmin_l_bfgs_b(func = f,
                                                 x0 = self.gamma,
                                                 fprime = fprime)


      print 'self.alpha = ', self.alpha, ' alpha = ', alpha
      print 'self.gamma = ', self.gamma, ' gamma = ', gamma
      #print 'self.theta = ', self.theta / \
      #  np.sum(self.theta, axis=1)[:,np.newaxis]
      #print 'eBeta = ', eBeta


      #return self.alpha, self.gamma
      return alpha, gamma
    
    def foo(self, gam, digsum):
      if (gam < 0):
        return np.inf
      return -(self.K*gammaln(1 + gam) - self.K*gammaln(gam) +
              (gam - 1)*digsum)

    def fooPrime(self, gam, digsum):
      return -(-self.K*digamma(gam) + self.K*digamma(1+gam) +
                            digsum)
        

    def update_global_params_EM(self, SS, **kwargs):
        raise ValueError('HDPHMM does not support EM')


    def update_global_params_VB(self, SS, **kwargs):

        #Set to prior Beta(1, gamma) if this is first iteration
        if (self.omega is None) or (self.rho is None):
            self.omega = (self.gamma + 1) * np.ones(self.K)
            self.rho = (1 / (self.gamma + 1)) * np.ones(self.K)

        #Update rho and omega through numerical optimization
        self.rho, self.omega = self.find_optimum_rhoOmega(SS, **kwargs)

        #Pick hyperparameters alpha, gamma that optimize the ELBO
        #self.alpha, self.gamma = self._find_optimal_alpha_gamma()     

        self.theta, self.theta0 = self._calcTheta(SS)

        

    def update_global_params_soVB(self, SS, rho, **kwargs):
        ''' Updates global parameters when learning with stochastic online VB.
            Note that the rho here is the learning rate parameter, not
            the global stick weight parameter rho
        '''
        rhoNew, omegaNew = self.find_optimum_rhoOmega(SS, **kwargs)
        theta, theta0 = self._calcTheta(SS)

        self.theta = rho*thetaNew + (1 - rho)*self.theta
        self.theta0 = rho*theta0New + (1 - rho)*self.theta0
        self.rho = rho*rhoNew + (1 - rho)*self.rho
        self.omega = rho*omegaNew + (1 - rho)*self.omega


    def _calcTheta(self, SS):
      #Calculate E_q[alpha * Beta_l] for l = 1, ..., K+1
      EBeta = np.ones(self.K+1)
    
      EBeta[1:self.K] = np.cumprod(1 - self.rho[:-1])
      EBeta[0:self.K] *= self.rho
      EBeta[self.K] = 1 - np.sum(EBeta[0:self.K])
      EBeta *= self.alpha

      #theta_kl = M_kl + E_q[alpha * Beta_l] (M_k,>K = 0)
      theta = np.zeros((self.K, self.K + 1))
      theta += EBeta[np.newaxis,:]
      theta[0:self.K, 0:self.K] += SS.respPairSums
 
      #theta0_k = r_1k + E_q[alpha * Beta_l] (r_1,>K = 0)
      theta0 = EBeta
      theta0[0:self.K] += SS.firstStateResp

      return theta, theta0
        

    def init_global_params(self, Data, K=0, **initArgs):
      self.K = K
      self.omega = (self.gamma + 1.0) * np.ones(self.K)
      self.rho = (1.0 / (self.gamma + 1.0)) * np.ones(self.K)
      
      #Empty suff stat bag that results in setting theta to the prior
      SS = SuffStatBag(K = self.K, D = Data.dim)
      SS.setField('firstStateResp', np.zeros(K), dims = ('K'))
      SS.setField('respPairSums', np.zeros((K,K)), dims = ('K','K'))
      
      self.theta, self.theta0 = self._calcTheta(SS)


    def calc_evidence(self, Data, SS, LP, todict = False, **kwargs):
        if SS.hasELBOTerm('Elogqz'):
            entropy = SS.getELBOTerm('Elogqz')
        else:
            entropy = self.elbo_entropy(Data, LP)

        return entropy + self.elbo_alloc() + \
            self.elbo_allocSlack(SS)



    def elbo_entropy(self, Data, LP):
        '''Calculates the entropy H(q(z)) that shows up in E_q[q(z)]
        '''

        #Normalize across rows of respPair_{tij} to get s_{tij}, the parameters
          #for q(z_t | z_{t-1})
        sigma  = (LP['respPair'] / 
                  (np.sum(LP['respPair'], axis = 2)[:, :, np.newaxis] + EPS))

        z_1 = -np.sum(LP['resp'][Data.seqInds[:-1],:] * \
                      np.log(LP['resp'][Data.seqInds[:-1],:] + EPS))
        restZ = -np.sum(LP['respPair'][1:,:,:] * np.log(sigma[1:,:,:] + EPS))
        return z_1 + restZ

    def elbo_alloc(self):
        '''Calculates the L_alloc term without the L_{alloc-slack} component
        '''

        #K + 1 - k for k = 1, ..., K
        thatOneTerm = [self.K - i for i in xrange(self.K)]

        ELogU = digamma(self.rho * self.omega) - digamma(self.omega)
        ELog1mU = digamma((1 - self.rho) * self.omega) - digamma(self.omega)
        
        
        #Includes norm. constant for pi0.  Note this is the term that requires
        #  the lower bound
        normPPi = (self.K+1)*(self.K * np.log(self.alpha) + \
                              np.sum(ELogU + thatOneTerm*ELog1mU))
        
        gamTheta = gammaln(self.theta)
        gamSumTheta = gammaln(np.sum(self.theta, axis = 1))
        normQPi = np.sum(gamSumTheta - np.sum(gamTheta, axis = 1))
        normQPi0 = gammaln(np.sum(self.theta0)) - np.sum(gammaln(self.theta0))
  
        return normPPi - normQPi - normQPi0


    def elbo_allocSlack(self, SS):
      '''Term that will be zero if ELBO is computed after the M-step
      '''
      return 0

      #Calculate E_q[alpha * Beta_l] for l = 1, ..., K+1
      EBeta = np.ones(self.K+1)
      EBeta[1:self.K] = np.cumprod(1 - self.rho[:-1])
      EBeta[0:self.K] *= self.rho
      EBeta[self.K] = 1 - np.sum(EBeta[0:self.K])
      EBeta *= self.alpha

      #Calculate E_q[log pi]
      ElogPi = digamma(self.theta) - \
               digamma(np.sum(self.theta, axis = 1))[:, np.newaxis]
      ElogPi0 = digamma(self.theta0) - digamma(np.sum(self.theta0))

      #sum = (E_q[alpha*beta_l] - theta_{kl} + M_{kl}) * E[log pi_{kl}]
      sum = EBeta[np.newaxis,:] - self.theta
      sum[0:self.K, 0:self.K] += SS.respPairSums
      sum *= ElogPi

      #sum0 = (E_q[alpha*beta_l] - theta_{0l} + N_{1l}) * E[log pi_{0l}]
      sum0 = EBeta - self.theta0
      sum0[0:self.K] += SS.firstStateResp

      return np.sum(sum) + np.sum(sum0)
      
      
      


      
    
            

  ######################################################### IO Utils
  #########################################################   for machines


    def to_dict(self):
        #convert the self.estZ dictionary to a list
        estz = list()
        for seq in xrange(np.size(self.estZ.keys())):
            if '%d'%(seq) in self.estZ:
                estz.append(self.estZ['%d'%(seq)])



        return dict(theta = self.theta, theta0 = self.theta0,
                    omega = self.omega, rho = self.rho,
                    estZ = estz, gamma = self.gamma, alpha = self.alpha)

    def from_dict(self, myDict):
        self.inferType = myDict['inferType']
        self.K = myDict['K']
        self.theta = myDict['theta']
        self.theta0 = myDict['theta0']
        self.omega = myDict['omega']
        self.rho = myDict['rho']
        self.estZ = myDict['estZ']

    def get_prior_dict(self):
        return dict(gamma = self.gamma, alpha = self.alpha)

        
    
        
