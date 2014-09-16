import numpy as np

from bnpy.allocmodel import AllocModel
from bnpy.suffstats import SuffStatBag
from bnpy.allocmodel.seq import HMMUtil
from bnpy.util import digamma, gammaln, EPS
from bnpy.allocmodel.admix import OptimizerForHDPStickBreak as OptimHDPSB

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

        #Beta params for transition distribution sticks
        self.u = None
        
        #Beta params for initial distribution stick
        self.b = None

        self.set_prior(**priorDict)

        self.estZ = dict()

    def set_prior(self, gamma = 5, alpha = 0.1, **kwargs):
        self.gamma = gamma
        self.alpha = alpha

    def get_active_comp_probs(self):
        ''' Return K vector of appearance probabilities for each of the K comps
        '''
        return OptimHDPSB._v2beta(self.rho)[:-1]


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

        expELogBeta = np.zeros((self.K, self.K))
        expELogPi = np.zeros(self.K)

        #Calculate arguments to the forward backward algorithm
        digBothU = digamma(self.u[1,:,:] + self.u[0,:,:])
        expELogBeta = digamma(self.u[1,:,:]) - digBothU
        expELogBeta[:, 1:] += \
                 np.cumsum(digamma(self.u[0,:,:-1]) - digBothU[:,:-1], axis = 1)
        np.exp(expELogBeta, out = expELogBeta)
        
        digBothB = digamma(self.b[1,:] + self.b[0,:])
        expELogPi = digamma(self.b[1,:]) - digBothB
        expELogPi[1:] += np.cumsum(digamma(self.b[0,:-1]) - digBothB[:-1])
        np.exp(expELogPi, out = expELogPi)

        #Run the forward backward algorithm on each sequence
        resp = None
        respPair = None
        for n in xrange(Data.nSeqs):

            seqResp, seqRespPair, _ = \
                HMMUtil.FwdBwdAlg(expELogPi, expELogBeta, \
                                      lpr[Data.seqInds[n]:Data.seqInds[n+1]])

            est = HMMUtil.viterbi(lpr[Data.seqInds[n]:Data.seqInds[n+1]],
                                   expELogPi, expELogBeta)
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

        #Calculate needed parameters for rho, omega optimization
        elogv = digamma(self.u[1,:,:]) - \
            digamma(self.u[1,:,:] + self.u[0,:,:])
        elog1mv = digamma(self.u[0,:,:]) - \
            digamma(self.u[1,:,:] + self.u[0,:,:])

        elogv = np.sum(elogv, axis = 0)
        elog1mv = np.sum(elog1mv, axis = 0)

        #Add on contribution from initial-state stick breaking weights
        elogv += digamma(self.b[1,:]) - digamma(self.b[1,:]+self.b[0,:])
        elog1mv += digamma(self.b[0,:]) - digamma(self.b[1,:]+self.b[0,:])


    
        if (self.rho is not None) and (self.omega is not None):
            initRho = self.rho
            initOmega = self.omega
        else:
            initRho = None
            initOmega = None
        try:

            rho, omega, fofu, Info = \
                  OptimHDPSB.find_optimum_multiple_tries(sumLogVd = elogv, \
                                                         sumLog1mVd = elog1mv, \
                                                         nDoc = self.K+1, \
                                                         gamma = self.alpha, \
                                                         alpha = self.gamma, \
                                                         initrho = initRho, \
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
        


    def update_global_params_EM(self, SS, **kwargs):
        raise ValueError('HDPHMM does not support EM')


    def update_global_params_VB(self, SS, **kwargs):

        if (self.u is None) or (self.b is None):
            self.u = np.array([np.ones((self.K, self.K)), 
                               np.ones((self.K, self.K))])
            self.b = np.array([np.ones(self.K), np.ones(self.K)])

        #Set to prior Beta(1, gamma) if this is first iteration
        if (self.omega is None) or (self.rho is None):
            self.omega = (self.gamma + 1) * np.ones(self.K)
            self.rho = (1 / (self.gamma + 1)) * np.ones(self.K)

        #Update rho and omega through numerical optimization
        self.rho, self.omega = self.find_optimum_rhoOmega(SS, **kwargs)

        self.u, self.b = self._calc_u_b(SS)
        

    def update_global_params_soVB(self, SS, rho, **kwargs):
        ''' Updates global parameters when learning with stochastic online VB.
            Note that the rho here is the learning rate parameter, not
            the global stick weight parameter rho
        '''
        rhoNew, omegaNew = self.find_optimum_rhoOmega(SS, **kwargs)
        uNew, bNew = self._calc_u_b(SS)

        self.u = rho*uNew + (1 - rho)*self.u
        self.b = rho*bNew + (1 - rho)*self.b
        self.rho = rho*rhoNew + (1 - rho)*self.rho
        self.omega = rho*omegaNew + (1 - rho)*self.omega


    def _calc_u_b(self, SS):
        rhoProds = np.ones(self.K)
        rhoProds[1:] = np.cumprod(1 - self.rho[:-1])

        u = np.array([np.zeros((self.K, self.K)), 
                           np.zeros((self.K, self.K))])
        b = np.array([np.ones(self.K), np.ones(self.K)])

        #Update u
        for i in xrange(self.K):
            for j in xrange(self.K):
                u[1,i,j] = self.alpha * self.rho[j] * rhoProds[j] + \
                    SS.respPairSums[i,j]
                u[0,i,j] = self.alpha * (1 - self.rho[j]) * rhoProds[j] + \
                    np.sum(SS.respPairSums[i,j+1:self.K])


        #u[1,:,:] = (self.alpha * self.rho * rhoProds) + \
        #           SS.respPairSums
        #u[0,:,:] = (self.alpha * (1-self.rho) * rhoProds)[np.newaxis,:]
        #next line does u[0,i,j] += np.sum(SS.firstStateResp[i+1:self.K])
        #u[0,:,:-1] = np.fliplr(np.cumsum(np.fliplr(SS.respPairSums[:,1:]), 
        #                                  axis = 1))
	#u[0,:,:] += (self.alpha * (1-self.rho) * rhoProds)[np.newaxis,:]


        #Update b
        for i in xrange(self.K):
            b[1,i] = self.alpha * self.rho[i] * rhoProds[i] + \
                SS.firstStateResp[i]
            b[0,i] = self.alpha * (1 - self.rho[i]) * rhoProds[i] + \
                np.sum(SS.firstStateResp[i+1:self.K])

        #b = np.array([np.zeros(self.K), np.zeros(self.K)])
        #b[1,:] = self.alpha * self.rho * rhoProds + SS.firstStateResp
        #b[0,:] = self.alpha * (1 - self.rho) * rhoProds
        #b[0,:-1] += np.cumsum(SS.firstStateResp[1:][::-1])[::-1]

        return u, b
        

    def init_global_params(self, Data, K=0, **initArgs):
        self.K = K
        self.omega = (self.gamma + 1) * np.ones(self.K)
        self.rho = (1 / (self.gamma + 1)) * np.ones(self.K)
      
        #Fake suff stat bag that assigns 1/K "observations" to each starting
        #  state and transition
        SS = SuffStatBag(K = self.K , D = Data.dim)
        SS.setField('firstStateResp', np.ones(K) / K, dims=('K'))
        SS.setField('respPairSums', np.ones((K,K)) / K, dims=('K','K'))

        self.u, self.b = self._calc_u_b(SS)
        
       

    def calc_evidence(self, Data, SS, LP, todict = False, **kwargs):
        if SS.hasELBOTerm('Elogqz'):
            entropy = SS.getELBOTerm('Elogqz')
        else:
            entropy = self.elbo_entropy(Data, LP)
        return entropy + self.elbo_alloc() + self.elbo_v0() + \
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
        '''Calculates the sum of all the normalization constants that show up
           in the ELBO.  Specifically the constants in p(v), q(v), p(y), and q(y)
        '''
        #For lack of a better name...
        thatOneTerm = [self.K - i for i in xrange(self.K)]

        normPy = self.K * np.log(self.alpha) + \
            np.sum(digamma(self.rho * self.omega) - \
                       digamma(self.omega) + thatOneTerm * \
                       (digamma((1-self.rho) * self.omega) - \
                            digamma(self.omega)))
        normQy = np.sum(gammaln(self.b[1,:] + self.b[0,:]) - \
                           gammaln(self.b[1,:]) - gammaln(self.b[0,:]))

        normPv = self.K**2 * np.log(self.alpha) + \
            self.K * np.sum(digamma(self.rho * self.omega) - \
                                digamma(self.omega) + thatOneTerm * \
                                (digamma((1-self.rho) * self.omega) - \
                                     digamma(self.omega)))
        normQv = np.sum(gammaln(self.u[1,:,:] + self.u[0,:,:]) - \
                           gammaln(self.u[1,:,:]) - \
                           gammaln(self.u[0,:,:]))
  
        return normPy + normPv - normQy - normQv


    def elbo_allocSlack(self, SS):
        '''Term that will be zero if ELBO is computed after the M-step
        '''
        return 0

        rhoProds = np.ones(self.K)
        rhoProds[1:] = np.cumprod(1 - self.rho[:-1])

        digBothU = digamma(self.u[1,:,:] + self.u[0,:,:])
        E_v = digamma(self.u[1,:,:]) - digBothU
        E_1mv = digamma(self.u[0,:,:]) - digBothU

        digBothB = digamma(self.b[1,:] + self.b[0,:])
        E_b = digamma(self.b[1,:]) - digBothB
        E_1mb = digamma(self.b[0,:]) - digBothB

        bTerm = 0
        uTerm = 0

        for i in xrange(self.K):
            bTerm += (self.alpha * self.rho[i]*rhoProds[i] - self.b[1,i] +
                      SS.firstStateResp[i]) * E_b[i]
            bTerm += (self.alpha * (1-self.rho[i])*rhoProds[i] - self.b[0,i] + 
                      np.sum(SS.firstStateResp[i+1:self.K])) * E_1mb[i]
            
            for j in xrange(self.K):
                uTerm += (self.alpha * self.rho[j]*rhoProds[j] - self.u[1,i,j] +
                          SS.respPairSums[i,j]) * E_v[i,j]
                uTerm +=(self.alpha*(1-self.rho[j])*rhoProds[j] - self.u[0,i,j]+
                         np.sum(SS.respPairSums[i,j+1:self.K])) * E_1mv[i,j]
        return bTerm + uTerm
    

                                                             
    
    def elbo_v0(self):
        '''Calculates E_q[log p(v_0)] - E_q[log q(v_0)], which shows up in 
           its full form in the ELBO
        '''
        normP = self.K * gammaln(1 + self.gamma) - \
            self.K * gammaln(self.gamma)
        normQ = np.sum(gammaln(self.omega) - gammaln(self.rho * self.omega) - \
                           gammaln((1 - self.rho) * self.omega))

        theMeat = np.sum((self.gamma - (1-self.rho) * self.omega) * \
                             (digamma((1-self.rho)*self.omega) - \
                                  digamma(self.omega)) + \
                             (1 - self.rho * self.omega) * \
                             (digamma(self.omega*self.rho) - digamma(self.omega)))

        return normP - normQ + theMeat
                              
    
            

  ######################################################### IO Utils
  #########################################################   for machines


    def to_dict(self):
        #convert the self.estZ dictionary to a list
        estz = list()
        for seq in xrange(np.size(self.estZ.keys())):
            if '%d'%(seq) in self.estZ:
                estz.append(self.estZ['%d'%(seq)])

        return dict(u = self.u, y = self.b, omega = self.omega,
                    rho = self.rho, estZ = estz)

    def from_dict(self, myDict):
        self.inferType = myDict['inferType']
        self.K = myDict['K']
        self.u = myDict['u']
        self.b = myDict['y']
        self.omega = myDict['omega']
        self.rho = myDict['rho']
        self.estZ = myDict['estZ']

    def get_prior_dict(self):
        return dict(gamma = self.gamma, alpha = self.alpha)

        
    
        
