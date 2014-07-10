import numpy as np

from bnpy.allocmodel import AllocModel
from bnpy.suffstats import SuffStatBag
from bnpy.allocmodel.seq import HMMUtil
from bnpy.util import digamma, gammaln, EPS
from bnpy.allocmodel.admix import OptimizerForHDPStickBreak as OptimHDPSB


class HDPHMM(AllocModel):

    def __init__(self, inferType, priorDict):
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

        self.tau = .1
        self.gamma = .1
        self.alpha = .1
        self.Lambda = .1
        self.prevv = None
        self.prevv0 = None
        self.prevz = None
        self.prevy = None


    def set_prior(self, gamma=1.1, alpha=1.1, Lambda=1, tau=1.1, **kwargs):
        self.gamma = gamma
        self.alpha = alpha
        self.Lambda = Lambda
        self.tau = tau


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

        expELogBeta = np.ones((self.K, self.K))
        expELogPi = np.ones(self.K)

        #Calculate arguments to the forward backward algorithm
        #expELogBeta = [digamma(self.u[1,i,j]) - \
        #               digamma(self.u[1,i,j] + self.u[0,i,j]) + \
        #                   np.sum(digamma(self.u[0,i,0:j-1]) - \
        #                       digamma(self.u[1,i,0:j-1] + self.u[0,i,0:j-1])) \
        #                   for i in xrange(self.K) for j in xrange(self.K)]
        for i in xrange(self.K):
            for j in xrange(self.K):
                theSum = np.sum(digamma(self.u[0,i,0:j]) - \
                                    digamma(self.u[0,i,0:j]+self.u[1,i,0:j]))
                expELogBeta[i,j] = digamma(self.u[1,i,j]) - \
                    digamma(self.u[1,i,j] + self.u[0,i,j]) + theSum

        for i in xrange(self.K):
            theSum = np.sum(digamma(self.b[0,0:i]) - \
                                digamma(self.b[1,0:i] + self.b[0,0:i]))
            expELogPi[i] = digamma(self.b[1,i]) - \
                digamma(self.b[1,i] + self.b[0,i]) + theSum
        #expELogPi = [digamma(self.b[1,i])-digamma(self.b[1,i] + self.b[0,i]) + \
        #                 np.sum(digamma(self.b[0,0:i-1]) - \
        #                            digamma(self.b[1,0:i-1]+self.b[0,0:i-1])) \
        #                 for i in xrange(self.K)]
        #expELogBeta = np.reshape(expELogBeta, (self.K, self.K))

 
        LP.update({'ELogBeta':expELogBeta})
        LP.update({'ELogPi':expELogPi})
        expELogBeta = np.exp(expELogBeta)
        expELogPi = np.exp(expELogPi)

        #Normalize so that FwdBwdAlg() will give accurate logMargPrSeq
        #expELogPi = expELogPi / np.sum(expELogPi)
        #for k in xrange(self.K):
        #    expELogBeta[k, :] = expELogBeta[k, :] / np.sum(expELogBeta[k,:])

        resp, respPair, logMargPr = \
            HMMUtil.FwdBwdAlg(expELogPi, expELogBeta, lpr)

        LP.update({'resp':resp})
        LP.update({'respPair':respPair})
        LP.update({'evidence':logMargPr})

        return LP      


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

        respPairSums = np.sum(respPair[1:,:,:], axis = 0)
        firstStateResp = resp[0,:]
        N = np.sum(resp, axis = 0)
        
        SS = SuffStatBag(K = self.K , D = Data.dim)
        SS.setField('firstStateResp', firstStateResp, dims=('K'))
        SS.setField('respPairSums', respPairSums, dims=('K','K'))
        SS.setField('N', N, dims=('K'))

        return SS


    def update_global_params_EM(self, SS, **kwargs):
        raise ValueError('HDPHMM.py does not support EM')


    def update_global_params_VB(self, SS, **kwargs):

        if self.u is None:
            self.u = np.array([np.ones((self.K, self.K)), 
                               np.ones((self.K, self.K))])
            self.b = np.array([np.ones(self.K), np.ones(self.K)])

        #Set to prior Beta(1, gamma) if this is first iteration
        if (self.omega is None) or (self.rho is None):
            self.omega = (self.gamma + 1) * np.ones(self.K)
            self.rho = (1 / (self.gamma + 1)) * np.ones(self.K)

        #Calculate needed parameters for rho, omega optimization
        elogv = np.zeros((self.K, self.K))
        elog1mv = np.zeros((self.K, self.K))
        for i in xrange(self.K):
            for j in xrange(self.K):
                elogv[i,j] = digamma(self.u[1,i,j]) - \
                    digamma(self.u[1,i,j] + self.u[0,i,j])
                elog1mv[i,j] = digamma(self.u[0,i,j]) - \
                    digamma(self.u[1,i,j] + self.u[0,i,j])
        elogv = np.sum(elogv, axis = 0)
        elog1mv = np.sum(elog1mv, axis = 0)

        elogv += digamma(self.b[1,:]) - digamma(self.b[1,:]+self.b[0,:])
        elog1mv += digamma(self.b[0,:]) - digamma(self.b[1,:]+self.b[0,:])

    
        if (self.rho is not None) and (self.omega is not None):
            initRho = self.rho
            initOmega = self.omega
        else:
            initRho = None
            initOmega = None
        
        self.rho, self.omega, fofu, Info = \
            OptimHDPSB.find_optimum_multiple_tries(sumLogVd = elogv, \
                                                   sumLog1mVd = elog1mv, \
                                                   nDoc = self.K+1, \
                                                   gamma = self.gamma, \
                                                   alpha = self.alpha, \
                                                   initrho = initRho, \
                                                   initomega = initOmega)


        rhoProds = np.array([np.prod(1-self.rho[0:i]) for i in xrange(self.K)])

        self.u = np.array([np.ones((self.K, self.K)), 
                           np.ones((self.K, self.K))])
        self.b = np.array([np.ones(self.K), np.ones(self.K)])

        #Update u
        for i in xrange(self.K):
            for j in xrange(self.K):
                self.u[1,i,j] = self.alpha * self.rho[i] * rhoProds[i] + \
                    SS.respPairSums[i,j]
                self.u[0,i,j] = self.alpha * (1 - self.rho[i]) * rhoProds[i] + \
                    np.sum(SS.respPairSums[i,j+1:self.K])

        #Update b
        for i in xrange(self.K):
            self.b[1,i] = self.tau * self.rho[i] * rhoProds[i] + \
                SS.firstStateResp[i]
            self.b[0,i] = self.tau * (1 - self.rho[i]) * rhoProds[i] + \
                np.sum(SS.firstStateResp[i+1:self.K])

        

    def calc_evidence(self, Data, SS, LP):
        if SS.hasELBOTerm('Elogqz'):
            print 'i dont want this why would you give it to me'
        

        return self.elbo_entropy(LP) + self.elbo_norms() + self.elbo_v0()



    def elbo_entropy(self, LP):
        '''Calculates the entropy H(q(z)) that shows up in E_q[q(z)]
        '''

        #Normalize across rows of respPair_{tij} to get s_{tij}, the parameters
          #for q(z_t | z_{t-1})
        
        s = (LP['respPair'] / 
             (np.sum(LP['respPair'], axis = 2)[:, :, np.newaxis] + EPS))
        z_1 = -np.sum(LP['resp'][0,:] * np.log(LP['resp'][0,:] + EPS))
        restZ = -np.sum(LP['respPair'][1:,:,:] * np.log(s[1:,:,:] + EPS))
        return z_1 + restZ


    def elbo_norms(self):
        '''Calculates the sum of all the normalization constants that show up
           in the ELBO.  Specifically the constants in p(v), q(v), p(y), and q(y)
        '''
        
        #For lack of a better name...
        thatOneTerm = [self.K - i for i in xrange(self.K)]

        normPy = self.K * np.log(self.tau) + \
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
        return dict(u = self.u, y = self.b, omega = self.omega,
                    rho = self.rho)

    def from_dict(self, myDict):
        self.inferType = myDict['inferType']
        self.K = myDict['K']
        self.u = myDict['u']
        self.b = myDict['b']
        self.omega = myDict['omega']
        self.rho = myDict['rho']

    def get_prior_dict(self):
        return dict(gamma = self.gamma, alpha = self.alpha, tau = self.tau,
                    Lambda = self.Lambda)
        
    
        
