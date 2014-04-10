'''
HDPRelModel.py
Bayesian nonparametric admixture model with unbounded number of components K

Attributes
-------
K        : # of components
alpha0   : scalar concentration param for global-level stick-breaking params v
gamma    : scalar conc. param for document-level mixture weights pi[d]

Local Parameters (document-specific)
--------
edge_variational : nDistinctEdges x K matrix
                  row i has params for word i's Discrete distr over K topics/communities

Global Parameters (shared across all documents)
--------
U1, U0   : K-length vectors, params for variational distribution over 
           stickbreaking fractions v1, v2, ... vK
            q(v[k]) ~ Beta(U1[k], U0[k])

theta : nNodes x K matrix, <-- Treating theta like a boss over here (i.e global)
        row d has params for node d's distribution pi[d] over the K topics
        q( pi[d] ) ~ Dir( alphaPi[d] )

E_logPi : nDoc x K matrix row d has E[ log pi[d] ] < -- stored inside the LP

'''
import numpy as np

import OptimizerForHDPFullVarModel as OptimHDP
from ..AllocModel import AllocModel

from bnpy.suffstats import SuffStatBag
from ...util import NumericUtil
from ...util import digamma, gammaln
from ...util import EPS, np2flatstr
import logging
Log = logging.getLogger('bnpy')


class HDPRelAssortModel(AllocModel):

  ######################################################### Constructors
  #########################################################
  def __init__(self, inferType, priorDict=None):
        if inferType == "EM":
            raise ValueError('HDPModel cannot do EM. Only VB possible.')
        self.inferType = inferType
        self.K = 0
        self.set_prior(priorDict)
        self.epsilon = 1e-3 # set epsilon term used for assortative mmsb models


  def set_prior(self, PriorParamDict):
        self.alpha0 = PriorParamDict['alpha0']
        self.gamma = PriorParamDict['gamma']
    
  def set_helper_params(self):
      ''' Set dependent attribs of this model, given the primary params U1, U0
          This includes expectations of various stickbreaking quantities
      '''
      E = OptimHDP._calcExpectations(self.U1, self.U0)
      self.Ebeta = E['beta']
      self.Elogv = E['logv']
      self.Elog1mv = E['log1-v']
      self.ElogEps1 = np.log(self.epsilon)
      self.ElogEps0 = np.log(1-self.epsilon)
      # Treating theta like a boss over here (i.e global)
      self.ElogTheta = digamma(self.theta) \
                        - digamma(np.sum(self.theta, axis=1))[:,np.newaxis]

  ######################################################### Accessors
  #########################################################
  def get_keys_for_memoized_local_params(self):
        ''' Return list of string names of the LP fields
            that moVB needs to memoize across visits to a particular batch
        '''
        return ['alphaPi']

  ######################################################### Local Params
  #########################################################
  def calc_local_params(self, Data, LP, 
                          nCoordAscentItersLP=20, 
                          convThrLP=0.01, doOnlySomeDocsLP=False, **kwargs):
    ''' Calculate document-specific quantities (E-step)
          Alternate updates to two terms until convergence
            (1) Approx posterior on topic-token assignment
            (2) Approx posterior on doc-topic probabilities

          Returns
          -------
          LP : local params dict, with fields
              alphaPi : nDoc x K+1 matrix, 
                 row d has params for doc d's Dirichlet over K+1 topics
              E_logPi : nDoc x K+1 matrix,
                 row d has doc d's expected log probability of each topic
              edge_variational : nDistinctEdge x K matrix
                 row i has params for word i's Discrete distr over K topics
              DocTopicCount : nDoc x K matrix

          Returns
          -------
          LP : local params dict, with fields
               edge_variational : nDistinctEdges x K matrix
                                row i has params for edge ij's Discrete distr over K topics

    '''
    # When given no local params LP as input, need to initialize from scratch
    # this forces likelihood to drive the first round of local assignments

    # Create an array for
    LP['edge_variational'] = np.zeros( (Data.nEdgeTotal, self.K) )

    # These terms below are mostly used to create an edge list of relevant terms used for faster updates
    LP['E_logsoftev_NormConst'] = np.zeros(Data.nEdgeTotal)
    LP['E_logsoftev_pi_i'] = np.zeros( (Data.nEdgeTotal, self.K) )
    LP['E_logsoftev_pi_j'] = np.zeros( (Data.nEdgeTotal, self.K) )
    LP['E_pi_i_exp'] = np.zeros( (Data.nEdgeTotal, self.K) )
    LP['E_pi_j_exp'] = np.zeros( (Data.nEdgeTotal, self.K) )
    LP['E_pi_i_exp_sum'] = np.zeros( Data.nEdgeTotal )
    LP['E_pi_j_exp_sum'] = np.zeros( Data.nEdgeTotal )

    for e in xrange(Data.nEdgeTotal):
      row_id = Data.edges[e,0]
      col_id = Data.edges[e,1]

      # These LP terms are used mostly for efficient computation of terms across all edges (i.e ELBO)
      LP['E_logsoftev_pi_i'][e,:] = self.ElogTheta[row_id,:self.K] # store E[log pi_i] for every edge ij
      LP['E_logsoftev_pi_j'][e,:] = self.ElogTheta[col_id,:self.K] # store E[log pi_j] for every edge ij
      LP['E_pi_i_exp'][e,:] = np.exp(LP['E_logsoftev_pi_i'][e,:])
      LP['E_pi_j_exp'][e,:] = np.exp(LP['E_logsoftev_pi_j'][e,:])
      LP['E_pi_i_exp_sum'][e] = np.sum( LP['E_pi_i_exp'][e,:])
      LP['E_pi_j_exp_sum'][e] = np.sum( LP['E_pi_j_exp'][e,:])

      # edge variational or edge responsibilities
      LP['edge_variational'][e,:] = self.ElogTheta[row_id,:self.K] \
                                  + self.ElogTheta[col_id,:self.K] \
                                  + LP['E_logsoftev_EdgeLik'][e,:]

      # we compute the normalization constant here in O(K)
      LP['E_logsoftev_NormConst'][e] = np.log( np.exp( np.log(LP['E_pi_i_exp_sum'][e]) + np.log(LP['E_pi_j_exp_sum'][e]) + LP['E_logsoftev_EdgeEps'][e] )  \
                                     + np.sum( np.exp( self.ElogTheta[row_id,:self.K] + self.ElogTheta[col_id,:self.K] )  \
                                     * ( np.exp(LP['E_logsoftev_EdgeLik'][e,:]) - np.exp(LP['E_logsoftev_EdgeEps'][e]) ) ) )

      # normalize responsibilities
      LP['edge_variational'][e,:] -= LP['E_logsoftev_NormConst'][e] # normalize

    # Turn these terms into regular probabilities
    LP['edge_variational'] = np.exp(LP['edge_variational'])

    # Used later for computations of global stick-breaking weights
    LP['E_logPiSumD'] = np.sum(self.ElogTheta, axis=1)
    LP['E_logPiSumK'] = np.sum(self.ElogTheta, axis=0)
    return LP

  def calc_ElogPi(self, LP):
        ''' Update expected log topic probability distr. for each document d
        '''
        alph = LP['alphaPi']
        LP['E_logPi'] = digamma(alph) - digamma(alph.sum(axis=1))[:,np.newaxis]
        return LP

  ######################################################### Suff Stats
  #########################################################
  def get_global_suff_stats(self, Data, LP, doPrecompEntropy=False, 
                                              doPrecompMergeEntropy=False,
                                              mPairIDs=None):
        ''' Theta is a global parameter here so we need to get its sufficient stats
          Sufficient statistics for these require precomputing certain terms
          node_ss_lik: nNodes x K matrix of likelihood sufficient statistics from responsibilities ev
          node_ss_eps: nNodes x K matrix of epsilon likelihood terms needed to update theta (is there a better way?)
        '''
        E, K = LP['E_logsoftev_EdgeLik'].shape
        N = Data.nNodeTotal
        SS = SuffStatBag(K=K, D=N)

        # Summary statistics
        node_ss_lik = np.zeros((N, K))
        node_ss_eps = np.zeros((N, K))
        ev = LP['edge_variational']

        for e in xrange(E):
          ii = Data.edges[e,0]
          jj = Data.edges[e,1]
          node_ss_lik[ii,:] += ev[e,:]
          node_ss_lik[jj,:] += ev[e,:]
          node_ss_eps[ii,:] += np.exp( self.ElogTheta[ii,:self.K] + LP['E_logsoftev_EdgeEps'][e] \
                           + np.log(LP['E_pi_j_exp_sum'][e] - LP['E_pi_j_exp'][e,:]) \
                           - LP['E_logsoftev_NormConst'][e] )# need to check this if there's a better way
          node_ss_eps[jj,:] += np.exp( self.ElogTheta[jj,:self.K] + LP['E_logsoftev_EdgeEps'][e] \
                           + np.log(LP['E_pi_i_exp_sum'][e] - LP['E_pi_i_exp'][e,:]) \
                           - LP['E_logsoftev_NormConst'][e] )

        SS.setField('nNodeTotal', N, dims=None)
        SS.setField('nEdgeTotal', E, dims=None)
        SS.setField('nDoc', N, dims=None)
        SS.setField('node_ss_lik', node_ss_lik, dims=('D','K'))
        SS.setField('node_ss_eps', node_ss_eps, dims=('D','K'))
        # This used in updating global stick breaking weights
        SS.setField('sumLogPiActive', LP['E_logPiSumK'][:self.K], dims='K')
        SS.setField('sumLogPiUnused', LP['E_logPiSumK'][-1], dims=None)

        return SS


  ######################################################### Global Params
  #########################################################

  def update_global_params_VB(self, SS, **kwargs):
        ''' Update global parameters that control topic probabilities
            v[k] ~ Beta( U1[k], U0[k])

            Also update theta, which we consider a global parameter in the HDP relational model
            theta ~ Dirichlet(alpha
        '''
        self.K = SS.K
        '''
        Commented out to show that the ELBO calculations are much smoother if u is not updated
        Turning this on results in big swings of the ELBO.

        u = self._estimate_u(SS)
        self.U1 = u[:self.K]
        self.U0 = u[self.K:]
        '''
        self.theta = self._estimate_theta(SS)
        self.set_helper_params()
        
  def update_global_params_soVB(self, SS, rho, **kwargs):
        '''
        TODO: Need to modify in the future
        '''
        assert self.K == SS.K
        u = self._estimate_u(SS)
        self.U1 = rho * u[:self.K] + (1-rho) * self.U1
        self.U0 = rho * u[self.K:] + (1-rho) * self.U0
        self.set_helper_params()

  def _estimate_theta(self, SS, **kwargs):
        ''' grabs update for theta, requires SS
        '''
        theta = np.zeros((SS.nNodeTotal, SS.K+1))
        theta[:,:self.K] = SS.node_ss_lik + SS.node_ss_eps
        theta += (self.gamma * self.Ebeta)[np.newaxis,:]
        return theta

  def _estimate_u(self, SS, **kwargs):
        ''' Calculate best 2*K-vector u via L-BFGS gradient descent
              performing multiple tries in case of numerical issues
        '''
        if hasattr(self, 'U1') and self.U1.size == self.K:
          initU = np.hstack([self.U1, self.U0])
        else:
          # Use the prior
          initU = np.hstack([np.ones(self.K), self.alpha0*np.ones(self.K)])
        sumLogPi = np.hstack([SS.sumLogPiActive, SS.sumLogPiUnused])

        try:
          u, fofu, Info = OptimHDP.estimate_u_multiple_tries(sumLogPi=sumLogPi,
                                        nDoc=SS.nDoc,
                                        gamma=self.gamma, alpha0=self.alpha0,
                                        initU=initU)
        except ValueError as error:
          if str(error).count('FAILURE') == 0:
            raise error
          if hasattr(self, 'U1') and self.U1.size == self.K:
            Log.error('***** Optim failed. Stay put. ' + str(error))
            return # EXIT with current state, failed to update
          else:
            Log.error('***** Optim failed. Stuck at prior. ' + str(error))
            u = initU # fall back on the prior otherwise
        return u

  def set_global_params(self, hmodel=None, 
                                U1=None, U0=None, 
                                K=0, beta=None, topic_prior=None,
                                Ebeta=None, EbetaLeftover=None, theta=None, **kwargs):
        if hmodel is not None:
          self.K = hmodel.allocModel.K
          self.U1 = hmodel.allocModel.U1
          self.U0 = hmodel.allocModel.U0
          self.set_helper_params()
          return

        if U1 is not None and U0 is not None:
          self.U1 = U1
          self.U0 = U0
          self.K = U1.size
          self.set_helper_params()
          return

        if Ebeta is not None and EbetaLeftover is not None:
          Ebeta = np.squeeze(Ebeta)
          EbetaLeftover = np.squeeze(EbetaLeftover)
          beta = np.hstack( [Ebeta, EbetaLeftover])
          self.K = beta.size - 1
          
        elif beta is not None:
          assert beta.size == K
          beta = np.hstack([beta, np.min(beta)/100.])
          beta = beta/np.sum(beta)
          self.K = beta.size - 1
        else:
          raise ValueError('Bad parameters. Vector beta not specified.')

        # Now, use the specified value of beta to find the best U1, U0
        assert beta.size == self.K + 1
        assert abs(np.sum(beta) - 1.0) < 0.001
        vMean = OptimHDP.beta2v(beta)
        # for each k=1,2...K
        #  find the multiplier vMass[k] such that both are true
        #  1) vMass[k] * vMean[k] > 1.0
        #  2) vMass[k] * (1-vMean[k]) > self.alpha0
        vMass = np.maximum( 1./vMean , self.alpha0/(1.-vMean))
        self.U1 = vMass * vMean
        self.U0 = vMass * (1-vMean)
        assert np.all( self.U1 >= 1.0 - 0.00001)
        assert np.all( self.U0 >= self.alpha0 - 0.00001)
        assert self.U1.size == self.K
        assert self.U0.size == self.K

        ####################################### Set Global Params for Theta
        if theta is not None and beta is not None:
          self.theta = theta
        else:
          self.theta = np.ones( (self.nNodeTotal, self.K + 1 ) )

        self.set_helper_params()



  ######################################################### Evidence
  #########################################################  
  def calc_evidence( self, Data, SS, LP ):
        ''' Calculate ELBO terms related to allocation model
        '''

        # Removed these terms
        E_logpV = self.E_logpV() # shouldn't have to modify
        E_logqV = self.E_logqV() # shouldn't have to modify

        E_logpPi = self.E_logpPi(SS) # should be fine
        E_logqPi = self.E_logqPi(LP) # should be fine
        E_logpZ =  self.E_logpZ(Data, LP) # need to modify
        E_logqZ =  self.E_logqZ(Data, LP)

        elbo_alloc = E_logpPi - E_logqPi \
                   + E_logpZ - E_logqZ
                   #+ E_logpV - E_logqV \
        return elbo_alloc

  ####################################################### ELBO terms for Z
  def E_logpZ( self, Data, LP):
        ''' for r_ijkk only  (where k != l)
         for r_ijkl, this term actually cancels out with a portion of the entropy term
        '''
        E_logpZ = np.sum(LP['edge_variational'] * (LP['E_logsoftev_pi_i'] + LP['E_logsoftev_pi_j']))
        return E_logpZ

  def E_logqZ( self, Data, LP):
        '''
        entropy calculations, we need to only calculate portions of the entropy due to
        the cancellation of terms for the epsilon likelihoods with E_logpZ
        '''
        E_logqZlik = np.sum( LP['edge_variational'] * np.log(LP['edge_variational']))
        E_logqZeps = np.sum( (LP['E_logsoftev_EdgeEps'] - LP['E_logsoftev_NormConst']) \
               * np.exp( (LP['E_logsoftev_EdgeEps'] - LP['E_logsoftev_NormConst']) ) \
               * LP['E_pi_i_exp_sum'] * LP['E_pi_j_exp_sum'] )
        E_logqZ = E_logqZlik + E_logqZeps

        return E_logqZ

  def E_logqZ_memo_terms_for_merge(self, Data, LP, mPairIDs=None):
        ''' Returns KxK matrix 
        ''' 
        wv = LP['word_variational']
        wv += EPS # Make sure all entries > 0 before taking log
        if mPairIDs is None:
          ElogqZMat = NumericUtil.calcRlogRdotv_allpairs(wv, Data.word_count)
        else:
          ElogqZMat = NumericUtil.calcRlogRdotv_specificpairs(wv, 
                                                Data.word_count, mPairIDs)
        return ElogqZMat

  ####################################################### ELBO terms for Pi
  def E_logpPi(self, SS):
    ''' Returns scalar value of E[ log p(PI | alpha0)]
    '''
    '''
    K = SS.K
    kvec = K + 1 - np.arange(1, K+1)
    # logDirNormC : scalar norm const that applies to each iid draw pi_d
    logDirNormC = gammaln(self.gamma) + (K+1) * np.log(self.gamma)
    logDirNormC += np.sum(self.Elogv) + np.inner(kvec, self.Elog1mv)
    # logDirPDF : scalar sum over all doc's pi_d
    sumLogPi = np.hstack([SS.sumLogPiActive, SS.sumLogPiUnused])
    logDirPDF = np.inner(self.gamma * self.Ebeta - 1., sumLogPi)
    '''

    # rewrote the code here to make it a little easier for me
    sticks = self.gamma * self.Ebeta
    logNormPi = gammaln( np.sum(sticks) ) - np.sum(gammaln(sticks))
    logDirPDFp = (sticks - 1) * np.sum(self.ElogTheta, axis=0)
    E_logpPi = (SS.nNodeTotal * logNormPi) + np.sum( logDirPDFp )

    #return (SS.nNodeTotal * logDirNormC) + logDirPDF
    return E_logpPi

  def E_logqPi(self, LP):
    ''' Returns scalar value of E[ log q(PI)],
          calculated directly from local param dict LP
    '''
    logDirNormC = gammaln( np.sum(self.theta,axis=1) ) - np.sum(gammaln(self.theta), axis=1)
    logDirPDFq = np.sum((self.theta - 1) * self.ElogTheta)
    E_logqPi = np.sum(logDirNormC) + logDirPDFq
    return E_logqPi

  def E_logqPi_Memoized_from_LP(self, LP):
    ''' Returns three variables 
                logDirNormC (scalar),
                logqPiActive (length K)
                logqPiUnused (scalar)
                whose sum is equal to E[log q(PI)]
            when added to other results of this function from different batches,
                the sum is equal to E[log q(PI)] of the entire dataset
    '''
    alph = LP['alphaPi']
    logDirNormC = np.sum(gammaln(alph.sum(axis=1)))
    piEntropyVec = np.sum((alph - 1.) * LP['E_logPi'], axis=0) \
                     - np.sum(gammaln(alph),axis=0)
    return logDirNormC, piEntropyVec[:-1], piEntropyVec[-1]


  ####################################################### ELBO terms for V
  def E_logpV(self):
    logBetaNormC = gammaln(self.alpha0 + 1.) \
                      - gammaln(self.alpha0)
    logBetaPDF = (self.alpha0-1.) * np.sum(self.Elog1mv)
    return self.K*logBetaNormC + logBetaPDF

  def E_logqV(self):
    logBetaNormC = gammaln(self.U1 + self.U0) \
                       - gammaln(self.U0) - gammaln(self.U1)
    logBetaPDF = np.inner(self.U1 - 1., self.Elogv) \
                     + np.inner(self.U0 - 1., self.Elog1mv)
    return np.sum(logBetaNormC) + logBetaPDF

  ####################################################### ELBO terms merge
  def memo_elbo_terms_for_merge(self, LP):
    ''' Calculate some ELBO terms for merge proposals for current batch

        Returns
        --------
        ElogpZMat   : KxK matrix
        sumLogPiMat : KxK matrix
        ElogqPiMat  : KxK matrix
    '''
    CMat = LP["DocTopicCount"]
    alph = LP["alphaPi"]
    digammaPerDocSum = digamma(alph.sum(axis=1))[:, np.newaxis]
    alph = alph[:, :-1] # ignore last column ("remainder" topic)

    ElogpZMat = np.zeros((self.K, self.K))
    sumLogPiMat = np.zeros((self.K, self.K))
    ElogqPiMat = np.zeros((self.K, self.K))
    for jj in range(self.K):
      M = self.K - jj - 1
      # nDoc x M matrix, alpha_{dm} for each merge pair m with comp jj
      mergeAlph = alph[:,jj][:,np.newaxis] + alph[:, jj+1:]
      # nDoc x M matrix, E[log pi_m] for each merge pair m with comp jj
      mergeElogPi = digamma(mergeAlph) - digammaPerDocSum
      assert mergeElogPi.shape[1] == M
      # nDoc x M matrix, count for merged topic m each doc
      mergeCMat = CMat[:, jj][:,np.newaxis] + CMat[:, jj+1:]
      ElogpZMat[jj, jj+1:] = np.sum(mergeCMat * mergeElogPi, axis=0)
          
      sumLogPiMat[jj, jj+1:] = np.sum(mergeElogPi,axis=0)
      curElogqPiMat = np.sum((mergeAlph-1.)*mergeElogPi, axis=0) \
                                      - np.sum(gammaln(mergeAlph),axis=0)
      assert curElogqPiMat.size == M
      ElogqPiMat[jj, jj+1:] = curElogqPiMat

    return ElogpZMat, sumLogPiMat, ElogqPiMat

  ######################################################### IO Utils
  #########################################################   for humans
    
  def get_info_string( self):
    ''' Returns human-readable name of this object'''
    s = 'HDP admixture model with K=%d comps. alpha=%.2f, gamma=%.2f'
    return s % (self.K, self.alpha0, self.gamma)
     
  ######################################################### IO Utils
  #########################################################   for machines
  def to_dict( self ):
    N, K = self.theta.shape
    e_theta = np.zeros( (N,K) )
    for ii in xrange(N):
      e_theta[ii,:] = self.theta[ii,:] / np.sum(self.theta[ii,:])
    return dict(U1=self.U1, U0=self.U0, theta=self.theta, e_theta = e_theta )
  
  def from_dict(self, Dict):
    self.inferType = Dict['inferType']
    self.K = Dict['K']
    self.U1 = Dict['U1']
    self.U0 = Dict['U0']
    self.set_helper_params()

  def get_prior_dict( self ):
    return dict(K=self.K, alpha0=self.alpha0, gamma=self.gamma)

