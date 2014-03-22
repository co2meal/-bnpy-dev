'''
HDPModel.py
Bayesian nonparametric admixture model with unbounded number of components K

Attributes
-------
K        : # of components
alpha0   : scalar concentration param for global-level stick-breaking params v
gamma    : scalar conc. param for document-level mixture weights pi[d]

Local Parameters (document-specific)
--------
theta : nDoc x K matrix, 
             row d has params for doc d's distribution pi[d] over the K topics
             q( pi[d] ) ~ Dir( theta[d] )
E_logPi : nDoc x K matrix
             row d has E[ log pi[d] ]
DocTopicCount : nDoc x K matrix
                  entry d,k gives the expected number of times
                              that topic k is used in document d
word_variational : nDistinctWords x K matrix
                  row i has params for word i's Discrete distr over K topics

Global Parameters (shared across all documents)
--------
U1, U0   : K-length vectors, params for variational distribution over 
           stickbreaking fractions v1, v2, ... vK
            q(v[k]) ~ Beta(U1[k], U0[k])
'''
import numpy as np

import LocalStepBagOfWords
import OptimizerForHDPFullVarModel as OptimHDP
from ..AllocModel import AllocModel

from bnpy.suffstats import SuffStatBag
from ...util import NumericUtil
from ...util import digamma, gammaln
from ...util import EPS, np2flatstr
import logging
Log = logging.getLogger('bnpy')


class HDPModel(AllocModel):

  ######################################################### Constructors
  #########################################################
  def __init__(self, inferType, priorDict=None):
        if inferType == "EM":
            raise ValueError('HDPModel cannot do EM. Only VB possible.')
        self.inferType = inferType
        self.K = 0
        self.set_prior(priorDict)

  def set_prior(self, PriorParamDict):
        self.alpha0 = PriorParamDict['alpha0']
        self.gamma = PriorParamDict['gamma']
    
  def set_helper_params(self):
      ''' Set dependent attribs of this model, given the primary params U1, U0
          This includes expectations of various stickbreaking quantities
      '''
      assert self.U1.size == self.K
      assert self.U0.size == self.K
      E = OptimHDP._calcExpectations(self.U1, self.U0)
      self.Ebeta = E['beta']
      self.Elogv = E['logv']
      self.Elog1mv = E['log1-v']

  ######################################################### Accessors
  #########################################################
  def get_keys_for_memoized_local_params(self):
        ''' Return list of string names of the LP fields
            that moVB needs to memoize across visits to a particular batch
        '''
        return ['theta']

  ######################################################### Local Params
  #########################################################
  def calc_local_params(self, Data, LP, **kwargs):
    ''' Calculate document-specific quantities (E-step)
          Alternate updates to two terms until convergence
            (1) Approx posterior on topic-token assignment
            (2) Approx posterior on doc-topic probabilities

          Returns
          -------
          LP : local params dict, with fields
              theta : nDoc x K+1 matrix, 
                 row d has params for doc d's Dirichlet over K+1 topics
              E_logPi : nDoc x K+1 matrix,
                 row d has doc d's expected log probability of each topic
              word_variational : nDistinctWords x K matrix
                 row i has params for word i's Discrete distr over K topics
              DocTopicCount : nDoc x K matrix
    '''
    LP = LocalStepBagOfWords.calcLocalDocParams(Data, LP, 
                                  self.gamma*self.Ebeta[:-1],
                                  unusedTopicPrior=self.gamma*self.Ebeta[-1],
                                  **kwargs)

    LP['word_variational'] = LP['expEloglik']
    for d in xrange(Data.nDoc):
      start = Data.doc_range[d,0]
      stop  = Data.doc_range[d,1]
      LP['word_variational'][start:stop] *= LP['expElogpi'][d]
    LP['word_variational'] /= LP['sumRTilde'][:, np.newaxis]

    assert np.allclose( LP['word_variational'].sum(axis=1), 1.0)
    return LP



  def get_doc_variational( self, Data, LP):
    ''' Update document-topic variational parameters
    '''
    zeroPad = np.zeros((Data.nDoc,1))
    DTCountMatZeroPad = np.hstack([LP['DocTopicCount'], zeroPad])
    LP['theta'] = DTCountMatZeroPad + self.gamma*self.Ebeta
    return LP

  def calc_ElogPi(self, LP):
        ''' Update expected log topic probability distr. for each document d
        '''
        alph = LP['theta']
        LP['E_logPi'] = digamma(alph) - digamma(alph.sum(axis=1))[:,np.newaxis]
        return LP
    
  def get_word_variational( self, Data, LP):
        ''' Update and return word-topic assignment variational parameters
        '''
        # Operate on wv matrix, which is nDistinctWords x K
        #  has been preallocated for speed (so we can do += later)
        wv = LP['word_variational']         
        K = wv.shape[1]        
        # Fill in entries of wv with log likelihood terms
        wv[:] = LP['E_logsoftev_WordsData']
        # Add doc-specific log prior to doc-specific rows
        ElogPi = LP['E_logPi'][:,:K]
        for d in xrange(Data.nDoc):
            wv[Data.doc_range[d,0]:Data.doc_range[d,1], :] += ElogPi[d,:]
        NumericUtil.inplaceExpAndNormalizeRows(wv)
        assert np.allclose(LP['word_variational'].sum(axis=1), 1)
        return LP


  ######################################################### Suff Stats
  #########################################################
  def get_global_suff_stats(self, Data, LP, doPrecompEntropy=False, 
                                              doPrecompMergeEntropy=False,
                                              mPairIDs=None):
    ''' Count expected number of times each topic is used across all docs    
    '''
    wv = LP['word_variational']
    _, K = wv.shape
    SS = SuffStatBag(K=K, D=Data.vocab_size)
    SS.setField('nDoc', Data.nDoc, dims=None)
    sumLogPi = np.sum(LP['E_logPi'], axis=0)
    SS.setField('sumLogPiActive', sumLogPi, dims='K')
    SS.setField('sumLogPiUnused', np.sum(LP['E_logPi_u'], axis=0), dims=None)

    if doPrecompEntropy:
      # Z terms
      SS.setELBOTerm('ElogpZ', self.E_logpZ(Data, LP), dims='K')
      SS.setELBOTerm('ElogqZ', self.E_logqZ(Data, LP), dims='K')
      # Pi terms
      # Note: no terms needed for ElogpPI
      # SS already has field sumLogPi, which is sufficient for this term
      ElogqPiC, ElogqPiA, ElogqPiU = self.E_logqPi_Memoized_from_LP(LP)
      SS.setELBOTerm('ElogqPiConst', ElogqPiC, dims=None)
      SS.setELBOTerm('ElogqPiActive', ElogqPiA, dims='K')
      SS.setELBOTerm('ElogqPiUnused', ElogqPiU, dims=None)

    if doPrecompMergeEntropy:
      ElogpZMat, sLgPiMat, ElogqPiMat = self.memo_elbo_terms_for_merge(LP)
      ElogqZMat = self.E_logqZ_memo_terms_for_merge(Data, LP, mPairIDs)
      SS.setMergeTerm('ElogpZ', ElogpZMat, dims=('K','K'))
      SS.setMergeTerm('ElogqZ', ElogqZMat, dims=('K','K'))
      SS.setMergeTerm('ElogqPiActive', ElogqPiMat, dims=('K','K'))
      SS.setMergeTerm('sumLogPiActive', sLgPiMat, dims=('K','K'))
    return SS


  ######################################################### Global Params
  #########################################################

  def update_global_params_VB(self, SS, comps=None, **kwargs):
        ''' Update global parameters that control topic probabilities
            v[k] ~ Beta( U1[k], U0[k])
        '''
        self.K = SS.K
        u = self._estimate_u(SS, **kwargs)
        if comps is None:
          self.U1 = u[:self.K]
          self.U0 = u[self.K:]
        else:
          self.U1[comps] = u[comps]
          self.U0[comps] = u[self.K + np.asarray(comps, dtype=np.int64)]
        if self.U1.size > self.K:
          self.U1 = self.U1[:self.K]
          self.U0 = self.U1[:self.K]
        self.set_helper_params()
        
  def update_global_params_soVB(self, SS, rho, **kwargs):
        assert self.K == SS.K
        u = self._estimate_u(SS)
        self.U1 = rho * u[:self.K] + (1-rho) * self.U1
        self.U0 = rho * u[self.K:] + (1-rho) * self.U0
        self.set_helper_params()

  def _estimate_u(self, SS, mergeCompB=None, **kwargs):
        ''' Calculate best 2*K-vector u via L-BFGS gradient descent
              performing multiple tries in case of numerical issues
        '''
        if hasattr(self, 'U1') and self.U1.size == self.K:
          initU = np.hstack([self.U1, self.U0])
        elif hasattr(self, 'U1') and mergeCompB is not None \
                                 and self.U1.size == self.K + 1:
          U1 = np.delete(self.U1, mergeCompB)
          U0 = np.delete(self.U0, mergeCompB)
          assert U0.size == self.K
          initU = np.hstack([U1, U0])
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
            print 'failed'
            Log.error('***** Optim failed. Stay put. ' + str(error))
            return # EXIT with current state, failed to update
          else:
            print 'prior'
            Log.error('***** Optim failed. Stuck at prior. ' + str(error))
            u = initU # fall back on the prior otherwise
        return u

  def set_global_params(self, hmodel=None, 
                                U1=None, U0=None, 
                                K=0, beta=None, topic_prior=None,
                                Ebeta=None, EbetaLeftover=None, **kwargs):
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
        assert beta.size == self.K + 1
    
        # Now, use the specified value of beta to find the best U1, U0
        self.U1, self.U0 = self._convert_beta2u(beta)        
        assert np.all( self.U1 >= 1.0 - 0.00001)
        assert np.all( self.U0 >= self.alpha0 - 0.00001)
        assert self.U1.size == self.K
        assert self.U0.size == self.K
        self.set_helper_params()

  def _convert_beta2u(self, beta):
    ''' Given a vector beta (size K+1),
          return educated guess for vectors u1, u0

        Returns
        --------
          U1 : 1D array, size K
          U0 : 1D array, size K
    '''
    assert abs(np.sum(beta) - 1.0) < 0.001
    vMean = OptimHDP.beta2v(beta)
    # for each k=1,2...K
    #  find the multiplier vMass[k] such that both are true
    #  1) vMass[k] * vMean[k] > 1.0
    #  2) vMass[k] * (1-vMean[k]) > self.alpha0
    vMass = np.maximum( 1./vMean , self.alpha0/(1.-vMean))
    U1 = vMass * vMean
    U0 = vMass * (1-vMean)
    return U1, U0    
    
  def insert_global_params(self, beta=None, **kwargs):
    Knew = beta.size
    beta = np.hstack([beta, np.min(beta)/100.])
    beta = beta/np.sum(beta)
    vMean = OptimHDP.beta2v(beta)
    vMass = np.maximum( 1./vMean , self.alpha0/(1.-vMean))

    self.K += Knew
    self.U1 = np.append(self.U1, vMass * vMean )
    self.U0 = np.append(self.U0, vMass * (1-vMean))

    assert self.U1.size == self.K
    assert self.U0.size == self.K
    self.set_helper_params()    

  ######################################################### Evidence
  #########################################################  
  def calc_evidence( self, Data, SS, LP ):
        ''' Calculate ELBO terms related to allocation model
        '''   
        E_logpV = self.E_logpV()
        E_logqV = self.E_logqV()
     
        E_logpPi = self.E_logpPi(SS)
        if SS.hasELBOTerms():
          E_logqPi = SS.getELBOTerm('ElogqPiConst') \
                      + SS.getELBOTerm('ElogqPiUnused') \
                      + np.sum(SS.getELBOTerm('ElogqPiActive'))
          E_logpZ = np.sum(SS.getELBOTerm('ElogpZ'))
          E_logqZ = np.sum(SS.getELBOTerm('ElogqZ'))
        else:
          E_logqPi = self.E_logqPi(LP)
          E_logpZ = np.sum(self.E_logpZ(Data, LP))
          E_logqZ = np.sum(self.E_logqZ(Data, LP))

        if SS.hasAmpFactor():
            E_logqPi *= SS.ampF
            E_logpZ *= SS.ampF
            E_logqZ *= SS.ampF

        elbo = E_logpPi - E_logqPi
        elbo += E_logpZ - E_logqZ
        elbo += E_logpV - E_logqV
        return elbo

  ####################################################### ELBO terms for Z
  def E_logpZ( self, Data, LP):
    ''' Returns K-length vector with E[log p(Z)] for each topic k
    '''
    return np.sum(LP['DocTopicCount'] * LP['E_logPi'], axis=0)

  def E_logqZ( self, Data, LP):  
    ''' Returns K-length vector with E[ log q(Z) ] for each topic k
            r_{dwk} * E[ log r_{dwk} ]
        where z_{dw} ~ Discrete( r_dw1 , r_dw2, ... r_dwK )
    '''
    wv = LP['word_variational']
    wv += EPS # Make sure all entries > 0 before taking log
    return NumericUtil.calcRlogRdotv(wv, Data.word_count)

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
    K = SS.K
    kvec = K + 1 - np.arange(1, K+1)
    # logDirNormC : scalar norm const that applies to each iid draw pi_d
    logDirNormC = gammaln(self.gamma) + (K+1) * np.log(self.gamma)
    logDirNormC += np.sum(self.Elogv) + np.inner(kvec, self.Elog1mv)
    # logDirPDF : scalar sum over all doc's pi_d
    sumLogPi = np.hstack([SS.sumLogPiActive, SS.sumLogPiUnused])
    logDirPDF = np.inner(self.gamma * self.Ebeta - 1., sumLogPi)
    return (SS.nDoc * logDirNormC) + logDirPDF

  def E_logqPi(self, LP):
    ''' Returns scalar value of E[ log q(PI)],
          calculated directly from local param dict LP
    '''
    theta = LP['theta']
    utheta = LP['theta_u']
    logDirNormC = gammaln(utheta + theta.sum(axis=1)) \
                  - (gammaln(utheta) + np.sum(gammaln(theta), axis=1))
    logDirPDF = np.sum((theta - 1.) * LP['E_logPi']) \
                  + (utheta - 1.) * np.sum(LP['E_logPi_u'])
    return np.sum(logDirNormC) + logDirPDF

  def E_logqPi_Memoized_from_LP(self, LP):
    ''' Returns three variables 
                logDirNormC (scalar),
                logqPiActive (length K)
                logqPiUnused (scalar)
                whose sum is equal to E[log q(PI)]
            when added to other results of this function from different batches,
                the sum is equal to E[log q(PI)] of the entire dataset
    '''
    nDoc = LP['theta'].shape[0]
    logDirNormC = np.sum(gammaln(LP['theta_u'] + LP['theta'].sum(axis=1)))
    piEntropyVec = np.sum((LP['theta'] - 1.) * LP['E_logPi'], axis=0) \
                     - np.sum(gammaln(LP['theta']),axis=0)
    piEntropyUnused = np.sum(LP['theta_u'] * LP['E_logPi_u'], axis=0) \
                     - nDoc * gammaln(LP['theta_u']) 
    return logDirNormC, piEntropyVec, piEntropyUnused


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
    CMat = LP['DocTopicCount']
    theta = LP['theta']
    digammasumTheta = digamma(LP['theta_u'] + theta.sum(axis=1))[:, np.newaxis]

    ElogpZMat = np.zeros((self.K, self.K))
    sumLogPiMat = np.zeros((self.K, self.K))
    ElogqPiMat = np.zeros((self.K, self.K))
    for jj in range(self.K):
      M = self.K - jj - 1
      # nDoc x M matrix, theta_{dm} for each merge pair m with comp jj
      mergeTheta = theta[:,jj][:,np.newaxis] + theta[:, jj+1:]
      # nDoc x M matrix, E[log pi_m] for each merge pair m with comp jj
      mergeElogPi = digamma(mergeTheta) - digammasumTheta
      assert mergeElogPi.shape[1] == M
      # nDoc x M matrix, count for merged topic m each doc
      mergeCMat = CMat[:, jj][:,np.newaxis] + CMat[:, jj+1:]
      ElogpZMat[jj, jj+1:] = np.sum(mergeCMat * mergeElogPi, axis=0)
          
      sumLogPiMat[jj, jj+1:] = np.sum(mergeElogPi,axis=0)
      curElogqPiMat = np.sum((mergeTheta-1.) * mergeElogPi, axis=0) \
                                      - np.sum(gammaln(mergeTheta),axis=0)
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
    return dict(U1=self.U1, U0=self.U0)              
  
  def from_dict(self, Dict):
    self.inferType = Dict['inferType']
    self.K = Dict['K']
    self.U1 = Dict['U1']
    self.U0 = Dict['U0']
    self.set_helper_params()

  def get_prior_dict( self ):
    return dict(K=self.K, alpha0=self.alpha0, gamma=self.gamma)

