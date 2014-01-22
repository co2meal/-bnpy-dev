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
alphaPi : nDoc x K matrix, 
             row d has params for doc d's distribution pi[d] over the K topics
             q( pi[d] ) ~ Dir( alphaPi[d] )
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

import HDPVariationalOptimizer as HVO
from ..AllocModel import AllocModel
from bnpy.suffstats import SuffStatBag
from ...util import digamma, gammaln, logsumexp
from ...util import EPS, np2flatstr

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
      E = HVO.calcExpectations(self.U1, self.U0)
      self.Ebeta = E['beta']
      self.Elogv = E['logv']
      self.Elog1mv = E['log1mv']
        

  ######################################################### Accessors
  #########################################################
    def get_keys_for_memoized_local_params(self):
        ''' Return list of string names of the LP fields
            that moVB needs to memoize across visits to a particular batch
        '''
        return ['alphaPi']

  ######################################################### Local Params
  #########################################################

    def calc_local_params(self, Data, LP, nCoordAscentItersLP=20, convThrLP=0.01, **kwargs):
        ''' Calculate document-specific quantities (E-step)
          Alternate updates to two terms until convergence
            (1) Approx posterior on topic-token assignment
                 q(word_variational | word_token_variables)
            (2) Approx posterior on doc-topic probabilities
                 q(doc_variational | document_topic_variables)

          Returns
          -------
          LP : local params dict, with fields
              Pi : nDoc x K+1 matrix, 
                 row d has params for doc d's Dirichlet over K+1 topics
              word_variational : nDistinctWords x K matrix
                 row i has params for word i's Discrete distr over K topics
              DocTopicCount : nDoc x K matrix
        '''
        # When given no prev. local params LP, need to initialize from scratch
        # this forces likelihood to drive the first round of local assignments
        if 'alphaPi' not in LP:
            LP['alphaPi'] = np.ones((Data.nDoc,self.K+1))
        else:
            assert LP['alphaPi'].shape[1] == self.K + 1

        LP = self.calc_ElogPi(LP)
        prevVec = LP['alphaPi'].flatten()

        # Allocate lots of memory once
        LP['word_variational'] = np.zeros(LP['E_logsoftev_WordsData'].shape)

        # Repeat until converged...
        for ii in xrange(nCoordAscentItersLP):
            # Update word_variational field of LP
            LP = self.get_word_variational(Data, LP)
        
            # Update DocTopicCount field of LP
            LP['DocTopicCount'] = np.zeros((Data.nDoc,self.K))
            for d in xrange(Data.nDoc):
                start,stop = Data.doc_range[d,:]
                LP['DocTopicCount'][d,:] = np.dot(
                                           Data.word_count[start:stop],        
                                           LP['word_variational'][start:stop,:]
                                           )
            # Update doc_variational field of LP
            LP = self.get_doc_variational(Data, LP)
            LP = self.calc_ElogPi(LP)

            # Assess convergence
            curVec = LP['alphaPi'].flatten()
            if np.allclose(prevVec, curVec, atol=convThrLP):
                break
            prevVec = curVec
        return LP

    def get_doc_variational( self, Data, LP):
        ''' Update document-topic variational parameters
        '''
        zeroPad = np.zeros((Data.nDoc,1))
        DTCountMatZeroPad = np.hstack([LP['DocTopicCount'], zeroPad])
        LP['alphaPi'] = DTCountMatZeroPad + self.gamma*self.Ebeta
        return LP

    def calc_ElogPi(self, LP):
        ''' Update expected log topic probability distr. for each document d
        '''
        alph = LP['alphaPi']
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
        # Take exp of wv in numerically stable manner (first subtract the max)
        #  in-place so no new allocations occur
        wv -= np.max(wv, axis=1)[:,np.newaxis]
        np.exp(wv, out=wv)
        # Normalize, so rows of wv sum to one
        wv /= wv.sum(axis=1)[:,np.newaxis]
        assert np.allclose(LP['word_variational'].sum(axis=1), 1)
        return LP


  ######################################################### Suff Stats
  #########################################################
    def get_global_suff_stats(self, Data, LP, doPrecompEntropy=False, 
                                              doPrecompMergeEntropy=False):
        ''' Count expected number of times each topic is used across all docs    
        '''
        wv = LP['word_variational']
        _, K = wv.shape
        # Turn dim checking off, since some stats have dim K+1 instead of K
        SS = SuffStatBag(K=K, D=Data.vocab_size)
        SS.setField('nDoc', Data.nDoc, dims=None)
        sumLogPi = np.sum(LP['E_logPi'], axis=0)
        SS.setField('sumLogPiActive', sumLogPi[:K], dims='K')
        SS.setField('sumLogPiUnused', sumLogPi[-1], dims=None)
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
            ElogqZMat = self.E_logqZ_memo_terms_for_merge(Data, LP)
            SS.setMergeTerm('ElogpZ', ElogpZMat, dims=('K','K'))
            SS.setMergeTerm('ElogqZ', ElogqZMat, dims=('K','K'))
            SS.setMergeTerm('ElogqPiActive', ElogqPiMat, dims=('K','K'))
            SS.setMergeTerm('sumLogPiActive', sLgPiMat, dims=('K','K'))
        return SS


  ######################################################### Global Params
  #########################################################
    def update_global_params_VB(self, SS, **kwargs):
        ''' Update global parameters that control topic probabilities beta
            beta[k] ~ Beta( U1[k], U0[k])
        '''
        self.K = SS.K
        if hasattr(self, 'U1'):
          initU1 = self.U1
          initU0 = self.U0
        else:
          initU1 = None
          initU0 = None
        sumLogPi = np.hstack([SS.sumLogPiActive, SS.sumLogPiUnused])
        U1, U0 = HVO.estimate_u(K=self.K, alpha0=self.alpha0, gamma=self.gamma,
                     sumLogPi=sumLogPi, nDoc=SS.nDoc, 
                     initU1=initU1, initU0=initU0)
        self.U1 = U1
        self.U0 = U0
        self.set_helper_params()
        
    def update_global_params_soVB(self, SS, rho, **kwargs):
        assert self.K == SS.K
        sumLogPi = np.hstack([SS.sumLogPiActive, SS.sumLogPiUnused])
        U1, U0 = HVO.estimate_u(K=self.K, alpha0=self.alpha0, gamma=self.gamma,
                     sumLogPi=sumLogPi, nDoc=SS.nDoc,
                     initU1=self.U1, initU0=self.U0)
        self.U1 = rho * U1 + (1-rho) * self.U1
        self.U0 = rho * U0 + (1-rho) * self.U0
        self.set_helper_params()

    def set_global_params(self, K=0, beta=None, U1=None, U0=None, 
                                Ebeta=None, EbetaLeftover=None, **kwargs):
        self.K = K
        if U1 is not None and U0 is not None:
          self.U1 = U1
          self.U0 = U0
        if Ebeta is not None and EbetaLeftover is not None:
          Ebeta = np.squeeze(Ebeta)
          EbetaLeftover = np.squeeze(EbetaLeftover)
          beta = np.hstack( [Ebeta, EbetaLeftover])
        elif beta is not None:
          assert beta.size == K
          beta = np.hstack([beta, np.min(beta)/100.])
          beta = beta/np.sum(beta)
        if beta is not None:
          assert abs(np.sum(beta) - 1.0) < 0.01
          vMean = HVO.beta2v(beta)
          vMass = 10
          self.U1 = vMass * vMean
          self.U0 = vMass * (1-vMean)
        else:
          raise ValueError('Bad HDP parameters')          
        assert self.U1.size == K
        assert self.U0.size == K
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

        elbo = E_logpPi - E_logqPi \
               + E_logpZ - E_logqZ \
               + E_logpV - E_logqV
        return elbo

    ####################################################### ELBO terms for Z
    def E_logpZ( self, Data, LP):
        ''' Returns K-length vector with E[ log p(Z) ] for each topic k
                E[ z_dwk ] * E[ log pi_{dk} ]
        '''
        K = LP['DocTopicCount'].shape[1]
        E_logpZ = LP["DocTopicCount"] * LP["E_logPi"][:, :K]
        return np.sum(E_logpZ, axis=0)

    def E_logqZ( self, Data, LP):  
        ''' Returns K-length vector with E[ log q(Z) ] for each topic k
                r_{dwk} * E[ log r_{dwk} ]
            where z_{dw} ~ Discrete( r_dw1 , r_dw2, ... r_dwK )
        '''
        wv = LP['word_variational']
        wv_logwv = wv * np.log(EPS + wv)
        E_log_qZ = np.dot(Data.word_count, wv_logwv)
        return E_log_qZ

    def E_logqZ_memo_terms_for_merge(self, Data, LP):
        ''' Returns KxK matrix 
        ''' 
        wv = LP['word_variational']
        wv += EPS # Make sure all entries > 0 before taking log
        ElogqZMat = np.zeros((self.K, self.K))
        for jj in range(self.K):
            J = self.K - jj - 1 # num of pairs for comp jj
            # curWV : nObs x J, resp for each data item under each merge with jj
            curWV = wv[:,jj][:,np.newaxis] + wv[:,jj+1:]
            # curWV : nObs x J, entropy for each data item
            curWV *= np.log(curWV)
            # curE_logqZ : J-vector, entropy for Data under each merge with jj
            curE_logqZ = np.dot(Data.word_count, curWV)            
            assert curE_logqZ.size == J
            ElogqZMat[jj,jj+1:] = curE_logqZ
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
        alph = LP['alphaPi']
        # logDirNormC : nDoc -len vector    
        logDirNormC = gammaln(alph.sum(axis=1)) - np.sum(gammaln(alph), axis=1)
        logDirPDF = np.sum((alph - 1.) * LP['E_logPi'])
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
        return 'HDP admixture model with K=%d comps. alpha=%.2f, gamma=%.2f' % (self.K, self.alpha0, self.gamma)
     
  ######################################################### IO Utils
  #########################################################   for machines
    def to_dict( self ):
        return dict(U1=self.U1, U0=self.U0)              
  
    def from_dict(self, Dict):
        self.inferType = Dict['inferType']
        self.K = Dict['K']
        if 'U1' in Dict:
          self.U1 = Dict['U1']
          self.U0 = Dict['U0']
          self.set_helper_params()

    def get_prior_dict( self ):
        return dict(K=self.K, alpha0=self.alpha0, gamma=self.gamma)

