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
doc_variational : nDoc x K matrix, 
                  row d has params for doc d's Dirichlet over the K topics
E_log_doc_variational : nDoc x K matrix
                  row d has E[ log mixture-weights for doc d ]
word_variational : nDistinctWords x K matrix
                  row i has params for word i's Discrete distr over K topics
DocTopicCount : nDoc x K matrix
                  entry d,k gives the expected number of times
                              that topic k is used in document d

Global Parameters (shared across all documents)
--------
U1, U0   : K-length vectors, params for variational distribution over 
           stickbreaking fractions v1, v2, ... vK
            q(v[k]) ~ Beta(U1[k], U0[k])


References
-------
Latent Dirichlet Allocation, by Blei, Ng, and Jordan
introduces a classic admixture model with Dirichlet-Mult observations.
'''
import numpy as np

import HDPVariationalOptimizer as HVO
from ..AllocModel import AllocModel
from bnpy.suffstats import SuffStatDict
from ...util import digamma, gammaln, logsumexp
from ...util import EPS, np2flatstr

class HDPModel(AllocModel):
    def __init__(self, inferType, priorDict=None):
        if inferType == "EM":
            raise ValueError('HDPModel cannot do EM. Only VB possible.')
        self.inferType = inferType
        self.K = 0
        self.set_prior(priorDict)

    def set_helper_params(self):
      ''' Set dependent attribs of this model, given the primary params U1, U0
          This includes expectations of various stickbreaking quantities
      '''
      E = HVO.calcExpectations(self.U1, self.U0)
      self.Ebeta = E['beta']
      self.Elogv = E['logv']
      self.Elog1mv = E['log1mv']
        
    ####################################################### Suff Stat Calc
    ####################################################### 
    def get_global_suff_stats(self, Data, LP, doPrecompEntropy=None, **kwargs):
        ''' Count expected number of times each topic is used across all docs    
        '''
        wv = LP['word_variational']
        _, K = wv.shape
        # Turn dim checking off, since some stats have dim K+1 instead of K
        SS = SuffStatDict(K=K, doCheck=False)
        SS.nDoc = Data.nDoc
        SS.sumLogPi = np.sum(LP['E_logPi'], axis=0)
        if doPrecompEntropy:
            SS.addPrecompELBOTerm('ElogpZ', self.E_log_pZ(Data, LP))
            SS.addPrecompELBOTerm('ElogqZ', self.E_log_qZ(Data, LP))
            SS.addPrecompELBOTerm('ElogpPI', self.E_log_pPI(Data, LP))
            SS.addPrecompELBOTerm('ElogqPI', self.E_log_qPI(Data, LP))
        return SS
        

    ####################################################### Calc Local Params
    ####################################################### (E-step)
    def calc_local_params( self, Data, LP, nCoordAscentIters=10):
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
                 row i has params for word i's Discrete distr over K active topics
              DocTopicCount : nDoc x K matrix
        '''
        # on the first iteration, initialize this to an empty array
        if 'DocTopicCount' not in LP:
            LP['DocTopicCount'] = np.zeros((Data.nDoc, self.K))
        
        doc_range = Data.doc_range
        word_count = Data.word_count
        prevVec = None
      
        # Repeat until converged...
        for ii in xrange(nCoordAscentIters):
            # Update doc_variational field of LP
            LP = self.get_doc_variational(Data, LP)

            # Update word_variational field of LP
            LP = self.get_word_variational(Data, LP)
        
            # Update DocTopicCount field of LP
            for d in xrange(Data.nDoc):
                start,stop = doc_range[d,:]
                LP['DocTopicCount'][d,:] = np.dot(
                                           word_count[start:stop],        
                                           LP['word_variational'][start:stop,:]
                                           )

            # Assess convergence 
            curVec = LP['DocTopicCount'].flatten()
            if prevVec is not None and np.allclose(prevVec, curVec):
                break
            prevVec = curVec
        return LP
    
    def get_doc_variational( self, Data, LP):
        ''' Update and return document-topic variational parameters
        '''
        zeroPad = np.zeros((Data.nDoc,1))
        DTCountMatZeroPadded = np.hstack([LP['DocTopicCount'], zeroPad])
        alph = DTCountMatZeroPadded + self.gamma*self.Ebeta
        LP['E_logPi'] = digamma(alph) - digamma(alph.sum(axis=1))[:,np.newaxis]
        LP['alphaPi'] = alph        
        return LP
    
    def get_word_variational( self, Data, LP):
        ''' Update and return word-topic assignment variational parameters
        '''
        # We call this wv_temp, since this will become the unnormalized
        # variational parameter at the word level
        log_wv_temp = LP['E_logsoftev_WordsData'].copy() # so we can do += later
        K = log_wv_temp.shape[1]
        for d in xrange( Data.nDoc ):
            start,stop = Data.doc_range[d,:]
            log_wv_temp[start:stop, :] += LP['E_logPi'][d,:K]
        lprPerItem = logsumexp(log_wv_temp, axis=1 )
        # Normalize wv_temp to get actual word level variational parameters
        wv = np.exp(log_wv_temp - lprPerItem[:,np.newaxis])
        wv /= wv.sum(axis=1)[:,np.newaxis] # row normalize
        assert np.allclose(wv.sum(axis=1), 1)
        LP['word_variational'] = wv
        return LP

    ####################################################### Calc ELBO
    #######################################################
    def calc_evidence( self, Data, SS, LP ):
        ''' Calculate ELBO terms related to allocation model
        '''   
        E_logpV = self.E_logpV()
        E_logqV = self.E_logqV()
     
        E_logpPi = self.E_logpPi(SS)
        E_logqPi = self.E_logqPi(LP)
        
        E_logpZ = np.sum(self.E_logpZ(Data, LP))
        E_logqZ = np.sum(self.E_logqZ(Data, LP))

        if SS.hasAmpFactor():
            E_logqPi *= SS.ampF
            E_logpZ *= SS.ampF
            E_logqZ *= SS.ampF

        elbo = (E_logpPi - E_logqPi) \
               + (E_logpZ - E_logqZ) \
               + (E_logpV - E_logqV)
        return elbo

    ####################################################### ELBO terms for Z
    def E_logpZ( self, Data, LP):
        ''' Returns K-length vector with E[ log p(Z) ] for each topic k
                E[ z_dwk ] * E[ log pi_{dk} ]
        '''
        E_logpZ = LP["DocTopicCount"] * LP["E_logPi"][:, :self.K]
        return E_logpZ
    
    def E_logqZ( self, Data, LP):  
        ''' Returns K-length vector with E[ log q(Z) ] for each topic k
                r_{dwk} * E[ log r_{dwk} ]
            where z_{dw} ~ Discrete( r_dw1 , r_dw2, ... r_dwK )
        '''
        wv = LP['word_variational']
        wv_logwv = wv * np.log(EPS + wv)
        E_log_qZ = np.dot(Data.word_count, wv_logwv)
        return E_log_qZ.sum(axis=0)    

    ####################################################### ELBO terms for Pi
    def E_logpPi(self, SS):
        ''' Returns scalar value of E[ log p(PI | alpha0)]
        '''
        K = SS.K
        kvec = K + 1 - np.arange(1, K+1)
        logDirNormC = gammaln(self.gamma) + (K+1) * np.log(self.gamma)
        logDirNormC += np.sum(self.Elogv) + np.inner(kvec, self.Elog1mv)

        logDirPDF = np.inner(self.gamma * self.Ebeta - 1, SS.sumLogPi)
        return SS.nDoc * logDirNormC + logDirPDF

    def E_logqPi(self, LP):
        ''' Returns scalar value of E[ log q(PI)]
        '''
        alph = LP['alphaPi']        
        logDirNormC = gammaln(alph.sum(axis=1)) - np.sum(gammaln(alph), axis=1)
        logDirPDF = np.sum((alph - 1.) * LP['E_logPi'])
        return np.sum(logDirNormC) + logDirPDF

    ####################################################### ELBO terms for V
    def E_logpV(self):
        logBetaNormC = gammaln(self.alpha0 + 1.) \
                       - gammaln(self.alpha0)
        logBetaPDF = (self.alpha0-1.) * np.sum(self.Elog1mv)
        return self.K*logBetaNormC + logBetaPDF

    def E_logqV(self):
        logBetaNormC = gammaln(self.U1 + self.U0) \
                        - gammaln(self.U0) - gammaln(self.U1)
        logBetaPDF =   np.inner(self.U1 - 1., self.Elogv) \
                      + np.inner(self.U0 - 1., self.Elog1mv)
        return np.sum(logBetaNormC) + logBetaPDF

    ####################################################### Update global params
    #######################################################
    def update_global_params_VB(self, SS, **kwargs):
        ''' Admixtures have no global allocation parameters! 
            The mixture weights are document specific.
        '''
        self.K = SS.K
        U1, U0 = HVO.estimate_u(K=self.K, alpha0=self.alpha0, gamma=self.gamma,
                     sumLogPi=SS.sumLogPi, nDoc=SS.nDoc)
        self.U1 = U1
        self.U0 = U0
        self.set_helper_params()
        
    def update_global_params_soVB(self, SS, rho, **kwargs):
        assert self.K == SS.K
        U1, U0 = HVO.estimate_u(K=self.K, alpha0=self.alpha0, gamma=self.gamma,
                     sumLogPi=SS.sumLogPi, nDoc=SS.nDoc)
        self.U1 = rho * U1 + (1-rho) * self.U1
        self.U0 = rho * U0 + (1-rho) * self.U0
        self.set_helper_params()

    #################### GET METHODS #############################
    def set_prior(self, PriorParamDict):
        self.alpha0 = PriorParamDict['alpha0']
        self.gamma = PriorParamDict['gamma']
    
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
    
    def get_info_string( self):
        ''' Returns human-readable name of this object'''
        return 'HDP admixture model with K=%d comps. alpha=%.2f,gamma=%.2f' % (self.K, self.alpha0, self.gamma)
     
    def get_keys_for_memoized_local_params(self):
        ''' Return list of string names of the LP fields
            that moVB needs to memoize across visits to a particular batch
        '''
        return ['DocTopicCount']

    def is_nonparametric(self):
        return True
