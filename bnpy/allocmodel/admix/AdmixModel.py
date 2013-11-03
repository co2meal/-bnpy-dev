'''
AdmixModel.py
Bayesian parametric admixture model with a finite number of components K

Attributes
-------
K        : # of components
alpha0   : scalar symmetric Dirichlet prior on mixture weights

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
None. No global structure is used except the (fixed) prior parameter alpha0.
Each document has its own mixture weights.

References
-------
Latent Dirichlet Allocation, by Blei, Ng, and Jordan
introduces a classic admixture model with Dirichlet-Mult observations.
'''
import numpy as np

from ..AllocModel import AllocModel
from bnpy.suffstats import SuffStatDict
from ...util import digamma, gammaln, logsumexp
from ...util import EPS, np2flatstr

class AdmixModel(AllocModel):
    def __init__(self, inferType, priorDict=None):
        if inferType == "EM":
            raise ValueError('AdmixModel cannot do EM. Only VB possible.')
        self.inferType = inferType
        self.K = 0
        if priorDict is None:
            self.alpha0 = 1.0 # Uniform!
        else:
            self.set_prior(priorDict)

    ####################################################### Suff Stat Calc
    ####################################################### 
    def get_global_suff_stats(self, Data, LP, doPrecompEntropy=None, **kwargs):
        ''' Count expected number of times each topic is used across all docs    
        '''
        wv = LP['word_variational']
        _, K = wv.shape
        SS = SuffStatDict(K=K)
        if doPrecompEntropy:
            SS.addPrecompELBOTerm('ElogpZ', self.E_log_pZ(Data, LP))
            SS.addPrecompELBOTerm('ElogqZ', self.E_log_qZ(Data, LP))
            SS.addPrecompELBOTerm('ElogpPI', self.E_log_pPI(Data, LP))
            SS.addPrecompELBOTerm('ElogqPI', self.E_log_qPI(Data, LP))
        return SS
         
    ####################################################### Calc Local Params
    ####################################################### (E-step)
    def get_keys_for_memoized_local_params(self):
        ''' Return list of string names of the LP fields
            that moVB needs to memoize across visits to a particular batch
        '''
        return ['alphaPi']

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
              doc_variational : nDoc x K matrix, 
                  row d has params for doc d's Dirichlet over the K topics
              word_variational : nDistinctWords x K matrix
                  row i has params for word i's Discrete distr over K topics                        
              DocTopicCount : nDoc x K matrix
        '''
        # When given no prev. local params LP, need to initialize from scratch
        # this forces likelihood to drive the first round of local assignments
        if 'alphaPi' not in LP:
            LP['alphaPi'] = np.ones((Data.nDoc,self.K))
        else:
            assert LP['alphaPi'].shape[1] == self.K

        LP = self.calc_ElogPi(LP)
        prevVec = LP['alphaPi'].flatten()

        # Allocate lots of memory once
        LP['word_variational'] = np.zeros(LP['E_logsoftev_WordsData'].shape)

        # Repeat until converged...
        for ii in xrange(nCoordAscentIters):
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
            if prevVec is not None and np.allclose(prevVec, curVec):
                break
            prevVec = curVec
        return LP
    

    def get_doc_variational( self, Data, LP):
        ''' Update document-topic variational parameters
        '''
        LP['alphaPi'] = LP['DocTopicCount'] + self.alpha0
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
        # Add doc-specific log prior to certain rows
        ElogPi = LP['E_logPi']
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


    ####################################################### Calc Global Params
    #######################################################   M-step
    def update_global_params( self, SS, rho=None, **kwargs ):
        ''' Update global parameters.
            However, parametric admixtures have no global alloc. parameters! 
            The mixture weights are document specific.
        '''
        self.K = SS.K
        
    def set_global_params(self, true_K=0, **kwargs):
        self.K = true_K


    ####################################################### Calc ELBO
    #######################################################   
    def calc_evidence( self, Data, SS, LP ):
        ''' Calculate ELBO terms related to allocation model
            p(z | pi) + p(pi | alpha) - q( phi | z) - q(theta | pi)
            where phi and theta represent our variational parameters
        '''        
        # Calculate ELBO assignments for document topic weights
        if SS.hasPrecompELBOTerm('ElogpPI'):
            E_log_pPI = SS.getPrecompELBOTerm('ElogpPI')
            E_log_qPI = SS.getPrecompELBOTerm('ElogqPI')
        elif SS.hasAmpFactor():
            E_log_pPI = SS['ampF']*self.E_log_pPI( Data, LP)
            E_log_qPI = SS['ampF']*self.E_log_qPI( Data, LP)
        else:
            E_log_pPI = self.E_log_pPI( Data, LP ) 
            E_log_qPI = self.E_log_qPI( Data, LP )
        
        # Calculate ELBO for word token assignment 
        if SS.hasPrecompELBOTerm('ElogpZ'):
            E_log_pZ = np.sum(SS.getPrecompELBOTerm('ElogpZ'))
            E_log_qZ = np.sum(SS.getPrecompELBOTerm('ElogqZ'))
        elif SS.hasAmpFactor():
            E_log_pZ = SS['ampF']*np.sum(self.E_log_pZ( Data, LP ))
            E_log_qZ = SS['ampF']*np.sum(self.E_log_qZ( Data, LP ))
        else:
            E_log_pZ = np.sum(self.E_log_pZ( Data, LP ))
            E_log_qZ = np.sum(self.E_log_qZ( Data, LP ))
        elbo_alloc = (E_log_pPI - E_log_qPI) + (E_log_pZ - E_log_qZ)        
        return elbo_alloc

    def E_log_pZ( self, Data, LP):
        ''' Returns K-length vector with E[ log p(Z) ] for each topic k
                E[ z_dwk ] * E[ log pi_{dk} ]
        '''
        E_log_pZ = LP['DocTopicCount'] * LP['E_logPi']
        return np.sum(E_log_pZ, axis=0)
    
    def E_log_qZ( self, Data, LP):  
        ''' Returns K-length vector with E[ log q(Z) ] for each topic k
                r_{dwk} * E[ log r_{dwk} ]
            where z_{dw} ~ Discrete( r_dw1 , r_dw2, ... r_dwK )
        '''
        wv = LP['word_variational']
        wv_logwv = wv * np.log(EPS + wv)
        E_log_qZ = np.dot(Data.word_count, wv_logwv)
        return E_log_qZ.sum(axis=0)    

    def E_log_pPI( self, Data, LP ):
        ''' Returns scalar value of E[ log p(Pi | alpha0)]
        '''
        K = self.K
        D = Data.nDoc
        logNormC = gammaln(K*self.alpha0)-K*gammaln(self.alpha0)
        logDirPDF = (self.alpha0 - 1) * LP['E_logPi'].sum()
        return (Data.nDoc * logNormC) + logDirPDF
    
    def E_log_qPI( self, Data, LP ):
        ''' Returns scalar value of E[ log q(Pi | alphaPi)]
        '''
        alph = LP['alphaPi']
        # logDirNormC : nDoc -len vector    
        logDirNormC = gammaln(alph.sum(axis=1)) - np.sum(gammaln(alph), axis=1)
        logDirPDF = np.sum((alph - 1.) * LP['E_logPi'])
        return np.sum(logDirNormC) + logDirPDF



    ####################################################### Accessors
    ####################################################### 
    def set_prior(self, PriorParamDict):
        self.alpha0 = PriorParamDict['alpha0']

    def to_dict( self ):
        return dict()              

    def from_dict(self, Dict):
        self.inferType = Dict['inferType']
        self.K = Dict['K']

    def get_prior_dict(self):
        return dict(alpha0=self.alpha0, K=self.K, inferType=self.inferType)
    
    def get_info_string(self):
        ''' Returns human-readable name of this object
        '''
        return 'Finite admixture model with K=%d comps, alpha=%.2f' % (self.K, self.alpha0)
    
    def get_model_name(self ):
        return 'admixture'
 
    def is_nonparametric(self):
        return False 