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
        return ['DocTopicCount']

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
        # doc_variational are the document level variational parameters phi
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
        LP['doc_variational'] = self.alpha0 + LP['DocTopicCount']
        LP['E_log_doc_variational'] = digamma( LP['doc_variational'] ) - digamma( LP['doc_variational'].sum(axis=1) )[:,np.newaxis]
        return LP
    
    def get_word_variational( self, Data, LP):
        ''' Update and return word-topic assignment variational parameters
        '''
        # We call this wv_temp, since this will become the unnormalized
        # variational parameter at the word level
        log_wv_temp = LP['E_logsoftev_WordsData'].copy() # so we can do += later
        for d in xrange( Data.nDoc ):
            start,stop = Data.doc_range[d,:]
            log_wv_temp[start:stop, :] += LP['E_log_doc_variational'][d,:]
        lprPerItem = logsumexp(log_wv_temp, axis=1 )
        # Normalize wv_temp to get actual word level variational parameters
        wv = np.exp(log_wv_temp - lprPerItem[:,np.newaxis])
        wv /= wv.sum(axis=1)[:,np.newaxis] # row normalize
        assert np.allclose(wv.sum(axis=1), 1)
        LP['word_variational'] = wv
        return LP
       
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
        E_log_pZ = LP["DocTopicCount"] * LP["E_log_doc_variational"]
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
        ''' Returns scalar value of E[ log p(PI | alpha0)]
        '''
        K = self.K
        D = Data.nDoc
        E_log_pPI = gammaln(K*self.alpha0)-K*gammaln(self.alpha0)    
        E_log_pPI *= D  # same prior over each Doc of data!
        for d in xrange( D ):
            E_log_pPI += (self.alpha0-1)*LP['E_log_doc_variational'][d,:].sum()
        return E_log_pPI
    
    def E_log_qPI( self, Data, LP ):
        ''' Returns scalar value of E[ log q(PI | doc_variational)]
        '''
        E_log_qPI = 0
        for d in xrange( Data.nDoc ):
            theta = LP['doc_variational'][d]
            E_log_qPI += gammaln(theta.sum()) - gammaln(theta).sum()
            E_log_qPI += np.inner(theta - 1,  LP['E_log_doc_variational'][d])
        return E_log_qPI

    ##############################################################    
    def update_global_params( self, SS, rho=None, **kwargs ):
        ''' Admixtures have no global allocation parameters! 
            The mixture weights are document specific.
        '''
        self.K = SS.K
        
    def set_global_params(self, true_K=0, **kwargs):
        self.K = true_K

    #################### GET METHODS #############################
    def set_prior(self, PriorParamDict):
        self.alpha0 = PriorParamDict['alpha0']
    
    def to_dict( self ):
        return dict()              
  
    def from_dict(self, Dict):
        self.inferType = Dict['inferType']
        self.K = Dict['K']
          
    def get_prior_dict( self ):
        return dict( alpha0=self.alpha0, K=self.K, inferType=self.inferType )
    
    def get_info_string( self):
        ''' Returns human-readable name of this object'''
        return 'Finite admixture model with K=%d comps, alpha=%.2f' % (self.K, self.alpha0)
    
    def get_model_name(self ):
        return 'admixture'
 
    def is_nonparametric(self):
        return False 

    def need_prev_local_params(self):
        return True
