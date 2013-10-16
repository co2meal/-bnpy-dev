'''
  MixModel.py
     Bayesian parametric admixture model with a finite number of components K

  Provides code for performing variational Bayesian inference,
     using a mean-field approximation.
     
 Author: Mike Hughes (mike@michaelchughes.com)

 Parameters
 -------
    K        : # of components
    alpha0   : scalar hyperparameter of symmetric Dirichlet prior on mix. weights

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

class AdmixModel( AllocModel ):
    def __init__( self, inferType, priorDict=None):
        if inferType == "EM":
            raise ValueError('AdmixModel cannot do EM. Only VB learning possible.')
        self.inferType = inferType
        self.K = 0
        if priorDict is None:
            self.alpha0 = 1.0 # Uniform!
        else:
            self.set_prior(priorDict)

    ########## Suff Stat Calc   
    def get_global_suff_stats( self, Data, LP, doPrecompEntropy=None, **kwargs):
        ''' Just count expected # assigned to each cluster across all Docs, as usual'''
        wv = LP['word_variational']
        _, K = wv.shape
        WC = Data.WC
        doc_variational_ss = np.zeros( (Data.nDocs, K) )
        # Loop through documents

        for d in xrange(Data.nDocs):
            start,stop = Data.DOC_ID[d,:]
            # get document-level sufficient statistics 
            doc_variational_ss[d,:] = np.dot( WC[start:stop,1], wv[start:stop,:] )

        #doc_variational_ss = Data.true_td.T     
        SS = SuffStatDict(doc_variational_ss = doc_variational_ss)
        SS.K = K
        self.K = K
        return SS
         
    ####### The E-Step is completed here #########################  
    def calc_local_params( self, Data, LP ):
        ''' E-step
          alternate between these updates until convergence
             q(phi | z)  (posterior on topic-token assignment)
         and q(theta | pi)  (posterior on Doc-topic distribution)'''
        
        # doc_variational are the document level variational parameters phi
        # on the first iteration, initialize this to an empty array
        try:
            LP['doc_variational']
        except KeyError:
            LP['doc_variational'] = np.zeros( (Data.nDocs, self.K) )

        DOC_ID = Data.DOC_ID
        WC = Data.WC
        prevVec = None
        for ii in xrange( 4 ):
            LP = self.get_doc_variational( Data, LP)
            LP = self.get_word_variational( Data, LP)        
            for d in xrange( Data.nDocs ):
                start,stop = DOC_ID[d,:]
                #doc_variational = Freq of unique word counts for document d x wvonsibilities
                LP['doc_variational'][d,:] = np.dot( WC[start:stop,1], LP['word_variational'][start:stop,:] )  
            curVec = LP['doc_variational'].flatten()
            if prevVec is not None and np.allclose( prevVec, curVec ):
                break
            prevVec = curVec
        return LP
    
    # Returns the document level variational parameters
    def get_doc_variational( self, Data, LP):
        LP['doc_variational'] = self.alpha0 + LP['doc_variational']
        LP['E_log_doc_variational'] = digamma( LP['doc_variational'] ) - digamma( LP['doc_variational'].sum(axis=1) )[:,np.newaxis]
        # Added this line to aid human inspection. self.Elogw is never used except to print status
        self.ElogPI = LP['E_log_doc_variational']
        return LP
    
    # Returns word level variational parameters
    def get_word_variational( self, Data, LP):
        # We call this wv_temp, since this will become the unnormalized
        # variational parameter at the word level
        wv_temp = LP['E_log_obs_word'].copy() # so we can do += later
        # Loop through documents and add expectations of document i and topics 1:K
        # Calculate the local variational parameters phi associated with our word-level observations
        for d in xrange( Data.nDocs ):
            start,stop = Data.DOC_ID[d,:]
            wv_temp[start:stop, :] += LP['E_log_doc_variational'][d,:]
        lprPerItem = logsumexp( wv_temp, axis=1 )
        # Normalize wv_temp to get actual word level variational parameters
        wv = np.exp( wv_temp-lprPerItem[:,np.newaxis] )
        wv /= wv.sum( axis=1)[:,np.newaxis] # row normalize
        assert np.allclose( wv.sum(axis=1), 1)
        LP['word_variational'] = wv
        #LP['word_variational_not_normalized'] = wv_temp
        return LP
       
    # Calculate ELBO for part related to the allocation model
    # p(z | pi) + p(pi | alpha) - q( phi | z) - q(theta | pi)
    # where phi and theta represent our variational parameters
    def calc_evidence( self, Data, SS, LP ):
        # PI refers to the parameters we use for the document x topic weights
        # Z refers to the topic indicator assignments for individual word tokens
        
        # Calculate ELBO assignments for document level assignments pi
        if 'ampG' in SS:
            pPI = SS['ampG']*self.E_log_pPI( LP)
            qPI = SS['ampG']*self.E_log_qPI(LP)
        else:
            pPI = self.E_log_pPI( Data, LP ) # evidence of 
            qPI = self.E_log_pPI( Data, LP ) # entropy of ...
        
        # Calculate ELBO for word level assignments z   
        if 'ampG' in SS:
            pZ = SS['ampG']*self.E_log_pZ( Data, LP )
            qZ = SS['ampF']*self.E_log_qZ( Data, LP )
        else:
            pZ = self.E_log_pZ( Data, LP )
            qZ = self.E_log_qZ( Data, LP )
        elbo_alloc = pPI + pZ - qPI - qZ
        
        # Debug parts of the ELBO
        debug = False
        if debug is True:
            print "pZ: " + str(pZ)
            print "qZ: " + str(qZ)
            print "pPI: " + str(pPI)
            print "qPI: " + str(qPI)
        return elbo_alloc

    #Part of the ELBO, calculates likelihood terms for word tokens z
    def E_log_pZ( self, Data, LP ):
        E_log_pZ = LP["doc_variational"] * LP["E_log_doc_variational"]
        return E_log_pZ.sum()
    
    #Part of the ELBO, calculates entropy terms for word tokens z
    def E_log_qZ( self, Data, LP):  
        temp = LP["word_variational"] * np.log(EPS+LP['word_variational'])
        E_log_qZ = np.dot(Data.WC[:,1].T, temp)
        return E_log_qZ.sum()    

    #Part of the ELBO, calculates likelihood terms for document-topic weights
    def E_log_pPI( self, Data, LP ):
        K = self.K
        D = Data.nDocs
        E_log_pPI = gammaln(K*self.alpha0)-K*gammaln(self.alpha0)    
        E_log_pPI *= D  # same prior over each Doc of data!
        for d in xrange( D ):
            E_log_pPI += (self.alpha0-1)*LP['E_log_doc_variational'][d,:].sum()
        return E_log_pPI
    
    #Part of the ELBO, calculates entropy terms for document-topic weights
    def E_log_qPI( self, Data, LP ):
        E_log_qPI = 0
        for d in xrange( Data.nDocs ):
            theta = LP['doc_variational'][d]
            E_log_qPI +=  gammaln(  theta.sum()) - gammaln(  theta ).sum() \
                  + np.inner(  theta-1,  LP['E_log_doc_variational'][d] )
        return E_log_qPI

    ##############################################################    
    def update_global_params( self, SS, rho=None, **kwargs ):
        '''Admixtures have no global allocation params! 
         Mixture weights are Doc/document specific.'''
        pass

    ############################################################## Sampling   
    def sample_from_pred_posterior( self ):
        pass
    
    #################### GET METHODS #############################
    def set_prior(self, PriorParamDict):
        self.alpha0 = PriorParamDict['alpha0']
    
    def to_dict( self ):
        return dict()              
  
    def from_dict(self, Dict):
        pass
          
    def get_prior_dict( self ):
        return dict( alpha0=self.alpha0, K=self.K, inferType=self.inferType )
    
    def get_info_string( self):
        ''' Returns human-readable name of this object'''
        return 'Finite admixture model with %d components | alpha=%.2f' % (self.K, self.alpha0)
    
    def get_model_name(self ):
        return 'admixture'

    def get_human_global_param_string(self):
        ''' Returns human-readable numerical repr. of parameters,
          for quick inspection of correctness'''
        mystr = ''
        for rowID in xrange(3):
            mystr += np2flatstr( np.exp(self.Elogw[rowID]), '%3.2f') + '\n'
        return mystr
 
    def is_nonparametric(self):
        return False 

    def need_prev_local_params(self):
        return True
