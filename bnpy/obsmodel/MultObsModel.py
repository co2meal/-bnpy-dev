'''
'''
import numpy as np

from ..distr import DirichletDistr
from ..distr import MultinomialDistr
from ..util import np2flatstr, EPS

from ObsCompSet import ObsCompSet

class MultObsModel( ObsCompSet ):
    
    def __init__( self, inferType, W=None, obsPrior=None,):
        self.inferType = inferType
        self.obsPrior = obsPrior
        self.W = None #vocabulary size
        self.comp = list()

    @classmethod
    def InitFromData(cls, inferType, priorArgDict, Data):
        nDocTotal = Data.nDocTotal
        # Something else probably needs to be done for EM here
        if inferType == 'EM':
            # Defines the Dirichlet (topic x word) that is initialized to ensure it fits data dimensions
            obsPrior = DirichletDistr.InitFromData(priorArgDict, Data)
        else:
            obsPrior = DirichletDistr.InitFromData(priorArgDict, Data)
        return cls(inferType, nDocTotal, obsPrior)

    def get_human_global_param_string(self, fmtStr='%3.2f'):
        if self.inferType == 'EM':
            return '\n'.join( [np2flatstr(self.obsPrior[k].phi, fmtStr) for k in xrange(self.K)] )
        else:
            return '\n'.join( [np2flatstr(self.obsPrior[k].lamvec/self.obsPrior[k].lamsum, fmtStr) for k in xrange(self.K)] )

    # Returns the sufficient statistics for the global topic x word variable lambda
    def get_global_suff_stats( self, Data, SS, LP ):
        # Grab topic x word sufficient statistics
        wv = LP['word_variational']
        nObs_mb, K = wv.shape
        word_count = Data.word_count
        word_id = Data.word_id
        lambda_kw = np.zeros( (K, Data.vocab_size) )
        
        # Loop through word tokens
        for ii in xrange( nObs_mb ):
            lambda_kw[:, word_id[ii]] += wv[ii,:] * word_count[ii]  
            
        # Return K x V matrix of sufficient stats (topic x word)
        SS.lambda_kw = lambda_kw
        self.K = K
        return SS

    def update_obs_params_EM( self, SS, **kwargs):
        phiHat = SS['TermCount']
        phiHat = phiHat/( EPS+ phiHat.sum(axis=1)[:,np.newaxis] )
        for k in xrange( self.K ):
            self.obsPrior[k] = MultinomialDistr( phiHat[k] )

    def update_obs_params_VB( self, SS, Krange, **kwargs):
        for k in Krange:
            self.comp[k] = self.obsPrior.get_post_distr( SS, k )

    def update_obs_params_soVB( self, SS, rho, Krange, **kwargs):
        # grab Dirichlet posterior for lambda and perform stochastic update
        for k in Krange:
            Dstar = self.obsPrior.get_post_distr(SS, k)
            self.comp[k].post_update_soVB(rho, Dstar)
      
    # Calculate at the word level the expectations associated with our observation model
    # For document d, word w, topic k, our local variational update is:
    #     phi_dwk \propto exp( E[log pi_ik ] + E[ log lambda_kw ] )
    # Calc_local params creates a nObsTotal x 1 array with values assocated with lambda_kw 
    def calc_local_params( self, Data, LP):
        if self.inferType == 'EM':
            LP['E_log_obs_word'] = self.log_obs_word( Data )
        else:
            LP['E_log_obs_word'] = self.E_log_obs_word( Data )
        return LP

    def log_obs_word( self, Data ):
        lpr = np.empty( (Data['nObsTotal'], self.K) )
        for k in xrange( self.K ):
            lpr[:,k] = self.obsPrior[k].log_pdf( Data )
        return lpr
    
    # Returns a nObsTotal x 1 array of expectations associated with the observation model  
    def E_log_obs_word( self, Data ):
        E_log_obs_word = np.empty( (Data.nObs, self.K) )
        lambda_kw = np.zeros((self.K, Data.vocab_size))
        word_id = Data.word_id
        # Since priors are stored in comp, recreate as matrix for easier indexing
        for k in xrange(self.K):
            lambda_kw[k,:] = self.comp[k].Elogphi # returns topic by word matrix expectations
        
        # Return a nObsTotal x 1 array of expected[lambda_kw] relevant for word_id = w
        for ii in xrange( Data.nObs ):
            E_log_obs_word[ii,:] = lambda_kw[:, word_id[ii]]
        return E_log_obs_word
  
  #########################################################  Evidence Bound Fcns  
    def calc_evidence( self, Data, SS, LP):
        if self.inferType == 'EM':
            return 0 # handled by alloc model
        # Calculate p(w | z, lambda) + p(lambda) - q(lambda)
        elbo_pWords = self.E_log_pW( Data, LP, SS) 
        elbo_pLambda = self.E_log_pLambda()
        elbo_qLambda = self.E_log_qLambda()
        lb_obs = elbo_pWords + elbo_pLambda - elbo_qLambda
        
        # Print parts of the ELBO for debugging
        debug = False
        if debug is True:
            print "pW: " + str(elbo_pWords)
            print "pL: " + str(elbo_pLambda)
            print "qL: " + str(elbo_qLambda)
        return lb_obs
  
    def E_log_pW( self, Data, LP, SS ):
        ''' E_{q(Z), q(Phi)} [ log p(X) ]'''
        elambda_kw = np.zeros( (self.K, Data.vocab_size) )
        for k in xrange( self.K ):
            elambda_kw[k,:] = self.comp[k].Elogphi
        # Calculate p(w | lambda, z )
        lpw = 0
        
        word_count = Data.word_count
        word_id = Data.word_id
        for ii in xrange( Data.nObs ):
            lpw += word_count[ii] * np.dot(LP["word_variational"][ii,:] ,elambda_kw[:, word_id[ii]])    
        return lpw
    
    def E_log_pLambda( self ):
        lp = self.obsPrior.get_log_norm_const()*np.ones( self.K)
        for k in xrange( self.K):
            lp[k] += np.sum( (self.obsPrior.lamvec - 1)*self.comp[k].Elogphi )
        return lp.sum()
  
    def E_log_qLambda( self ):
        ''' Return negative entropy!'''    
        lp = np.zeros( self.K)
        for k in xrange( self.K):
            lp[k] = self.comp[k].get_entropy()
        return -1*lp.sum()

    def get_prior_dict( self ):
        PDict = self.obsPrior.to_dict()
        return PDict
############### GET METHODS ##################
    def get_info_string(self):
        return 'Multinomial distribution'
  
    def get_info_string_prior(self):
        if self.obsPrior is None:
            return 'None'
        else:
            return 'Dirichlet'
        
'''
    def set_obs_dims( self, Data):
        self.D = Data['nVocab']
        if self.obsPrior is not None:
            self.obsPrior.set_dims( self.D )
            
    def save_params( self, filename):
        pass
            
'''

