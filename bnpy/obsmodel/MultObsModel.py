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
        nDocs = Data.nDocs
        if inferType == 'VB':
            # Defines the Dirichlet (topic x word) that is initialized to ensure it fits data dimensions
            obsPrior = DirichletDistr.InitFromData(priorArgDict, Data)
        return cls(inferType, nDocs, obsPrior)

    def get_human_global_param_string(self, fmtStr='%3.2f'):
        if self.inferType == 'EM':
            return '\n'.join( [np2flatstr(self.obsPrior[k].phi, fmtStr) for k in xrange(self.K)] )
        else:
            return '\n'.join( [np2flatstr(self.obsPrior[k].lamvec/self.obsPrior[k].lamsum, fmtStr) for k in xrange(self.K)] )

    # Returns the sufficient statistics for the global topic x word variable lambda
    def get_global_suff_stats( self, Data, SS, LP ):
        # Grab topic x word sufficient statistics
        wv = LP['word_variational']
        _, K = wv.shape
        WC = Data.WC
        lambda_kw = np.zeros( (K, Data.nWords) )
        
        # Loop through documents
        for ii in xrange( Data.nObsTotal ):
            word_ind = WC[ii,0] 
            lambda_kw[:, word_ind] += wv[ii,:] * WC[ii, 1]  
        # Return K x V matrix of sufficient stats (topic x word)
        
        #lambda_kw = Data.true_tw
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

    def update_obs_params_VB_stochastic( self, SS, rho, Ntotal, **kwargs):
        ampF = Ntotal/SS['Ntotal']
        for k in xrange( self.K):
            postDistr = self.obsPrior.getPosteriorDistr( ampF*SS['TermCount'][k] )
            if self.obsPrior[k] is None:
                self.obsPrior[k] = postDistr
            else:
                self.obsPrior[k].rho_update( rho, postDistr )
      
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
        WC = Data.WC
        E_log_obs_word = np.empty( (Data.nObsTotal, self.K) )
        lambda_kw = np.zeros((self.K, Data.nWords))
        
        # Since priors are stored in comp, recreate as matrix for easier indexing
        for k in xrange(self.K):
            lambda_kw[k,:] = self.comp[k].Elogphi # returns topic by word matrix expectations
        
        # Return a nObsTotal x 1 array of expected[lambda_kw] relevant for word_id = w
        for ii in xrange( Data.nObsTotal ):
            E_log_obs_word[ii,:] = lambda_kw[:,WC[ii,0]]
        return E_log_obs_word
  
  #########################################################  Evidence Bound Fcns  
    def calc_evidence( self, Data, SS, LP):
        if self.inferType == 'EM':
            return 0 # handled by alloc model
        # Calculate p(w | z, lambda) + p(lambda) - q(lambda)
        pW = self.E_logpX( Data, LP, SS) 
        pL = self.E_log_pLambda()
        qL = self.E_log_qLambda()
        lb_obs = pW + pL - qL
        
        # Print parts of the ELBO for debugging
        debug = False
        if debug is True:
            print "pW: " + str(pW)
            print "pL: " + str(pL)
            print "qL: " + str(qL)
        return lb_obs
  
    def E_logpX( self, Data, LP, SS ):
        ''' E_{q(Z), q(Phi)} [ log p(X) ]'''
        elambda_kw = np.zeros( (self.K, Data.nWords) )
        for k in xrange( self.K ):
            elambda_kw[k,:] = self.comp[k].Elogphi
        # Calculate p(w | lambda, z )
        lpw = 0
        for ii in xrange( Data.nObsTotal ):
            lpw += Data.WC[ii,1] * np.dot(LP["word_variational"][ii,:] ,elambda_kw[:,Data.WC[ii,0]])    
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

