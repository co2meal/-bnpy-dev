'''
Distr.py 

Generic exponential family probability distribution object
'''
import numpy as np

class BernDistr( object ):
    @classmethod
    def InitFromData( cls, argDict, Data):
        raise NotImplementedError("TODO after Data format for topic models")
    
    def __init__( self, mu=None):
        if mu is None:
            self.mu = None
        else:
            self.mu = np.asarray(mu)
    
    def set_helpers(self):
        pass
        
    ############################################################## Param updates  
    ##############################################################
    def get_post_distr( self, SS ):
        ''' Create new Distr object with posterior params
        '''
        pass
    
    def post_update( self, SS ):
        ''' Posterior update of internal params given data'''
        
        pass
    
    def post_update_soVB( self, rho, *args ):
        ''' Stochastic online update of internal params'''
        pass
    
    ############################################################## E step cond probs.  
    ##############################################################
    def log_pdf( self, X ):
        ''' Returns log p( x | theta )'''
        return X*np.log(self.mu) + (X-1)*np.log(1-self.mu)
    
    def E_log_pdf( self, X, prior ):
        ''' Returns E[ log p( x | theta ) ] under q(theta) <- this distr'''
        return X * prior.ElogA + (X-1) * prior.ElogB
    
    ############################################################## Exp Fam Accessors  ##############################################################
    def get_log_norm_const(self):
        ''' Returns log( Z ), where PDF(x) :=  1/Z(theta) f( x | theta )'''
        pass
    
    def get_entropy( self ):
        ''' Returns entropy of this distribution 
        H[ p(x) ] = -1*\int p(x|theta) log p(x|theta) dx '''
    pass

    
    ############################################################## I/O  ##############################################################
    def to_dict(self):
        pass
    
    def from_dict(self, pDict):
        pass