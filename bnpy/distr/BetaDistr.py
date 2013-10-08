'''
Distr.py 

Generic exponential family probability distribution object
'''
import numpy as np
from scipy.special import digamma, gammaln

class BetaDistr( object ):
    @classmethod
    
    def InitFromData( cls, argDict, Data):
        ha = argDict['a']
        hb = argDict['b']
        
        return cls( a = ha, b = hb)
    
    def __init__( self, a=None, b=None):
        if a is None:
            self.a = None
            self.b = None
        else:
            self.a = np.array(a) #set hyper-parameters for heads
            self.b = np.array(b) #set hyper-parameters for tails
            self.set_helpers()
    
    def set_helpers(self):
        #Calculate E[x] for beta
        self.mean = self.a / (self.a + self.b)
        self.lambda_a = self.a
        self.lambda_b = self.b
        self.ElogW1 = digamma(self.a) - digamma(self.a + self.b)
        self.ElogW2 = digamma(self.b) - digamma(self.a + self.b)  
        
    ############################################################## Param updates  
    ##############################################################
    def get_post_distr( self, SS ):
        ''' Create new Distr object with posterior params'''
        a = self.a + SS.oa #heads
        b = self.b + SS.ob #tails
         
        return BetaDistr(a, b)
    
    def post_update( self, SS ):
        ''' Posterior update of internal params given data'''
        self.a += SS.Nheads
        self.b += SS.Ntails 
    
    def post_update_soVB( self, rho, *args ):
        ''' Stochastic online update of internal params'''
        pass
    
    ############################################################## E step cond probs.  
    ##############################################################
    def log_pdf( self ):
        pass
    
    def E_log_pdf( self ):
        ''' Returns E[ log p( x | theta ) ] under q(theta) <- this distr'''
        ''' ha and hb are the hyperparameters that are assumed fixed for beta '''
        ha = self.ha
        hb = self.hb
        return gammaln(ha + hb) - gammaln(ha) - gammaln(hb) + (ha-1)*np.log(self.ElogA) + (hb-1)*np.log(self.ElogB) 
    
    ############################################################## Exp Fam Accessors  ##############################################################
    def get_log_norm_const(self):
        ''' Returns log( Z ), where PDF(x) :=  1/Z(theta) f( x | theta )'''
        return gammaln(self.a + self.b) - gammaln(self.a) - gammaln(self.b)
    
    def get_entropy( self ):
        ''' Returns entropy of this distribution '''
        return self.get_log_norm_const() + (self.a-1)*np.log(self.ElogA) + (self.b-1)*np.log(self.ElogB) 
    
    ############################################################## I/O  ##############################################################
    def to_dict(self):
        return dict( a=self.a, b=self.b, ElogA=self.ElogA, ElogB = self.ElogB )
    
    def from_dict(self, pDict):
        pass