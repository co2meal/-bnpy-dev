'''
DirichletDistr.py

Dirichlet distribution in D-dimensions
    
Attributes
-------
lamvec  :  Dx1 vector of non-negative numbers
                lamvec[d] >= 0 for all d
  
'''
import numpy as np
import scipy.linalg
from scipy.special import digamma, gammaln

class DirichletDistr(object):
    @classmethod
    def InitFromData(cls, argDict, Data):
        # Data should contain information about dirichlet vocabulary size
        if argDict["lambda"] is not None:
            lamda = argDict["lambda"]
        else:
            lamda = 1.0
        lamvec = lamda * np.ones(Data.vocab_size) 
        return cls(lamvec = lamvec)
      
    def __init__(self, lamvec=None, **kwargs):
        self.lamvec = lamvec
        if lamvec is not None:
            self.D = lamvec.size
            self.set_helpers()

    def set_helpers(self):
        assert self.lamvec.ndim == 1
        self.lamsum = self.lamvec.sum()
        digammalamvec = digamma(self.lamvec)
        digammalamsum = digamma(self.lamsum)
        self.Elogphi   = digammalamvec - digammalamsum
        

    ############################################################## Param updates  
    def get_post_distr(self, SS):
        ''' Create new Distr object with posterior params'''
        return DirichletDistr(SS.WordCounts + self.lamvec)
    
    def post_update( self, SS ):
        ''' Posterior update of internal params given data'''
        self.lamvec += SS.WordCounts
        self.set_helpers()

    def post_update_soVB( self, rho, starD):
        ''' Stochastic online update of internal params'''
        self.lamvec = rho * starD.lamvec + (1.0 - rho) * self.lamvec
        self.set_helpers()

  ############################################################## Norm Constants  
  ##############################################################
    @classmethod
    def calc_log_norm_const(cls, lamvec):
        return gammaln(lamvec) - np.sum(gammaln(lamvec))
  
    def get_log_norm_const(self):
        ''' Returns log( Z ), where PDF(x) :=  1/Z(theta) f( x | theta )'''
        return np.sum(gammaln(self.lamvec)) - gammaln(self.lamsum)

    def get_entropy( self ):
        ''' Returns entropy of this distribution 
          H[ p(x) ] = -1*\int p(x|theta) log p(x|theta) dx'''
        H = self.get_log_norm_const()
        H -= np.inner(self.lamvec - 1., self.Elogphi)
        return H
    
    ############################################################## Conditional Probs.  
    ##############################################################    
    def E_log_pdf(self, Data):
        ''' Returns E[ log p( x | theta ) ] under q(theta) <- this distr'''
        raise NotImplementedError("TODO")    

    ############################################################## I/O  
    ##############################################################
    def to_dict(self):
        return dict(name=self.__class__.__name__, lamvec=self.lamvec)
    
    def from_dict(self, PDict):
        self.lamvec = np.squeeze(PDict['lamvec'])
        self.set_helpers()
