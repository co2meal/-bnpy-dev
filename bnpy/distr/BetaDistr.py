'''
BetaDistr.py

Beta distribution in 1-dimension x ~ Beta(a,b)
    
Attributes
-------
lamvec  :  2x1 vector of non-negative numbers
                lamvec[d] >= 0 for all d
  
'''
import numpy as np
import scipy.linalg
from scipy.special import digamma, gammaln

class BetaDistr(object):
    @classmethod
    def InitFromData(cls, argDict, Data):
        # Data should contain information about Beta vocabulary size
        if argDict["lamA"] and argDict["lamB"] is not None:
            lamA = argDict["lamA"]
            lamB = argDict["lamB"]
        else:
            lamA = 0.1
            lamB = 0.1
        return cls(lamA = lamA, lamB = lamB)
      
    def __init__(self, lamA=None, lamB=None, **kwargs):
        self.lamA = lamA
        self.lamB = lamB
        if lamA and lamB is not None:
            self.set_helpers(**kwargs)

    def set_helpers(self, doNormConstOnly=False, **kwargs):
        assert self.lamvec.ndim == 1
        self.lamsum = self.lamvec.sum()
        if hasattr(self, '_logNormC'):
          del self._logNormC
        if not doNormConstOnly:
          digammalamvec = digamma(self.lamvec)
          digammalamsum = digamma(self.lamsum)
          self.Elogphi   = digammalamvec - digammalamsum
        

    ############################################################## Param updates  
    def get_post_distr(self, SS, k=None, kB=None, **kwargs):
        ''' Create new Distr object with posterior params'''
        if kB is not None:
          return BetaDistr(SS.HeadCounts + self.lamA, SS.TailCounts + self.lamB **kwargs)
        else:
          return BetaDistr(SS.HeadCounts + self.lamA, **kwargs)

    def post_update_soVB( self, rho, starD):
        ''' Stochastic online update of internal params'''
        self.lamvec = rho * starD.lamvec + (1.0 - rho) * self.lamvec
        self.set_helpers()

  ######################################################### Norm Constants  
  #########################################################
    @classmethod
    def calc_log_norm_const(cls, lamvec):
        return gammaln(lamvec) - np.sum(gammaln(lamvec))
  
    def get_log_norm_const(self):
        ''' Returns log( Z ), where PDF(x) :=  1/Z(theta) f( x | theta )'''
        if hasattr(self, '_logNormC'):
          return self._logNormC
        self._logNormC = np.sum(gammaln(self.lamvec)) - gammaln(self.lamsum)
        return self._logNormC 

    def get_entropy( self ):
        ''' Returns entropy of this distribution 
          H[ p(x) ] = -1*\int p(x|theta) log p(x|theta) dx
        '''
        H = self.get_log_norm_const()
        H -= np.inner(self.lamvec - 1., self.Elogphi)
        return H

    ####################################################### I/O  
    #######################################################
    def to_dict(self):
        return dict(name=self.__class__.__name__, lamvec=self.lamvec)
    
    def from_dict(self, PDict):
        self.lamvec = np.squeeze(PDict['lamvec'])
        self.set_helpers()
