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
        self.lamsum = self.lamA + self.lamB
        if hasattr(self, '_logNormC'):
          del self._logNormC
        if not doNormConstOnly:
          digammalamA = digamma(self.lamA)
          digammalamB = digamma(self.lamB)
          digammalamsum = digamma(self.lamA + self.lamB)
          self.ElogWA   = digammalamA - digammalamsum
          self.ElogWB   = digammalamB - digammalamsum

    ############################################################## Param updates  
    def get_post_distr(self, SS, k=None, kB=None, **kwargs):
        ''' Create new Distr object with posterior params'''
        if kB is not None:
          return BetaDistr(SS.HeadCounts + self.lamA, SS.TailCounts + self.lamB **kwargs)
        else:
          return BetaDistr(SS.HeadCounts + self.lamA, **kwargs)

    def post_update_soVB( self, rho, starD):
        ''' Stochastic online update of internal params'''
        self.lamA = rho * starD.lamA + (1.0 - rho) * self.lamA
        self.lamB = rho * starD.lamB + (1.0 - rho) * self.lamB
        self.set_helpers()

    ####################################################### I/O  
    #######################################################
    def to_dict(self):
        return dict(name=self.__class__.__name__, lamA=self.lamA, lamB=self.lamB)
    
    def from_dict(self, PDict):
        self.lamA = PDict['lamA']
        self.lamB = PDict['lamB']
        self.set_helpers()
