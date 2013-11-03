''' 
GaussGammaDistr.py 

Joint Gaussian-Gamma distribution: D independent Gaussian-Gamma distributions

Attributes
--------
m : mean for Gaussian, length D
beta : scalar precision parameter for Gaussian covariance
a : parameter for Gamma, scalar value
b : parameter for Gamma, vector length D
'''
import numpy as np
import scipy.linalg

from bnpy.util import MVgammaln, MVdigamma
from bnpy.util import LOGTWO, LOGPI, LOGTWOPI, EPS
from bnpy.util import gammaln, digamma

from .Distr import Distr

class GaussGammaDistr( Distr ):
  @classmethod
  def InitFromData( cls, argDict, Data):
    ''' Constructor for GaussGamma distr object 
        ensures that it matches dimension with Data
    '''
    D = Data.dim
    m0 = 0 if 'm0' not in argDict else argDict['m0']
    beta = 1 if 'beta' not in argDict else argDict['beta']
    a = 1 if 'a' not in argDict else argDict['a']
    b0 = 0.9 if 'b0' not in argDict else argDict['b0']

    m = np.ones(D)*m0
    b = np.ones(D)*b0

    return cls(m=m, beta=beta, a=a, b=b)
    
  def __init__(self, m=None, beta=None, a=None, b=None, **kwargs):
    print m
    self.D = b.shape[0]
    self.m = m
    self.beta = beta
    self.a = a
    self.b = b
    self.Cache = dict()
    
    
  ######################################################### Param updates 
  ######################################################### (M step)
  def get_post_distr( self, compSS):
    ''' Create new Distr object with posterior params
        See Bishop equations 10.59 - 10.63 (modified for Gaussian-Gamma)
    '''
    EN = float(compSS.N)
    Ex = compSS.x
    Exx = compSS.xx
    beta = self.beta + EN
    m = ( self.beta*self.m + Ex ) / beta
    a = self.a + 0.5*EN
    b = self.b + 0.5*(Exx + self.beta*np.square(self.m) - beta*np.square(m))
    return GaussGammaDistr(m, beta, a, b)
    
  ############################################################## Local params (E step)
  ##############################################################
  def E_log_pdf( self, Data ):
    ''' Returns E[ log p( x | theta ) ] under q(theta) <- this distr
    '''
    logp = 0.5*self.ElogdetLam() \
          -0.5*self.D*LOGTWOPI\
          -0.5*self.E_weightedSOS(Data.X)
    return logp
    
  def E_weightedSOS(self, X):
    '''Calculate d(X)[n] = E[(x_n - m_k)T * lambda * (x_n -mk)]
    '''           
    dX = (X-self.m) # NxD
    weighted_SOS = np.sum(np.square(dX)/self.b, axis=1)*self.a
    weighted_SOS += self.D/self.beta
    return weighted_SOS
    
  def ElogdetLam(self):
    try:
      return self.Cache['ElogdetLam']
    except KeyError:
      self.Cache['ElogdetLam'] = (self.D*digamma(self.a) - np.sum(np.log(self.b)))
      return self.Cache['ElogdetLam']
      
  def entropyGamma(self):
    '''Calculate entropy of this Gamma distribution,
         as defined in Bishop PRML B.31
    '''
    return self.D*gammaln(self.a) \
           - self.D*(self.a-1)*digamma(self.a) + self.D*self.a \
           - np.sum(np.log(self.b))
           
  ############################################################## I/O 
  ##############################################################
  def to_dict(self):
    return dict(name=self.__class__.__name__, \
                 m=self.m, beta=self.beta, a=self.a, b=self.b)
    
  def from_dict(self, Dict):
    self.m = Dict['m']
    self.a = Dict['a']
    self.b = Dict['b']
    self.beta = Dict['beta']
    self.D = self.b.shape[0]
    self.Cache = dict()

  def to_string(self):
    np.set_printoptions( precision=3, suppress=True)
    msg = 'm = ' + str(self.m[:2] ) +'\n'
    msg += 'beta =' +str(self.beta) + '\n'
    msg += 'a =' +str(self.a) + '\n'
    msg += 'b =' +str(self.b[:2]) + '\n'
    if self.D > 2:
      msg += '...'
    return msg
