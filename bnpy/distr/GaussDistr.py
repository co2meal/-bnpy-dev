'''
GaussianDistr.py 

Gaussian probability distribution

Attributes
-------
m : D-dim vector, mean
S : DxD matrix, covariance matrix
'''
import numpy as np
import scipy.linalg
from ..util import dotABT, MVgammaln, MVdigamma, gammaln, digamma
from ..util import LOGTWOPI, EPS
from .Distr import Distr

class GaussDistr( Distr ):
      
  ######################################################### Constructor  
  #########################################################
  def __init__(self, m=None, L=None, Sigma=None):
    self.m = np.asarray( m )  
    if L is not None:
      self.L = np.asarray(L)
      self.doSigma = 0
    else:
      self.Sigma = np.asarray(Sigma)
      self.doSigma = 1
    self.D = self.m.size
    self.Cache = dict()

  ######################################################### Log Cond. Prob.  
  #########################################################   E-step
  def log_pdf(self, Data):
    ''' Calculate log soft evidence for all data items under this distrib
        
        Returns
        -------
        logp : Data.nObs x 1 vector, where
                logp[n] = log p( Data[n] | self's mean and prec matrix )
    '''
    return -1*self.get_log_norm_const() - 0.5*self.dist_mahalanobis(Data.X)
  
  def dist_mahalanobis(self, X):
    '''  Given NxD matrix X, compute  Nx1 vector Dist
            Dist[n] = ( X[n]-m )' L (X[n]-m)
    '''
    if self.doSigma:
      Q = np.linalg.solve(self.cholSigma(), (X-self.m).T)
    else:
      Q = dotABT(self.cholL(), X-self.m)
    Q *= Q
    return np.sum(Q, axis=0)
    
  ######################################################### Global updates
  #########################################################   M-step
  ''' None required. M-step handled by GaussObsModel.py
  '''

  ######################################################### Basic properties
  ######################################################### 
  @classmethod
  def calc_log_norm_const( self, logdetL, D):
    return 0.5 * D * LOGTWOPI - 0.5 * logdetL

  def get_log_norm_const( self ):
    ''' Returns log( Z ), where
         PDF(x) :=  1/Z(theta) f( x | theta )
    '''
    if self.doSigma:
      return 0.5*self.D*LOGTWOPI + 0.5*self.logdetSigma()
    else:
      return 0.5*self.D*LOGTWOPI - 0.5*self.logdetL()
        
  ######################################################### Accessors  
  #########################################################
  def get_covar(self):
    if self.doSigma:
      return self.Sigma
    try:
      return self.Cache['invL']
    except KeyError:
      self.Cache['invL'] = np.linalg.inv( self.L )
      return self.Cache['invL']
      
  def cholSigma(self):
    print self.L
    try:
      return self.Cache['cholSigma']
    except KeyError:
      self.Cache['cholSigma'] = scipy.linalg.cholesky(self.Sigma, lower=1)
      return self.Cache['cholSigma']
      
  def logdetSigma(self):
    try:
      return self.Cache['logdetSigma']
    except KeyError:
      self.Cache['logdetSigma'] = 2.0*np.sum(np.log(np.diag(self.cholSigma())))
    return self.Cache['logdetSigma']
     
  def cholL(self):
    try:
      return self.Cache['cholL']
    except KeyError:
      self.Cache['cholL'] = scipy.linalg.cholesky(self.L, lower=0)
    return self.Cache['cholL']
          
  def logdetL(self):
    try:
      return self.Cache['logdetL']
    except KeyError:
      self.Cache['logdetL'] = 2.0*np.sum(np.log(np.diag(self.cholL())))
    return self.Cache['logdetL']



  ######################################################### I/O Utils 
  #########################################################
  def to_dict(self):
    if self.doSigma:
      return dict(m=self.m, Sigma=self.Sigma, name=self.__class__.__name__ )
    else:
      return dict(m=self.m, L=self.L, name=self.__class__.__name__ )

  def from_dict(self, Dict):
    if 'L' in Dict:
      self.L = Dict['L']
      self.doSigma = False
    elif 'Sigma' in Dict:
      self.Sigma = Dict['Sigma']
      self.doSigma = True
    self.m = Dict['m']
    self.D = self.L.shape[0]
    self.Cache = dict()
