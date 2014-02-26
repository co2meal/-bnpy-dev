''' 
GaussGammaDistr.py 

Joint Gaussian-Gamma distribution: D independent Gaussian-Gamma distributions

Attributes
--------
m : mean for Gaussian, length D
kappa : scalar precision parameter for Gaussian covariance

a : parameter for Gamma, vector length D
b : parameter for Gamma, vector length D
'''
import numpy as np
import scipy.linalg

from bnpy.util import MVgammaln, MVdigamma
from bnpy.util import LOGTWO, LOGPI, LOGTWOPI, EPS
from bnpy.util import gammaln, digamma

from .Distr import Distr

class GaussGammaDistr( Distr ):

  ######################################################### Constructor  
  #########################################################

  def __init__(self, m=None, kappa=None, a=None, b=None, **kwargs):
    # UNPACK
    self.m = np.squeeze(m)
    self.kappa = float(np.squeeze(kappa))
    self.a = np.squeeze(np.asarray(a))
    self.b = np.squeeze(np.asarray(b))
    # Dimension
    assert self.b.ndim <= 1
    self.D = self.b.size
    self.Cache = dict()
    
  @classmethod
  def CreateAsPrior( cls, argDict, Data):
    ''' Creates Gaussian-Gamma prior for params that generate Data.
        Returns GaussGammaDistr object with same dimension as Data.
        Provided argDict specifies prior's expected mean and variance.
    '''
    D = Data.dim
    m0 = argDict['m0']
    kappa = argDict['kappa']
    a0 = argDict['a0']
    b0 = argDict['b0']
    m = m0 * np.ones(D)
    a = a0 * np.ones(D)
    b = b0 * np.ones(D)
    return cls(m=m, kappa=kappa, a=a, b=b)
    

  ######################################################### Log Cond. Prob.  
  #########################################################   E-step

  def E_log_pdf( self, Data ):
    ''' Calculate E[ log p( x_n | theta ) ] under q(theta) <- this distr
    '''
    logPDFConst = -0.5 * self.D * LOGTWOPI + 0.5 * np.sum(self.E_logLam())
    logPDFData = -0.5 * self.E_distMahalanobis(Data.X)
    return logPDFConst + logPDFData
    """
    logp = 0.5*self.ElogdetLam() \
          -0.5*self.D*LOGTWOPI\
          -0.5*self.E_weightedSOS(Data.X)
    return logp
    """    

  def E_distMahalanobis(self, X):
    ''' Calculate E[ (x_n - \mu)^T diag(\lambda) (x_n - mu) ]
          which has simple form due to diagonal structure.

        Args
        -------
        X : nObs x D matrix

        Returns
        -------
        dist : nObs-length vector, entry n = distance(X[n])
    '''
    Elambda = self.a / self.b
    if X.ndim == 2:
      weighted_SOS = np.sum( Elambda * np.square(X - self.m), axis=1)
    else:
      weighted_SOS = np.sum(Elambda * np.square(X - self.m))
    weighted_SOS += self.D/self.kappa
    return weighted_SOS

  def E_weightedSOS(self, X):
    '''Calculate d(X)[n] = E[(x_n - m_k)T * lambda * (x_n -mk)]
    '''           
    dX = (X-self.m) # NxD
    weighted_SOS = np.sum(np.square(dX)/self.b, axis=1)*self.a
    weighted_SOS += self.D/self.kappa
    return weighted_SOS

  ######################################################### Param updates 
  ######################################################### (M step)
  def get_post_distr( self, SS, k=None, kB=None, **kwargs):
    ''' Create new Distr object with posterior params
        See Bishop equations 10.59 - 10.63 (modified for Gaussian-Gamma)
    '''
    if k is None:
      EN = SS.N
      Ex = SS.x
      Exx = SS.xx
    elif kB is not None:
      EN = float(SS.N[k] + SS.N[kB])
      Ex = SS.x[k] + SS.x[kB]
      Exx = SS.xx[k] + SS.xx[kB]
    else:
      EN = float(SS.N[k])
      Ex = SS.x[k]
      Exx = SS.xx[k]
    kappa = self.kappa + EN
    m = (self.kappa * self.m + Ex) / kappa
    a = self.a + 0.5*EN
    b = self.b + 0.5*(Exx + self.kappa*np.square(self.m) - kappa*np.square(m))
    return GaussGammaDistr(m, kappa, a, b)
     

  ######################################################### Basic properties
  ######################################################### 
  @classmethod
  def calc_log_norm_const(cls, a, b, m, kappa):
    logNormConstNormal = 0.5 * D * (LOGTWOPI + np.log(kappa))
    logNormConstGamma  = np.sum(gammaln(a)) - np.inner(a, np.log(b))
    return logNormConstNormal + logNormConstGamma
  
  def get_log_norm_const(self):
    ''' p(mu,Lam) = NormalGamma( . | self)
                   = 1/Z f(mu|Lam) g(Lam), where Z is const w.r.t mu,Lam
        This function returns 
            log( Z )= log \int f() g() d mu d Lam
    '''
    D = self.D
    a = self.a
    b = self.b
    logNormConstNormal = 0.5 * D * (LOGTWOPI - np.log(self.kappa))
    logNormConstGamma  = np.sum(gammaln(a)) - np.inner(a, np.log(b))
    return logNormConstNormal + logNormConstGamma
    
  def E_log_pdf_Phi(self, Distr, doNormConst=True):
    ''' Evaluate expectation of log PDF for companion distribution
    '''
    assert Distr.D == self.D
    selfELam = self.a / self.b
    logPDF = np.inner(Distr.a - 0.5, self.E_logLam()) \
                - np.inner(Distr.b, selfELam) \
                - 0.5 * Distr.kappa * self.E_distMahalanobis(Distr.m)
    if doNormConst:
      return logPDF - Distr.get_log_norm_const()
    return logPDF

  def get_entropy(self):
    ''' Calculate entropy of this Gauss-Gamma disribution,
    '''
    entropyGamma = self.entropyGamma()
    

  def entropyGamma(self):
    '''Calculate entropy of this Gamma distribution,
         as defined in Bishop PRML B.31
    '''
    a = self.a
    b = self.b
    return np.sum(gammaln(a)) - np.inner(a-1.0, digamma(a)) \
                  + np.sum(a) - np.sum(np.log(b))
  """
    return self.D*gammaln(self.a) \
           - self.D*(self.a-1)*digamma(self.a) + self.D*self.a \
           - np.sum(np.log(self.b))
  """        
  ######################################################### Accessors
  #########################################################
  def E_logLam(self):
    ''' E[ \log \lambda_d ]
        Returns vector, length D
    '''
    return digamma(self.a) - np.log(self.b)

  def E_sumlogLam(self):
    ''' \sum_d E[ \log \lambda_d ]
        Returns scalar
    '''
    return np.sum(digamma(self.a) - np.log(self.b))

  def E_Lam(self):
    ''' E[ \lambda_d * \mu_d ]
        Returns vector, length D
    '''
    return (self.a / self.b)

  def E_LamMu(self):
    ''' E[ \lambda_d * \mu_d ]
        Returns vector, length D
    '''
    return (self.a / self.b) * self.m

  def E_LamMu2(self):
    ''' E[ \lambda_d * \mu_d * \mu_d ]
        Returns vector, length D
    '''
    return (self.a / self.b) * np.square(self.m) + 1./self.kappa

  """
  def ElogdetLam(self):
    try:
      return self.Cache['ElogdetLam']
    except KeyError:
      self.Cache['ElogdetLam'] = self.D*digamma(self.a) \
                                 - np.sum(np.log(self.b))
      return self.Cache['ElogdetLam']
  """

  ############################################################## I/O 
  ##############################################################
  def to_dict(self):
    return dict(name=self.__class__.__name__, \
                 m=self.m, kappa=self.kappa, a=self.a, b=self.b)
    
  def from_dict(self, Dict):
    self.m = Dict['m']
    self.a = Dict['a']
    self.b = Dict['b']
    self.kappa = Dict['kappa']
    self.D = self.b.shape[0]
    self.Cache = dict()

  def to_string(self, offset="  "):
    Estddev = self.a / self.b[:2]
    np.set_printoptions(precision=3, suppress=False)
    msg  = offset + 'Expected Mean    ' + str(self.m[:2] ) + '\n'
    msg += offset + 'Expected Std Dev ' + str(Estddev) + '\n'
    if self.D > 2:
      msg += '...'
    return msg
