''' 
GaussWishDistr.py 

Joint Gaussian-Wishart distribution

Attributes
--------
dF : scalar degrees of freedom for Wishart
invW : scale matrix for Wishart, size D x D
m : mean vector for Gaussian, length D
kappa : scalar precision parameter for Gaussian covariance
'''
import numpy as np
import scipy.linalg

from bnpy.util import MVgammaln, MVdigamma
from bnpy.util import LOGTWO, LOGPI, LOGTWOPI, EPS
from bnpy.util import gammaln, digamma

from WishartDistr  import WishartDistr
from .Distr import Distr

class GaussWishDistr( Distr ):
  @classmethod
  def InitFromData( cls, argDict, Data):
    ''' Constructor for GaussWish distr object 
        ensures that it matches dimension with Data
        and that it has the expected covar matrix specified by argDict
    '''
    D = Data.dim
    m = np.zeros(D)
    dF = np.maximum( argDict['dF'], D+2)
    sF = argDict['sF']
    kappa = argDict['kappa']
    if argDict['smatname'] == 'eye':
      Sigma = sF * np.eye(D)
    elif argDict['smatname'] == 'covdata':
      Sigma = sF * np.cov(Data.X.T, bias=1)
    elif argDict['smatname'] == 'fromtruelabels':
      ''' Set Cov. Matrix by empirical Bayes
            average the within-class sample covariances
      '''
      Z = Data.TrueLabels
      Zvals = np.unique(Z)
      Kmax = len(Zvals)
      CovHat = np.zeros((Kmax,D,D))
      wHat = np.zeros(Kmax)
      for kID,kk in enumerate(Zvals):
        wHat[kID] = np.sum(Z==kk)
        CovHat[kID] = np.cov(Data.X[Z==kk].T, bias=1)
      wHat = wHat/np.sum(wHat)
      Sigma = 1e-7 * np.eye(D)
      for kID in range(Kmax):
        Sigma += wHat[kID]*CovHat[kID]
    else:
      raise ValueError( 'Unrecognized scale matrix name %s' % (argDict['smatname']))
    invW = Sigma * (dF - D - 1) 
    #TODO: is this necessary? 1e-4*np.eye( self.D )
    return cls(dF=dF, kappa=kappa, m=m, invW=invW)
    
  def __init__(self, dF=None, invW=None, kappa=None, m=None, **kwargs):
    if invW is not None:
      self.dF = dF
      self.invW = np.asarray(invW)
      if self.invW.ndim > 2:
        self.invW = np.squeeze(self.invW)
      self.m     = np.asarray(m)
      if self.m.ndim > 1:
        self.m = self.m[0]
      self.kappa = kappa
      self.D = self.invW.shape[0]
      assert self.m.ndim == 1
      assert self.invW.ndim == 2
      assert self.m.size == self.D
    else:
      self.dF = None
      self.invW = None
      self.kappa = None
      self.m = None
    self.Cache = dict()
    
    
  ######################################################### Param updates 
  ######################################################### (M step)
  def get_post_distr( self, compSS):
    ''' Create new Distr object with posterior params
        See Bishop equations 10.59 - 10.63
    '''
    EN = float(compSS.N)
    Ex = compSS.x
    ExxT = compSS.xxT
    kappa = self.kappa + EN
    m = ( self.kappa*self.m + Ex ) / kappa
    if EN > EPS:
      mDiff = Ex/EN - self.m
      NCovMat = ExxT - np.outer(Ex,Ex)/EN
      kR = (self.kappa*EN)/(self.kappa+EN)
      invW  = self.invW + NCovMat + kR*np.outer(mDiff,mDiff)
    else:
      invW = self.invW
    return GaussWishDistr(self.dF+EN, invW, kappa, m)
    
  def post_update( self, priorD, compSS):
    ''' In-place posterior update of params given data
    '''
    EN = compSS.N
    Ex = compSS.x
    ExxT = compSS.xxT
    m = ( priorD.kappa*priorD.m + Ex ) / (priorD.kappa+EN)
    if EN > EPS:
      mDiff = Ex/EN - priorD.m
      NCovMat = ExxT - np.outer(Ex,Ex)/EN
      kR = (priorD.kappa*EN)/(priorD.kappa+EN)
      invW  = priorD.invW + NCovMat + kR*np.outer(mDiff,mDiff)
    else:
      invW = priorD.invW
    self.dF = priorD.dF + EN
    self.kappa = priorD.kappa+EN
    self.m = m
    self.invW = invW
    self.Cache = dict()
    
  def post_update_soVB(self, rho, starDistr, *args ):
    ''' In-place online update of params
    '''
    etaCUR = self.get_natural_params()
    etaSTAR = starDistr.get_natural_params()
    etaNEW = list(etaCUR)
    for i in xrange(len(etaCUR)):
      etaNEW[i] = rho*etaSTAR[i] + (1-rho)*etaCUR[i]
    self.set_natural_params( tuple(etaNEW) )

  ############################################################## Local params (E step)
  ##############################################################
  def E_log_pdf( self, Data ):
    ''' Returns E[ log p( x | theta ) ] under q(theta) <- this distr
    '''
    logp = 0.5*self.ElogdetLam() \
          -0.5*self.dF*self.dist_mahalanobis( Data.X ) \
          -0.5*self.D/self.kappa
    return logp
    
  def dist_mahalanobis(self, X):
    '''Calculate Mahalanobis distance to x
             dist(x) = (x-m)'*W*(x-m)
       If X is a matrix, we compute a vector of distances to each row vector
             dist(X)[n] = (x_n-m)'*W*(x_n-m)
    '''           
    dX = (X-self.m).T
    Q  = np.linalg.solve( self.cholinvW(), dX )
    Q *= Q
    return np.sum( Q, axis=0)
    
  ############################################################## Exp Fam accessors
  ##############################################################
  def get_log_norm_const( self ):
    ''' p( mu,Lam) = MNIW( . | self)
                   = 1/Z f(mu|Lam) g(Lam), where Z is const w.r.t mu,Lam
        This function returns 
            log( Z )= log \int f() g() d mu d Lam
    '''
    D = self.D
    dF = self.dF
    return 0.5*D*dF*LOGTWO + MVgammaln(0.5*dF, D) \
              + 0.5*dF*self.logdetW() \
              + 0.5*D*np.log(self.kappa)
    
  def get_natural_params( self ):
    etatuple = self.dF, self.kappa, self.kappa*self.m, self.invW + self.kappa*np.outer(self.m, self.m)
    return etatuple

  def set_natural_params( self, etatuple ):
    self.dF = etatuple[0]
    self.kappa = etatuple[1]
    self.m = etatuple[2]/self.kappa
    self.invW = etatuple[3] - self.kappa*np.outer( self.m,self.m)
    self.Cache = dict()
    
  ############################################################## Internal accessors
  ##############################################################
  def cholinvW(self):
    try:
      return self.Cache['cholinvW']
    except KeyError:
      try:
        self.Cache['cholinvW'] = scipy.linalg.cholesky(self.invW, lower=True)
      except:
        print self.invW[:3, :3]
        raise # throw the same exception (prob a linalg error) again
      return self.Cache['cholinvW']
      
  def logdetW(self):
    try:
      return self.Cache['logdetW']
    except KeyError:
      self.Cache['logdetW'] = -2.0*np.sum( np.log( np.diag( self.cholinvW() ))) 
      return self.Cache['logdetW']
  
  def ElogdetLam(self):
    try:
      return self.Cache['ElogdetLam']
    except KeyError:
      self.Cache['ElogdetLam'] = MVdigamma(0.5*self.dF,self.D) \
                                  + self.D*LOGTWO  + self.logdetW()
      return self.Cache['ElogdetLam']
      
  def invcholSigma( self ):
    ''' Get inverse of chol of Covariance matrix for Student-t predictive posterior
          see K. Murphy's BayesGauss.pdf, eq. 232
    ''' 
    try:
      return self.Cache['invcholSigma']
    except KeyError:
      kR = (self.kappa+1.0)/self.kappa
      Sigma = kR*self.invW/(self.dF - self.D + 1)
      cholSigma = scipy.linalg.cholesky(Sigma,lower=True)
      self.Cache['invcholSigma'] = np.linalg.inv( cholSigma)
    return self.Cache['invcholSigma']

  def cholSigma( self ):
    ''' Get chol of Covariance matrix for Student-t predictive posterior
          see K. Murphy's BayesGauss.pdf, eq. 232
    ''' 
    try:
      return self.Cache['cholSigma']
    except KeyError:
      kR = (self.kappa+1.0)/self.kappa
      Sigma = kR*self.invW/(self.dF - self.D + 1)
      cholSigma = scipy.linalg.cholesky(Sigma,lower=True)
      self.Cache['cholSigma'] = cholSigma
    return self.Cache['cholSigma']
   
  def student_t_log_norm_const( self):
    ''' See Murphy's bayesGauss.pdf Eq 313  (term before the product symbol 'x')
    '''
    try:
      return self.Cache['student_t_log_norm_const']
    except KeyError:
      v = self.dF -self.D + 1
      Gterms = gammaln(0.5*(v+self.D)) - gammaln(0.5*v)
      Vterms = 0.5*self.D*(np.log(v) + LOGPI )
      logdetSigma = -2.0*np.sum(np.log(np.diag(self.invcholSigma() )))      
      #logdetSigma = 2.0*np.sum(np.log(np.diag(self.cholSigma() )))
      self.Cache['student_t_log_norm_const'] = Gterms - Vterms - 0.5*logdetSigma
    return self.Cache['student_t_log_norm_const']
    
  def logWishNormConst(self):
    return WishartDistr.calc_log_norm_const(self.logdetW(), self.dF, self.D)

  def entropyWish(self):
    '''Calculate entropy of this Wishart distribution,
         as defined in Bishop PRML B.82
    '''
    return self.logWishNormConst() \
           - 0.5*(self.dF-self.D-1)*self.ElogdetLam() \
           + 0.5*self.D*self.dF
           
  def traceW( self, S):
    '''Calculate trace( S* self.W ) in numerically stable way
    '''
    return np.trace( np.linalg.solve( self.invW, S)  )
    #try:
    #  U = scipy.linalg.cholesky( S , lower=True )
    #except scipy.linalg.LinAlgError:
    #  U = scipy.linalg.cholesky( S + EPS*np.eye( self.D), lower=True ) 
    #q = scipy.linalg.solve_triangular( self.cholinvW(), U, lower=True)
    #return np.sum(q**2)
    
  def ECovMat(self):
    return self.invW/(self.dF -self.D-1)
    
  ############################################################## I/O 
  ##############################################################
  def to_dict(self):
    return dict(name=self.__class__.__name__,
                 dF=self.dF, invW=self.invW, m=self.m, kappa=self.kappa)
    
  def from_dict(self, Dict):
    self.m = Dict['m']
    self.invW = Dict['invW']
    self.dF = Dict['dF']
    self.kappa = Dict['kappa']
    self.D = self.invW.shape[0]
    self.Cache = dict()

  def to_string(self):
    S = self.ECovMat()[:2,:2]
    np.set_printoptions( precision=3, suppress=True)
    msg = 'E[ Mean   ] = ' + str(self.m[:2] ) +'\n'
    msg += 'Cov[ Mean ] = \n'
    msg += str( 1.0/self.kappa*S ) + '\n'
    msg += 'E[ CovMat ] = \n'
    msg += str(S) 
    if self.D > 2:
      msg += '...'
    return msg
