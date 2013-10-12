''' 
WishartDistr.py 

Wishart probability distribution object.

Attributes
-------
v : degrees of freedom
invW : scale matrix of size DxD
D  : dimension of the scale matrix
'''
import numpy as np
import scipy.linalg
import scipy.io

from bnpy.util import dotATA, dotABT, dotATA
from bnpy.util import MVgammaln, MVdigamma, gammaln, digamma
from bnpy.util import LOGTWO, LOGPI, LOGTWOPI, EPS

from Distr import Distr

class WishartDistr( Distr ):

  @classmethod
  def InitFromData( cls, argDict, Data):
    ''' Constructor for Wishart distr object 
        ensures that it matches dimension with Data
        and that it has the expected covar matrix specified by argDict
    '''
    D = Data.dim
    v  = np.maximum( argDict['dF'], D+2)
    if argDict['smatname'] == 'eye':
      Sigma = argDict['sF'] * np.eye(D)
    elif argDict['smatname'] == 'covdata':
      Sigma = argDict['sF'] * np.cov( X.T, bias=1)
    else:
      raise ValueError( 'Unrecognized scale matrix name %s' %(smatname) )
    return cls( v=v, invW=Sigma*(v-D-1) )

  def __init__( self, v=None, invW=None, **kwargs):
    self.v = float(v)
    self.invW = np.asarray(invW)
    self.D = self.invW.shape[0]
    self.Cache = dict()

  def cholinvW(self):
    try:
      return self.Cache['cholinvW']
    except KeyError:
      self.Cache['cholinvW'] = scipy.linalg.cholesky(self.invW,lower=True)
      return self.Cache['cholinvW']
      
  def logdetW(self):
    try:
      return self.Cache['logdetW']
    except KeyError:
      self.Cache['logdetW'] = -2.0*np.sum(np.log(np.diag(self.cholinvW())))
      return self.Cache['logdetW']
      
  def ElogdetLam(self):
    try:
      return self.Cache['ElogdetLam']
    except KeyError:
      ElogdetLam = MVdigamma(0.5*self.v,self.D) + self.D*LOGTWO +self.logdetW()
      self.Cache['ElogdetLam'] = ElogdetLam
      return ElogdetLam

  def ECovMat(self):
    try:
      return self.Cache['ECovMat']
    except KeyError:
      self.Cache['ECovMat'] = self.invW/(self.v-self.D-1)
      return self.Cache['ECovMat']
   
  def ELam(self):
    try:
      return self.Cache['ELam']
    except KeyError:
      self.Cache['ELam'] = self.v*np.linalg.solve(self.invW,np.eye(self.D))
      return self.Cache['ELam']

  ####################################################### Parameter Updates
  def get_post_distr(self, SScomp):
    ''' Create new Distr object with posterior params
    '''
    if SScomp.N == 0:
      return WishartDistr( v=self.v, invW=self.invW)
    
    v    = self.v + SScomp.N
    if hasattr(SScomp,'x'):
      pass # TO DO: non-zero mean
    else:
      invW = self.invW + SScomp.xxT
    return WishartDistr(v=v, invW=invW)
    
  def post_update(self, priorD, SScomp ):
    ''' Posterior update of internal params given data
    '''
    self.Cache = dict()
    if SScomp.N == 0:
      return None
    self.v    = priorD.v + SScomp.N
    if hasattr(SScomp,'x'):
      pass # TO DO: non-zero mean
    else:
      invW = priorD.invW + SScomp.xxT
    self.v = v
    self.invW = invW

  def post_update_soVB( self, rho, starD ):
    ''' Online update of internal params
    '''
    self.v = rho*starD.v + (1-rho)*self.v
    self.invW = rho*starD.invW + (1-rho)*self.invW
    self.Cache = dict()
    
  ############################################################## Samplers
  ##############################################################
  def sample(self,numSamples=1):
    ''' Returns samples from self
        References: 
            1) "A Numerical Procedure to Generate a Sample Covariance Matrix" - Odell and Feiveson, JASA 1966
            2) http://www.stat.duke.edu/~km68/materials/214.9%20(Wishart).pdf
    '''
    if(self.v<self.D):
        raise Exception, 'Sampling not supported, when dof < dimesnion ' 
    
    samples = np.zeros([self.D,self.D,numSamples]);
    for ctr in xrange(numSamples):
        U = samples[:,:,ctr]#np.zeros([self.D,self.D]);
        # generate diagonal elements from a chisquare distribution with df = v-i+1
        U[np.diag_indices(self.D)] = [np.sqrt(np.random.chisquare(self.v-(i+1)+1)) for i in xrange(self.D-1) ]
        # generate upper triangular elements from N(0,1) 
        U[np.triu_indices(self.D, 1)] = np.random.normal(0,1,[0.5*self.D*(self.D-1)])
        print U
        # U ~ Wishart(v,I), use Bartlett decomposition to get U ~ Wishart(v,invW)
        UPt = np.dot(U,self.cholinvW().T)
        samples[:,:,ctr] = np.dot(UPt.T,UPt)
    return samples

  ####################################################### accessors
  def get_log_norm_const( self ):
    ''' Returns log( Z ), where
         PDF(x) :=  1/Z(theta) f( x | theta )
    '''
    v = self.v # readability
    D = self.D
    return 0.5*v*D*LOGTWO + MVgammaln(0.5*v, D) + 0.5*v*self.logdetW() 

  @classmethod
  def calc_log_norm_const( self, logdetW, v, D):
    return 0.5*v*D*LOGTWO + MVgammaln(0.5*v, D) + 0.5*v*logdetW  
  
    
  def get_entropy( self ):
    ''' Returns entropy of this distribution 
          H[ p(x) ] = -1*\int p(x|theta) log p(x|theta) dx
        Remember, entropy for continuous distributions can be negative
          e.g. see Bishop Ch. 1 Eq. 1.110 for Gaussian discussion
    '''
    v = self.v
    D = self.D
    H = self.get_log_norm_const() -0.5*(v-D-1)*self.ElogdetLam() + 0.5*v*D
    return H
   
  def log_pdf( self ):
    ''' Returns log p( x | theta )
    '''
    pass
    
  def E_log_pdf( self, Data ):
    ''' Returns E[ log p( x | theta ) ] under q(theta) <- this distr
    '''
    logp = 0.5*self.ElogdetLam() \
          -0.5*self.E_dist_mahalanobis( Data.X )
    return logp
    
  def E_dist_mahalanobis(self, X ):
    '''Calculate Mahalanobis distance to x
             dist(x) = dX'*E[Lambda]*dX
       If X is a matrix, we compute a vector of distances to each row vector
             dist(X)[n] = dX[n]'*E[Lambda]*dX[n]
    '''
    Q = np.linalg.solve( self.cholinvW(), X.T )
    Q *= Q
    return self.v*np.sum( Q, axis=0)

  def traceW( self, S):
    '''Calculate trace( S* self.W ) in numerically stable way
    '''
    return np.trace( np.linalg.solve(self.invW, S)  )
    
  def E_traceLam( self, S=None, cholS=None):
    '''Calculate trace( S* E[Lambda] ) in numerically stable way
        I found in past that this can actually impact 7th sig. digit, etc. in some experiments
    '''
    if cholS is not None:
      Q = scipy.linalg.solve_triangular( self.cholinvW(), cholS, lower=True)
      return self.v * np.sum(Q**2)
    return self.v*np.trace( np.linalg.solve(self.invW, S) )

    
  ####################################################### IO
  def to_string(self):
    np.set_printoptions( precision=3, suppress=True)
    msg = 'E[ CovMat ] = \n'
    msg += str(self.ECovMat()[:2,:2]) 
    if self.D > 2:
      msg += '...'
    return msg

  def to_dict(self):
    return dict(v=self.v, invW=self.invW, name=self.__class__.__name__)

  def from_dict(self, PDict):
    self.v = PDict['v']
    self.invW = PDict['invW']
    self.D = self.invW.shape[0]
    self.Cache = dict()
