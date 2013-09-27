'''
GaussObsCompSet.py

This object represents the explicit *prior* distribution (if any)
as well as the set/collection of mixture component parameters 1,2,... K   
'''
from IPython import embed
import numpy as np
import scipy.io
import scipy.linalg
import os
import copy

from ..distr import GaussDistr
from ..distr import GaussWishDistr

from ..util import LOGTWO, LOGPI, LOGTWOPI, EPS
from ..util import np2flatstr, dotATA, dotATB, dotABT
from ..util import MVgammaln, MVdigamma

from ObsCompSet import ObsCompSet

class GaussObsCompSet( ObsCompSet ):

  def __init__(self, inferType, D=None, obsPrior=None, min_covar=None):
    self.inferType = inferType
    self.D = D
    self.obsPrior = obsPrior
    self.comp = list()
    self.K = 0
    if min_covar is not None:
      self.min_covar = min_covar
      
  @classmethod
  def InitFromCompDicts(cls, oDict, obsPrior, compDictList):
    ''' Create GaussObsCompSet, all K component distrs, and its prior in one call
    '''
    if 'min_covar' in oDict:
      self = cls( oDict['inferType'], obsPrior=obsPrior, min_covar=oDict['min_covar'])
    else:
      self = cls( oDict['inferType'], obsPrior=obsPrior)
    self.K = len(compDictList)
    self.comp = [None for k in range(self.K)]
    for k in xrange(self.K):
      if self.inferType == 'EM':
        self.comp[k] = GaussDistr( **compDictList[k] )
      else:
        self.comp[k] = GaussWishDistr( **compDictList[k]) 
      self.D = self.comp[k].D
    return self
    
  @classmethod
  def InitFromData(cls, inferType, priorArgDict, Data):
    ''' Create GaussObCompSet and its prior distr in one call
        The resulting object then needs to be initialized via init_global_params,
        otherwise it has no components and can't be used in learn algs.
    '''
    D = Data.dim
    if inferType == 'EM':
      obsPrior = None
      return cls(inferType, D, obsPrior, min_covar=priorArgDict['min_covar'])
    else:
      obsPrior = GaussWishDistr.InitFromData(priorArgDict,Data)
      return cls(inferType, D, obsPrior)
  
  ############################################################## Gaussian accessors  
  ##############################################################  
  def get_mean_for_comp(self, kk):
    return self.comp[kk].m
    
  def get_covar_mat_for_comp(self, kk):
    if self.inferType =='EM':
      return np.linalg.inv(self.comp[kk].L)
    else:
      return self.comp[kk].invW / (self.comp[kk].dF - self.D - 1)
      
  ############################################################## Suff Stat  
  ############################################################## 
  def get_global_suff_stats( self, Data, SS, LP, **kwargs):
    ''' Calculate suff stats for the global parameter update
        Args
        -------
        Data : bnpy XData object
        SS : bnpy SuffStatDict object
        LP : dict of local params, with field
              resp : Data.nObs x K array whose rows sum to one
                      resp[n,k] gives posterior prob of comp k for data item n
        
        Returns
        -------
        SS : SuffStatDict object, with new fields
              x : K x D array of component-specific sums
              xxT : K x D x D array of component-specific "sums of outer products"
                    analogous to a sum of squares, but for covar matrix
    '''
    X = Data.X
    resp = LP['resp']
    K = resp.shape[1]
    
    # Expected mean for each k
    SS.x = dotATB(resp, X)
    # Expected covar for each k 
    sqrtResp = np.sqrt(resp)
    xxT = np.zeros( (K, self.D, self.D) )
    for k in xrange(K):
      xxT[k] = dotATA(sqrtResp[:,k][:,np.newaxis]*Data.X )
    SS.xxT = xxT
    return SS
    
  ############################################################## Global Param updates  
  ############################################################## 
  def update_obs_params_EM( self, SS, Krange, **kwargs):
    I = np.eye(self.D)
    for k in Krange:
      mean    = SS['x'][k]/SS['N'][k]
      covMat  = SS['xxT'][k]/SS['N'][k] - np.outer(mean,mean)
      covMat  += self.min_covar * I      
      precMat = np.linalg.solve( covMat, I )
      self.comp[k] = GaussDistr(m=mean, L=precMat)
           				 
  def update_obs_params_VB( self, SS, Krange, **kwargs):
    for k in Krange:
      self.comp[k] = self.obsPrior.get_post_distr(SS.getComp(k))
      
  def update_obs_params_soVB( self, SS, rho, Krange, **kwargs):
    for k in Krange:
      curSS = SS.getComp(k)
      Dstar = self.obsPrior.get_post_distr(SS.getComp(k))
      self.comp[k].post_update_soVB(rho, Dstar)

  ############################################################## Evidence calc  
  ############################################################## 
  def calc_evidence( self, Data, SS, LP):
    if self.inferType == 'EM':
     return 0 # handled by alloc model
    else:
      return self.E_logpX( LP, SS) + self.E_logpPhi() - self.E_logqPhi()
  
  def E_logpX( self, LP, SS ):
    ''' E_{q(Z), q(Phi)} [ log p(X) ]
        Bishop PRML eq. 10.71
    '''
    lpX = -self.D*LOGTWOPI*np.ones( self.K )
    for k in range( self.K ):
      if np.allclose( SS['N'][k], 0):
        lpX[k] += self.comp[k].ElogdetLam() - self.D/self.comp[k].kappa
      else:
        mean    = SS['x'][k]/SS['N'][k]
        covMat  = SS['xxT'][k]/SS['N'][k] - np.outer(mean,mean)
        lpX[k] += self.comp[k].ElogdetLam() - self.D/self.comp[k].kappa \
                - self.comp[k].dF* self.comp[k].traceW( covMat )  \
                - self.comp[k].dF* self.comp[k].dist_mahalanobis(mean )
    return 0.5*np.inner(SS['N'],lpX)
    
  def E_logpPhi( self ):
    return self.E_logpLam() + self.E_logpMu()
      
  def E_logqPhi( self ):
    return self.E_logqLam() + self.E_logqMu()
  
  def E_logpMu( self ):
    ''' First four RHS terms (inside sum over K) in Bishop 10.74
    '''
    lp = np.empty( self.K)    
    for k in range( self.K ):
      mWm = self.comp[k].dist_mahalanobis( self.obsPrior.m )
      lp[k] = self.comp[k].ElogdetLam() \
                -self.D*self.obsPrior.kappa/self.comp[k].kappa \
                -self.obsPrior.kappa*self.comp[k].dF*mWm
    lp += self.D*( np.log( self.obsPrior.kappa ) - LOGTWOPI)
    return 0.5*lp.sum()
    
  def E_logpLam( self ):
    ''' Last three RHS terms in Bishop 10.74
    '''
    lp = np.empty( self.K) 
    for k in xrange( self.K ):
      lp[k] = 0.5*(self.obsPrior.dF - self.D -1)*self.comp[k].ElogdetLam()
      lp[k] -= 0.5*self.comp[k].dF*self.comp[k].traceW(self.obsPrior.invW)
    return lp.sum() - self.K * self.obsPrior.logWishNormConst()
    
  def E_logqMu( self ):
    ''' First two RHS terms in Bishop 10.77
    '''    
    lp = np.zeros( self.K)
    for k in xrange( self.K):
      lp[k] = 0.5*self.comp[k].ElogdetLam()
      lp[k] += 0.5*self.D*( np.log( self.comp[k].kappa ) - LOGTWOPI )
    return lp.sum() - 0.5*self.D*self.K
                     
  def E_logqLam( self ):
    ''' Last two RHS terms in Bishop 10.77
    '''
    lp = np.zeros( self.K)
    for k in xrange( self.K):
      lp[k] -= self.comp[k].entropyWish()
    return lp.sum()
 
  
  ############################################################## human-readable I/O
  ############################################################## 
  def get_name(self):
    return 'Gauss'

  def get_info_string(self):
    return 'Gaussian distribution'
  
  def get_info_string_prior(self):
    if self.obsPrior is None:
      return 'None'
    else:
      return 'Gaussian-Wishart jointly on \mu,\Lam\n'+ self.obsPrior.to_string()

  def get_human_global_param_string(self):
    return '\n'.join( [np2flatstr(self.comp[k].m,'% 7.2f') for k in xrange(self.K)])
  
  ############################################################## machine I/O  
  ############################################################## 
  def get_prior_dict( self ):
    if self.obsPrior is None:
      PDict = dict(min_covar=self.min_covar, name="NoneType")
    else:
      PDict = self.obsPrior.to_dict()
    return PDict
 