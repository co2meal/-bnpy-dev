'''
ZMGaussObsCompSet.py

Observation likelihood model for K components of zero-mean Gaussians.
Includes as attributes
  inferType : str type of inference performed (EM or VB or soVB or moVB)
  obsPrior : prior distribution (None if EM, Wishart if VB)
  comp : list of K component distributions (ZMGauss if EM, Wishart if VB)

if EM is the inferType
  seeks point-estimate 'Sigma' of the covar for each component.
if VB (or soVB, moVB) is inferType
  seeks proper Wishart distribution of the covar of each component.
  
Constructors
-------
From scratch (sets K=0 and has no parameters until initialized)

Using a pre-created obsPrior object (or None)
>> obsModel = ZMGaussObsCompSet('EM', min_covar=1e-9)
>> obsModel = ZMGaussObsCompSet('VB', obsPrior=myWishartDistr)

Create this object and an identity prior all in one line,
ensuring the dimension matches the data object of interest "Data"
>> obsModel = ZMGaussObsCompSet.InitFromData('VB', dict(smatname='eye'), Data)

From pre-allocated parameters (sets K and parameters exactly)

Here, we make an EM version with 10 covariance matrices (each scaled identity)
>> oDict = dict(inferType='EM', min_covar=1e-9)
>> cDictList = [dict(Sigma=k * np.eye(2)) for k in 10]
>> obsModel = ZMGaussObsCompSet.InitFromCompDicts(oDict, None, cDictList)
'''
import numpy as np
import scipy.linalg
import os
import copy

from ObsCompSet import ObsCompSet
from bnpy.distr import ZMGaussDistr
from bnpy.distr import WishartDistr
from bnpy.util import np2flatstr, dotATA, dotATB, dotABT
from bnpy.util import LOGPI, LOGTWOPI, LOGTWO, EPS

class ZMGaussObsCompSet( ObsCompSet ):
  
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
    if 'min_covar' in oDict:
      self = cls( oDict['inferType'], obsPrior=obsPrior, min_covar=oDict['min_covar'])
    else:
      self = cls( oDict['inferType'], obsPrior=obsPrior)
    self.K = len(compDictList)
    self.comp = [None for k in range(self.K)]
    if self.inferType == 'EM':
      for k in xrange(self.K):
        self.comp[k] = ZMGaussDistr( **compDictList[k] )
        self.D = self.comp[k].D
    elif self.inferType.count('VB') > 0:
      for k in xrange(self.K):
        self.comp[k] = WishartDistr( **compDictList[k] )
        self.D = self.comp[k].D
    return self
    
  @classmethod
  def InitFromData(cls, inferType, priorArgDict, Data):
    D = Data.dim
    if inferType == 'EM':
      obsPrior = None
      return cls(inferType, D, obsPrior, min_covar=priorArgDict['min_covar'])
    else:
      obsPrior = WishartDistr.InitFromData(priorArgDict,Data)
      return cls(inferType, D, obsPrior)
  
      
  #########################################################  Standard Gaussian accessors
  ######################################################### 
  def get_mean_for_comp( self, k):
    return np.zeros( self.D )
    
  def get_covar_mat_for_comp(self, k):
    return self.comp[k].ECovMat()
  
  #########################################################  Suff Stat Calc
  ######################################################### 
  def get_global_suff_stats(self, Data, SS, LP, **kwargs):
    ''' Calculate suff stats for the covariance matrix of each component
        see ZMGaussDerivation
        xxT[k] = E[ x * x.T ] where x is the col vector of each observation
            = sum_{n=1}^N r_nk x * x.T
    '''
    sqrtResp = np.sqrt(LP['resp'])
    K = sqrtResp.shape[1]
    xxT = np.zeros( (K, self.D, self.D) )
    for k in xrange(K):
      xxT[k] = dotATA(sqrtResp[:,k][:,np.newaxis]*Data.X )
    SS.xxT = xxT
    return SS

  #########################################################  Global Param Update Calc
  #########################################################
  def update_obs_params_EM( self, SS, Krange, **kwargs):
    for k in Krange:
      #curSS = SS.comp[k]
      curSS = SS.getComp(k)
      covMat  = curSS.xxT/curSS.N
      covMat  += self.min_covar*np.eye(self.D)
      self.comp[k] = ZMGaussDistr( Sigma=covMat )
           				 
  def update_obs_params_VB( self, SS, Krange, **kwargs):
    for k in Krange:
      curSS = SS.getComp(k)
      self.comp[k] = self.obsPrior.get_post_distr(curSS)

  def update_obs_params_soVB( self, SS, rho, Krange, **kwargs):
    for k in Krange:
      curSS = SS.getComp(k)
      Dstar = self.obsPrior.get_post_distr(curSS)
      self.comp[k].post_update_soVB( rho, Dstar)
      
  #########################################################  Evidence Calc
  #########################################################
  def calc_evidence( self, Data, SS, LP=None):
    if self.inferType == 'EM': 
      # handled in alloc model and aggregated in HModel
      return 0
    else:
      return self.E_logpX(SS) + self.E_logpPhi() - self.E_logqPhi()    

  def E_logpX( self, SS ):
    lpX = np.zeros( self.K )
    for k in range(self.K):
      if SS.N[k] == 0:
        continue
      lpX[k] = SS.N[k]*self.comp[k].ElogdetLam() - \
                 self.comp[k].E_traceLam( SS.xxT[k] )
    return 0.5*np.sum( lpX ) - 0.5*np.sum(SS.N)*self.D*LOGTWOPI
     
  def E_logpPhi( self ):
    return self.E_logpLam()
      
  def E_logqPhi( self ):
    return self.E_logqLam()  
    
  def E_logpLam( self ):
    lp = np.zeros( self.K) 
    for k in xrange( self.K ):
      lp[k] = 0.5*(self.obsPrior.v - self.D -1)*self.comp[k].ElogdetLam()
      lp[k] -= 0.5*self.comp[k].E_traceLam( cholS=self.obsPrior.cholinvW() )
    return lp.sum() - self.K * self.obsPrior.get_log_norm_const()
 
  def E_logqLam( self ):
    lp = np.zeros( self.K)
    for k in xrange( self.K):
      lp[k] = -1*self.comp[k].get_entropy()
    return lp.sum()

  ############################################################## human readable I/O  
  ##############################################################  
  def get_name(self):
    return 'ZMGauss'
      
  def get_info_string(self):
    return 'Zero-mean Gaussian distribution'
      
  def get_info_string_prior(self):
    if self.obsPrior is None:
      return 'None'
    else:
      return 'Wishart on precision matrix \Lam \n' + self.obsPrior.to_string()

  def get_human_global_param_string(self):
    if self.qType == 'EM':
      return '\n'.join( [np2flatstr(self.comp[k].get_covar(),'% 4.2f') for k in range(self.K)] )
    else:
      return '\n'.join( [np2flatstr(self.comp[k].invW,'% 4.2f') for k in range(self.K)])
  

  ############################################################## MAT file I/O  
  ############################################################## 
  def get_prior_dict( self ):
    if self.obsPrior is None:
      PDict = dict(min_covar=self.min_covar, name="NoneType")
    else:
      PDict = self.obsPrior.to_dict()
    return PDict
    
