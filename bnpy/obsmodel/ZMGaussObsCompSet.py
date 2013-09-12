''' ZMGaussObsCompSet : object for managing a prior and set of K components
       for a zero-mean Gaussian observation model
'''
from IPython import embed
import numpy as np
import scipy.linalg
import os
import copy

from ObsCompSet import ObsCompSet
from bnpy.distr import ZMGaussDistr
from bnpy.distr import WishartDistr
from bnpy.util import np2flatstr, dotATA, dotATB, dotABT
from bnpy.util import LOGPI, LOGTWOPI, LOGTWO, EPS
from bnpy.suffstats import ZMGaussSuffStat

PriorConstr = dict(EM=None, VB=WishartDistr)

class ZMGaussObsCompSet( ObsCompSet ):
  
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
    if PriorConstr[inferType] is None:
      obsPrior = None
      return cls(inferType, D, obsPrior, min_covar=priorArgDict['min_covar'])
    else:
      obsPrior = PriorConstr[inferType].InitFromData(priorArgDict,Data)
      return cls(inferType, D, obsPrior)
  
  def __init__(self, inferType, D=None, obsPrior=None, min_covar=None):
    self.inferType = inferType
    self.D = D
    self.obsPrior = obsPrior
    self.comp = list()
    self.K = 0
    if min_covar is not None:
      self.min_covar = min_covar
      
  def reset(self):
    self.K = 0
    self.comp = []  
  
  ##############################################################    
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
  

  ##############################################################    
  ############################################################## MAT file I/O  
  ############################################################## 
  def get_prior_dict( self ):
    if self.obsPrior is None:
      PDict = dict(min_covar=self.min_covar, name="NoneType")
    else:
      PDict = self.obsPrior.to_dict()
    return PDict
    
  #########################################################  
  #########################################################  Standard Gaussian accessors
  ######################################################### 
  def get_mean( self, k):
    return np.zeros( self.D )
    
  def get_covar_mat(self, k):
    return self.comp[k].ECovMat()
  
  #########################################################  
  #########################################################  Suff Stat Calc
  ######################################################### 
  def get_global_suff_stats(self, Data, SS, LP, **kwargs):
    sqrtResp = np.sqrt( LP['resp'] )
    K = sqrtResp.shape[1]
    SSxxT = np.zeros( (K, self.D, self.D) )
    for k in xrange(K):
      SSxxT[k] = dotATA(sqrtResp[:,k][:,np.newaxis]*Data.X )
    SS.fillComps(ZMGaussSuffStat, xxT=SSxxT)
    return SS

  #########################################################  
  #########################################################  Global Param Update Calc
  #########################################################
  def update_obs_params_EM( self, SS, Krange, **kwargs):
    for k in Krange:
      curSS = SS.comp[k]
      covMat  = curSS.xxT/curSS.N
      covMat  += self.min_covar*np.eye(self.D)
      self.comp[k] = ZMGaussDistr( Sigma=covMat )
           				 
  def update_obs_params_VB( self, SS, Krange, **kwargs):
    for k in Krange:
      self.comp[k] = self.obsPrior.get_post_distr( SS.comp[k] )

  def update_obs_params_soVB( self, SS, rho, Krange, **kwargs):
    for k in Krange:
      Dstar = self.obsPrior.get_post_distr( SS.comp[k] )
      self.comp[k].post_update_soVB( rho, Dstar)
      
  #########################################################  
  #########################################################  Evidence Calc
  #########################################################
  def calc_evidence( self, Data, SS, LP=None):
    if self.inferType == 'EM': 
      # handled in alloc model
      return 0
    else:
      return self.E_logpX(SS) + self.E_logpPhi() - self.E_logqPhi()    

  def E_logpX( self, SS ):
    lpX = np.zeros( self.K )
    for k in range(self.K):
      if SS.Nvec[k] == 0:
        continue
      lpX[k] = SS.Nvec[k]*self.comp[k].ElogdetLam() - \
                 self.comp[k].E_traceLam( SS.comp[k].xxT )
    return 0.5*np.sum( lpX ) - 0.5*np.sum(SS.Nvec)*self.D*LOGTWOPI
     
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
