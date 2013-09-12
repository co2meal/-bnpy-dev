''' ZMGaussObsCompSet : object for managing a prior and set of K components
       for a zero-mean Gaussian observation model
'''
from IPython import embed
import numpy as np
import scipy.linalg
import os
import copy

from ObsCompSet import ObsCompSet
from bnpy.distr.ZMGaussianDistr import ZMGaussianDistr
from bnpy.distr.WishartDistr import WishartDistr
from bnpy.util import np2flatstr, dotATA, dotATB, dotABT
from bnpy.util import LOGPI, LOGTWOPI, LOGTWO, EPS

class ZMGaussObsCompSet( ObsCompSet ):
   
  def __init__( self, K=3, qType='EM', obsPrior=None, min_covar=1e-8):
    self.K = K
    self.qType = qType
    self.obsPrior = obsPrior
    if self.obsPrior is not None:
      self.obsPrior.qType = qType
    if self.qType == 'EM':
      self.min_covar = min_covar
    self.comp = [None for k in xrange(K)]
    self.D = None

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
  
  def get_prior_dict( self ):
    if self.obsPrior is None:
      return dict( K=self.K, min_covar=self.min_covar )
    PDict = self.obsPrior.to_dict()
    PDict['K'] = self.K
    return PDict
  
  def from_dict( self, CompList ):
    if self.qType == 'EM':
      for k in xrange( self.K ):
        self.comp[k] = ZMGaussianDistr( **CompList[k] )
    elif self.qType.count('VB') > 0 or self.qType.count('GS') > 0:
      for k in xrange( self.K):
        self.comp[k] = WishartDistr( **CompList[k] )
        
  #########################################################  Config param settings  
  def config_from_data( self, Data, **kwargs):
    self.D = Data['X'].shape[1]
    if self.obsPrior is not None:
      self.obsPrior.config_from_data( Data, **kwargs)
      self.comp = [copy.deepcopy(self.obsPrior) for k in range(self.K) ]

  
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
  def get_global_suff_stats(self, Data, SS, LP, Krange=None, Ntotal=None, **kwargs):
    if Krange is None:  Krange = xrange( self.K)
    X = Data['X']
    sqrtResp = np.sqrt( LP['resp'] )
    SSxxT = np.zeros( (self.K, self.D, self.D) )
    for k in Krange:
      SSxxT[k] = dotATA(sqrtResp[:,k][:,np.newaxis]*X )

    if Ntotal is None:
      SS['xxT'] = SSxxT
    else:
      SS['xxT'] = SS['ampF']*SSxxT
    return SS

  def get_merge_suff_stats( self, Data, SS, kA, kB):
    SS['xxT'][kA] = SS['xxT'][kA] + SS['xxT'][kB]
    return SS
        
  def inc_suff_stats( self, k, Xcur, SS):
    if 'xxT' not in SS:
      SS['xxT'] = np.zeros( (self.K, self.D, self.D) )
    SS['xxT'][k] += np.outer(Xcur, Xcur)
    self.comp[k].post_update( self.obsPrior, SS['N'][k], ExxT=SS['xxT'][k])
    return SS
    
  def dec_suff_stats( self, k, Xcur, SS):
    if 'xxT' not in SS:
      SS['xxT'] = np.zeros( (self.K, self.D, self.D) )
    if k is None:
      return SS
    SS['xxT'][k] -= np.outer(Xcur, Xcur)    
    self.comp[k].post_update( self.obsPrior, SS['N'][k], ExxT=SS['xxT'][k])
    return SS
    
  #########################################################  
  #########################################################  Param Update Calc
  #########################################################
  def sample_global_params( self, SS, PRNG=np.random, **kwargs):
    self.comp = [None for k in xrange(self.K)]
    for k in Krange:
      Lam = self.obsPrior.sample_post( SS['N'][k], ExxT=SS['xxT'][k] )
      self.comp[k] = ZMGaussianDistr( L=Lam )
    
  def update_obs_params_CGS(self, SS, Krange, **kwargs):
    self.comp = [None for k in xrange(self.K)]
    for k in Krange:
      self.comp[k] = self.obsPrior.get_post_distr(SS['N'][k], ExxT=SS['xxT'][k])
      
  def update_obs_params_EM( self, SS, Krange, **kwargs):
    I = np.eye(self.D)
    for k in Krange:
      covMat  = SS['xxT'][k]/SS['N'][k]
      covMat  += self.min_covar*I
      self.comp[k] = ZMGaussianDistr( Sigma=covMat )
      # TO DO: consider inverse parameterization
           				 
  def update_obs_params_VB( self, SS,  Krange, **kwargs):
    for k in Krange:
      self.comp[k] = self.obsPrior.get_post_distr(SS['N'][k],ExxT=SS['xxT'][k])
      #self.comp[k].post_update( self.obsPrior, SS['N'][k], ExxT=SS['xxT'][k] )      

  def update_obs_params_VB_stochastic( self, SS, rho, Krange, **kwargs):
    for k in Krange:
      Dstar = self.obsPrior.get_post_distr( SS['N'][k], ExxT=SS['xxT'][k] )
      self.comp[k].online_update( rho, Dstar)
      
  #########################################################  
  #########################################################  Evidence Calc
  #########################################################
  def calc_log_marg_lik_combo( self, SS, kA, kB=None):
    if kB is None:
      post = self.obsPrior.get_post_distr( SS['N'][kA], ExxT=SS['xxT'][kA] )
    else:
      post = self.obsPrior.get_post_distr( SS['N'][kA]+SS['N'][kB], ExxT=SS['xxT'][kA]+SS['xxT'][kB] )
    return post.get_log_norm_const()
  
  def calc_log_marg_lik( self, Data, SS):
    lpNew = np.empty( self.K )
    for k in xrange(self.K):
      lpNew[k] = self.comp[k].get_log_norm_const()
    lpPrior = self.obsPrior.get_log_norm_const()
    lp = np.sum( lpNew - lpPrior )
    return lp -0.5*Data['nObs']*self.D*LOGTWOPI   

  def calc_evidence( self, Data, SS, LP=None):
    if self.qType == 'EM': 
      return 0 # handled by alloc model
    elif self.qType == 'CGS':
      return self.calc_log_marg_lik( Data, SS)
    else:
      return self.E_logpX(SS) + self.E_logpPhi() - self.E_logqPhi()    

  def E_logpX( self, SS ):
    ''' E_{q(Z), q(Phi)} [ log p(X) ]
    '''     
    lpX = np.zeros( self.K )
    for k in range(self.K):
      if SS['N'][k] == 0:
        continue
      lpX[k] = SS['N'][k]*self.comp[k].ElogdetLam() - self.comp[k].E_traceLam( SS['xxT'][k] )
    return 0.5*np.sum( lpX )
    #return 0.5*np.sum( lpX ) - 0.5*np.sum(SS['N'])*self.D*LOGTWOPI
     
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
