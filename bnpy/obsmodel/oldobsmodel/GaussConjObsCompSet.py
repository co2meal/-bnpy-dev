'''
  GaussJointObsCompSet.py
  High-level representation of Gaussian observation model
     for exponential family, using *joint* prior on \mu, \Lam
     
  This object represents the explicit *prior* distribution (if any)
     as well as the set/collection of mixture component parameters 1,2,... K   
'''
from IPython import embed
import numpy as np
import scipy.io
import scipy.linalg
import os
import copy

from bnpy.distr import GaussianDistr
from bnpy.distr import GaussWishConjDistr

from bnpy.util import LOGTWO, LOGPI, LOGTWOPI, EPS
from bnpy.util import np2flatstr, dotATA, dotATB, dotABT
from bnpy.util import MVgammaln, MVdigamma

from ObsCompSet import ObsCompSet

class GaussConjObsCompSet( ObsCompSet ):

  ##########################################################  Viz
  def get_mean(self, kk):
    return self.comp[kk].m
    
  def get_covar_mat(self, kk):
    if self.qType =='EM':
      return np.linalg.inv( self.comp[kk].L )
    else:
      return self.comp[kk].invW/( self.comp[kk].dF-self.D-1)

  ##########################################################  Init/Accessors
  def __init__( self, K, qType='EM', obsPrior=None, min_covar=1e-8):
    self.K = K
    self.qType = qType
    self.obsPrior = obsPrior
    if obsPrior is not None:
      self.obsPrior.qType = qType
    if hasattr(self.obsPrior, 'D'):
      self.D = self.obsPrior.D
    else:
      self.D = None
    self.min_covar = min_covar
    self.comp = [None for k in xrange(K)]
    
  def get_name(self):
    return 'Normal'

  def get_info_string(self):
    return 'Gaussian distribution'
  
  def get_info_string_prior(self):
    if self.obsPrior is None:
      return 'None'
    else:
      return 'Gaussian-Wishart jointly on \mu,\Lam\n'+ self.obsPrior.to_string()

  def get_human_global_param_string(self):
    return '\n'.join( [np2flatstr(self.comp[k].m,'% 7.2f') for k in xrange(self.K)])
  
  def get_prior_dict( self ):
    if self.obsPrior is None:
      return dict()
    return self.obsPrior.to_dict()

  def from_dict( self, CompList ):
    if self.qType == 'EM':
      for k in xrange( self.K ):
        self.comp[k] = GaussianDistr( **CompList[k] )
    elif self.qType.count('VB') > 0 or self.qType.count('GS')>0:
      for k in xrange( self.K):
        self.comp[k] = GaussWishConjDistr()
        self.comp[k].from_dict( CompList[k] )
        
  #########################################################  Config param settings    
  def config_from_data( self, Data, **kwargs):
    self.D = Data['X'].shape[1]
    if self.obsPrior is not None:
      self.obsPrior.config_from_data( Data, **kwargs)
      self.comp = [copy.deepcopy(self.obsPrior) for k in range(self.K) ]

  ################################################################## Suff stats
  def get_global_suff_stats( self, Data, SS, LP, Krange=None, Ntotal=None, **kwargs ):
    X = Data['X']
    try: XT = Data['XT']
    except: XT = X.T.copy() 

    resp = LP['resp']
    if Krange is None:   
      Krange = xrange( self.K)
      SS['x'] = dotATB( resp, X) 
    else:
      SS['x'] = np.zeros( (self.K, self.D) )
      SS['x'][Krange] = dotATB( resp[:,Krange], X)

    SSxxT = np.zeros( (self.K, self.D, self.D) )
    for k in Krange:
      SSxxT[k] = np.dot( XT*resp[:,k], X)

    if Ntotal is None:
      SS['xxT'] = SSxxT
    else:
      SS['x'] = SS['ampF']*SS['x']
      SS['xxT'] = SS['ampF']*SSxxT
    return SS
  
  def get_merge_suff_stats( self, Data, SS, kA, kB):
    SS['x'][kA] = SS['x'][kA] + SS['x'][kB]
    SS['xxT'][kA] = SS['xxT'][kA] + SS['xxT'][kB]
    return SS    
            
  def inc_suff_stats( self, k, Xcur, SS):
    if 'x' not in SS:
      SS['x'] = np.zeros( (self.K, self.D) )
      SS['xxT'] = np.zeros( (self.K, self.D, self.D) )
    SS['x'][k] += Xcur
    SS['xxT'][k] += np.outer(Xcur, Xcur)
    self.comp[k].post_update(self.obsPrior, SS['N'][k], SS['x'][k], SS['xxT'][k]) 
    return SS
    
  def dec_suff_stats( self, k, Xcur, SS):
    if 'x' not in SS:
      SS['x'] = np.zeros( (self.K, self.D) )
      SS['xxT'] = np.zeros( (self.K, self.D, self.D) )
    if k is None:
      return SS
    SS['x'][k] -= Xcur
    SS['xxT'][k] -= np.outer(Xcur, Xcur)    
    self.comp[k].post_update(self.obsPrior, SS['N'][k], SS['x'][k], SS['xxT'][k]) 
    return SS
  
    
  #########################################################  
  #########################################################  Param Update Calc
  ######################################################### 
  def update_obs_params_EM( self, SS, Krange, **kwargs):
    I = np.eye(self.D)
    for k in Krange:
      mean    = SS['x'][k]/SS['N'][k]
      covMat  = SS['xxT'][k]/SS['N'][k] - np.outer(mean,mean)
      covMat  += self.min_covar * I      
      precMat = np.linalg.solve( covMat, I )
      self.comp[k] = GaussianDistr( mean, precMat )
           				 
  def update_obs_params_VB( self, SS, Krange, **kwargs):
    for k in Krange:
      self.comp[k] = self.obsPrior.get_post_distr( \
                                 SS['N'][k],SS['x'][k],SS['xxT'][k] )
      
  def update_obs_params_VB_stochastic( self, SS, rho, Krange, **kwargs):
    ''' TO DO
        -------
        Write down the exp family representation of the joint GaussWish prior
          and figure out natural parameters, etc.
    '''
    for k in Krange:
      starDistr = self.obsPrior.get_post_distr( SS['N'][k],SS['x'][k],SS['xxT'][k] )
      self.comp[k].online_update( rho, starDistr )
  
  def update_obs_params_CGS(self, SS, Krange, **kwargs):
    self.comp = [None for k in xrange(self.K)]
    for k in Krange:
      self.comp[k] = self.obsPrior.get_post_distr(SS['N'][k],SS['x'][k],SS['xxT'][k] )
      
  #########################################################  Evidence Bound Fcns  
  def calc_evidence( self, Data, SS, LP):
    if self.qType == 'EM': return 0 # handled by alloc model
    if self.qType == 'CGS':
      return self.calc_log_marg_lik( Data, SS)
    return self.E_logpX( LP, SS) \
           + self.E_logpPhi() - self.E_logqPhi()
  
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
    #CHANGED 2013021.  logWishNormConst returns log(Z) where pdf is 1/Z f(stuff)
    #return lp.sum() + self.K * self.obsPrior.logWishNormConst() 
    
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


  #########################################################  
  #########################################################  Sampler Ev Calc
  #########################################################
  def calc_log_marg_lik_combo( self, SS, kA, kB=None):
    if kB is None:
      post = self.obsPrior.get_post_distr( SS['N'][kA], SS['x'][kA], SS['xxT'][kA] )
    else:
      post = self.obsPrior.get_post_distr( SS['N'][kA]+SS['N'][kB],  SS['x'][kA]+ SS['x'][kB], SS['xxT'][kA]+SS['xxT'][kB] )
    return post.get_log_norm_const()

  def calc_log_marg_lik( self, Data, SS):
    lpX = np.empty( self.K )
    for k in xrange(self.K):
      lpX[k] = self.comp[k].get_log_norm_const()
    lp0 = self.obsPrior.get_log_norm_const()
    logM = np.sum( lpX - lp0 )
    return logM
