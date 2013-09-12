''' GaussCCObsCompSet : object for managing a prior and set of K components
       for conditionally conjugate gaussian
'''
from IPython import embed
import numpy as np
import scipy.linalg
import copy

from bnpy.util import dotATB, dotABT, dotATA
from bnpy.util import LOGPI, LOGTWOPI, LOGTWO, EPS
from bnpy.util import MVgammaln, MVdigamma
from bnpy.util import np2flatstr

from ObsCompSet import ObsCompSet
from bnpy.distr import GaussianDistr
from bnpy.distr import WishartDistr
from bnpy.distr import GaussWishCCDistr

class GaussCCObsCompSet( ObsCompSet ):

  ##########################################################  Viz
  def get_mean(self, kk):
    return self.comp[kk].muD.m
    
  def get_covar_mat(self, kk):
    if self.qType =='EM':
      return np.linalg.inv( self.comp[kk].L )
    else:
      return self.comp[kk].LamD.invW/( self.comp[kk].LamD.v-self.D-1)

  ##########################################################  Init/Accessors
  def __init__( self, K, qType='EM', obsPrior=None, min_covar=1e-8):
    self.K = K
    self.qType = qType
    self.obsPrior = obsPrior
    if obsPrior is not None:
      self.obsPrior.qType = qType
      if hasattr(self.obsPrior,'D'):
        self.D = self.obsPrior.D
    else:
      self.D = None
    if qType == 'EM':
      self.min_covar = min_covar
    self.comp = [None for k in xrange(K)]

  def get_name(self):
    return 'Gaussian'

  def get_info_string(self):
    return 'Gaussian distribution'
  
  def get_info_string_prior(self):
    if self.obsPrior is None:
      return 'None'
    else:
      np.set_printoptions( precision=3, suppress=True)
      msg = 'Gaussian on \mu, Wishart on \Lam\n'
      msg += '   E[ mu ] = '+str(self.obsPrior.muD.m[:2])+'\n'
      msg += '   Cov[mu] = \n'+ str( np.linalg.inv(self.obsPrior.muD.L))+'\n'
      msg += self.obsPrior.LamD.to_string()
      return msg

  def get_human_global_param_string(self):
    if self.qType == 'EM':
      return '\n'.join([np2flatstr(self.comp[k].m, '% 7.2f') for k in xrange(self.K)] )
    else:
      return '\n'.join([np2flatstr(self.comp[k].muD.m,'% 7.2f') for k in xrange(self.K)] )
  
  def get_prior_dict( self ):
    if self.obsPrior is None:
      return dict( K=self.K, min_covar=self.min_covar )
    PDict = self.obsPrior.to_dict()
    PDict['K'] = self.K
    return PDict
  
  def from_dict( self, CompList ):
    if self.qType == 'EM':
      for k in xrange( self.K ):
        self.comp[k] = GaussianDistr( **CompList[k] )
    elif self.qType.count('VB') > 0:
      for k in xrange( self.K):
        self.comp[k] = GaussWishCCDistr()
        self.comp[k].from_dict( CompList[k] )
          
  #########################################################  Config param settings  
  def config_from_data( self, Data, **kwargs):
    self.D = Data['X'].shape[1]
    if self.obsPrior is not None:
      self.obsPrior.config_from_data( Data, **kwargs)
      self.comp = [copy.deepcopy(self.obsPrior) for k in range(self.K) ]
    
  #########################################################  
  #########################################################  Suff Stat Calc
  #########################################################   
  def get_global_suff_stats( self, Data, SS, LP, Krange=None, Ntotal=None ):
    X = Data['X']
    try:     XT = Data['XT']
    except:  XT = X.T.copy() 

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


  def inc_suff_stats( self, ks, Xcur, SS):
    if 'x' not in SS:
      SS['x'] = np.zeros( (self.K, self.D) )
      SS['xxT'] = np.zeros( (self.K, self.D, self.D) )
    assert len(ks)==1
    SS['x'][ks] += Xcur
    SS['xxT'][ks] += np.outer(Xcur, Xcur)
    return SS
    
  def dec_suff_stats( self, ks, Xcur, SS):
    if 'x' not in SS:
      SS['x'] = np.zeros( (self.K, self.D) )
      SS['xxT'] = np.zeros( (self.K, self.D, self.D) )
    if SS['N'][ks] > 0:
      SS['x'][ks] -= Xcur
      SS['xxT'][ks] -= np.outer(Xcur, Xcur)
    return SS
  
    
  #########################################################  
  #########################################################  Param Update Calc
  ######################################################### 
  def update_obs_params_EM( self, SS, Krange, **kwargs):
    for k in Krange:      
      mean    = SS['x'][k]/SS['N'][k]
      covMat  = SS['xxT'][k]/SS['N'][k] - np.outer(mean,mean)
      covMat  += self.min_covar *np.eye( self.D )      
      precMat = np.linalg.solve( covMat, np.eye(self.D) )
      #if self.obsPrior is not None:
      #  TODO handle prior updates here   
      self.comp[k] = GaussianDistr( mean, precMat )
    
  def update_obs_params_VB(self, SS, Krange):
    for k in Krange:
      try:
        ELam = self.comp[k].LamD.ELam()
      except Exception:
        ELam = self.obsPrior.LamD.ELam()
      self.comp[k] = self.obsPrior.get_post_distr( SS['N'][k], SS['x'][k], SS['xxT'][k],ELam)

  def update_obs_params_VB_stochastic( self, SS, rho, Krange, **kwargs):
    for k in Krange:
      try:
        ELam = self.qobsDistr[k].LamD.ELam()
      except Exception:
        ELam = self.obsPrior.LamD.ELam()
      postD = self.obsPrior.get_post_distr( SS['N'][k], SS['x'][k], SS['xxT'][k],ELam)
      if self.comp[k] is None:
        self.comp[k] = postD
      else:
        self.comp[k].LamD.online_update( rho, postD.LamD )
        self.comp[k].muD.online_update( rho, postD.muD )

  #########################################################  
  #########################################################  Evidence Calc
  #########################################################     
  def calc_evidence( self, Data, SS, LP):
    if self.qType == 'EM': 
      return 0 # handled by alloc model
    return self.E_logpX( LP, SS) + self.E_logpPhi() - self.E_logqPhi()
  
  def E_logpX( self, LP, SS ):
    ''' E_{q(Z), q(Phi)} [ log p(X) ]
       Bishop PRML eq. 10.71
    '''
    lpX = np.zeros( self.K )
    for k in xrange( self.K ):
      LamD = self.comp[k].LamD
      muD  = self.comp[k].muD
      lpX[k]  = 0.5*SS['N'][k]*LamD.ElogdetLam()
      lpX[k] -= 0.5*SS['N'][k]*self.D*LOGTWOPI
      lpX[k] -= 0.5*LamD.E_traceLam( SS['N'][k]*muD.get_covar() )

      xmT = np.outer(SS['x'][k],muD.m)
      xmxmT  =  SS['xxT'][k] - xmT - xmT.T + SS['N'][k]*np.outer(muD.m, muD.m)
      lpX[k] -= 0.5*LamD.E_traceLam( xmxmT )
    return lpX.sum()
    
  def E_logpPhi( self ):
    return self.E_logpLam() + self.E_logpMu()
      
  def E_logqPhi( self ):
    return self.E_logqLam() + self.E_logqMu()
  
  def E_logpMu( self ):
    '''
    '''
    muP = self.obsPrior.muD
    lp = muP.get_log_norm_const() * np.ones( self.K )   
    for k in range( self.K ):
      muD = self.comp[k].muD
      lp[k] -= 0.5*np.trace( np.dot(muP.L, muD.get_covar() ) )
      lp[k] -= 0.5*muP.dist_mahalanobis( muD.m )
    return lp.sum()
    
  def E_logpLam( self ):
    '''
    '''
    LamP = self.obsPrior.LamD
    lp = LamP.get_log_norm_const() * np.ones( self.K )
    for k in xrange( self.K ):
      LamD = self.comp[k].LamD
      lp[k] += 0.5*( LamP.v - LamP.D - 1 )*LamD.ElogdetLam()
      lp[k] -= 0.5*LamD.E_traceLam( LamP.invW )
    return lp.sum() 
    
  def E_logqMu( self ):
    ''' Return negative entropy!
    '''    
    lp = np.zeros( self.K)
    for k in xrange( self.K):
      lp[k] = self.comp[k].muD.get_entropy()
    return -1*lp.sum()
                     
  def E_logqLam( self ):
    ''' Return negative entropy!
    '''
    lp = np.zeros( self.K)
    for k in xrange( self.K):
      lp[k] = self.comp[k].LamD.get_entropy()
    return -1*lp.sum()
    
    
    
    
  #########################################################  
  #########################################################  Sampling Calc
  #########################################################     
  def predictive_posterior( self, Xnew, SS):
    ''' p( Xnew | SS : summary stats for previously seen data) 
    ''' 
    raise ValueError( 'Gaussian with cond. conj. prior is not collapsable for sampling')
    
  