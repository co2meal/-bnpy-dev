''' ObsCompSet : Generic object for managing a prior and set of K components
'''
from IPython import embed
import numpy as np
import scipy.linalg
import os
import copy

class ObsCompSet( object ):

  def __init__( self ):
    pass
    
  def get_name(self):
    pass
      
  def get_info_string(self):
    pass
      
  def get_info_string_prior(self):
    pass

  def get_human_global_param_string(self):
    pass
  
  def get_prior_dict( self ):
    pass
    
  #########################################################  Config param settings  
  def config_from_data( self, Data, **kwargs):
    pass
    
  #########################################################  
  #########################################################  Suff Stat Calc
  #########################################################   
  def get_global_suff_stats(self):
    pass

  def get_merge_suff_stats(self):
    pass
        
  def inc_suff_stats(self):
    pass
    
  def dec_suff_stats(self):
    pass
    
  #########################################################  
  #########################################################  Param Update Calc
  ######################################################### 
  def update_obs_params_CGS(self):
    pass
      
  def update_obs_params_EM(self):
    pass
    
  def update_obs_params_VB(self):
    pass
    
  def update_obs_params_VB_stochastic(self):
    pass

  #########################################################  
  #########################################################  Evidence Calc
  #########################################################     
  def calc_evidence(self):
    pass 
    
  #########################################################  
  #########################################################  DIRECTLY INHERITED
  #########################################################  
  def set_qType( self, qType):
    self.qType = qType
    if self.obsPrior is not None:
      self.obsPrior.qType = qType
  
  def update_global_params( self, SS, rho=None, Krange=None):
    ''' M-step update
    '''
    if Krange is None:
        Krange = xrange(self.K)
    if self.qType == 'CGS':
      self.update_obs_params_CGS(SS, Krange)
    elif self.qType == 'EM':
      self.update_obs_params_EM( SS, Krange )
    elif self.qType.count('VB')>0:
      if rho is None or rho == 1.0:
        self.update_obs_params_VB( SS, Krange )
      else:
        self.update_obs_params_VB_stochastic( SS, rho, Krange )
  
  def predictive_posterior( self, Xnew, SS ):
    ''' p( Xnew | SS : summary stats for previously seen data) 
    ''' 
    logps = np.zeros( self.K )
    for k in xrange( self.K ):
      logps[k] = self.comp[k].log_pdf_predict( Xnew)
    logps = logps - logps.max()  
    return np.exp( logps )
  
  #########################################################  Soft Evidence Fcns  
  def calc_local_params( self, Data, LP):
    if self.qType == 'EM':
      LP['E_log_soft_ev'] = self.log_soft_ev_mat( Data )
    elif self.qType.count('VB') >0:
      LP['E_log_soft_ev'] = self.E_log_soft_ev_mat( Data )
    return LP

  def log_soft_ev_mat( self, Data, Krange=None):
    ''' E-step update,  for EM-type inference
    '''
    if Krange is None:
      Krange = xrange(self.K)
    lpr = np.zeros( (Data['nObs'], self.K) )
    for k in Krange:
      lpr[:,k] = self.comp[k].log_pdf( Data )
    return lpr
      
  def E_log_soft_ev_mat( self, Data, Krange=None ):
    ''' E-step update, for VB-type inference
    '''    
    if Krange is None:
      Krange = xrange(self.K)
    lpr = np.zeros( (Data['nObs'], self.K) )
    for k in Krange:
      lpr[:,k] = self.comp[k].E_log_pdf( Data )
    return lpr
    
  #########################################################  Comp List add/remove
  def set_K( self, K):
    if K > self.K:
      self.comp.extend( [copy.deepcopy(self.obsPrior) for k in xrange(K-self.K)] )
    self.K = K
     
  def reset_K( self, K):
    self.K = K
    self.comp = [copy.deepcopy(self.obsPrior) for k in xrange(K)]
     
  def add_empty_component( self ):
    self.K = self.K+1
    self.comp.append( copy.deepcopy(self.obsPrior) )

  def add_component( self, c=None ):
    self.K = self.K+1
    if c is None:
      self.comp.append( copy.deepcopy(self.obsPrior) )
    else:
      self.comp.append( c )
  
  def remove_component( self, delID):
    self.K = self.K - 1
    comp = [ self.comp[kk] for kk in range(self.K) if kk is not delID ]
    self.comp = comp    
    
  def delete_components( self, keepIDs ):
    if type(keepIDs) is not list:
      keepIDs = [keepIDs]
    comp = [ self.comp[kk] for kk in range(self.K) if kk in keepIDs ]
    self.comp = comp
    self.K = len(keepIDs)
    
