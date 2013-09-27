'''
ObsCompSet.py

Generic object for managing a prior and set of K components
'''
from IPython import embed
import numpy as np
import scipy.linalg
import os
import copy

class ObsCompSet( object ):

  def __init__(self, inferType, obsPrior=None, D=None):
    self.inferType = inferType
    self.D = D
    self.obsPrior = obsPrior
    
  def reset(self):
    pass  
    
  def set_inferType( self, inferType):
    self.inferType = inferType
  
  ############################################################## set prior parameters  
  ############################################################## 
  @classmethod
  def InitFromData( cls, inferType, obsPriorParams, Data):
    pass
    
  ############################################################## human readable I/O  
  ##############################################################  
  def get_info_string( self):
    ''' Returns one-line human-readable terse description of this object
    '''
    pass

  def get_human_global_param_string(self):
    ''' Returns human-readable numerical repr. of parameters,
          for quick inspection of correctness
    '''
    pass

  ############################################################## MAT file I/O  
  ##############################################################  
  def to_dict_essential(self):
    PDict = dict(name=self.__class__.__name__, inferType=self.inferType)
    if hasattr(self,"K"):
      PDict['K']=self.K
    if hasattr(self,'min_covar'):
      PDict['min_covar'] = self.min_covar
    return PDict
      
  def from_dict(self):
    pass
    
  def get_prior_dict( self ):
    pass
        
  #########################################################  Suff Stat Calc
  #########################################################   
  def get_global_suff_stats(self):
    pass

    
  #########################################################  Param Update Calc
  ######################################################### 
  def update_obs_params_EM(self):
    pass
    
  def update_obs_params_VB(self):
    pass
    
  def update_obs_params_VB_soVB(self):
    pass

  #########################################################  Evidence Calc
  #########################################################     
  def calc_evidence(self):
    pass 
     
  def update_global_params( self, SS, rho=None, Krange=None):
    ''' M-step update
    '''
    self.K = SS.K
    if len(self.comp) != self.K:
      self.comp = [copy.deepcopy(self.obsPrior) for k in xrange(self.K)]
    if Krange is None:
        Krange = xrange(self.K)
    if self.inferType == 'EM':
      self.update_obs_params_EM( SS, Krange )
    elif self.inferType.count('VB')>0:
      if rho is None or rho == 1.0:
        self.update_obs_params_VB( SS, Krange )
      else:
        self.update_obs_params_soVB( SS, rho, Krange )
  
  ######################################################### Local Param updates  
  def calc_local_params( self, Data, LP=dict()):
    if self.inferType == 'EM':
      LP['E_log_soft_ev'] = self.log_soft_ev_mat( Data )
    elif self.inferType.count('VB') >0:
      LP['E_log_soft_ev'] = self.E_log_soft_ev_mat( Data )
    return LP

  def log_soft_ev_mat( self, Data, Krange=None):
    ''' E-step update,  for EM-type inference
    '''
    if Krange is None:
      Krange = xrange(self.K)
    lpr = np.zeros( (Data.nObs, self.K) )
    for k in Krange:
      lpr[:,k] = self.comp[k].log_pdf( Data )
    return lpr
      
  def E_log_soft_ev_mat( self, Data, Krange=None ):
    ''' E-step update, for VB-type inference
    '''    
    if Krange is None:
      Krange = xrange(self.K)
    lpr = np.zeros( (Data.nObs, self.K) )
    for k in Krange:
      lpr[:,k] = self.comp[k].E_log_pdf( Data )
    return lpr
    
  #########################################################  Comp List add/remove
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
    
