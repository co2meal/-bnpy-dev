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
  
  ######################################################### set prior parameters  
  ######################################################### 
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
  def update_global_params(self, SS, rho=None, Krange=None):
    ''' M-step update of global parameters for each component of this obs model.
        After this update, self will have exactly the number of 
          components specified by SS.K.
        If this number is changed, all components are rewritten from scratch.
        Args
        -------
        SS : sufficient statistics object (bnpy.suffstats.SuffStatDict)
        rho : learning rate for current step of stochastic online VB (soVB)

        Returns
        -------
        None (update happens *in-place*).         
    '''
    # TODO: if Krange specified, can we smartly do a component-specific update?

    # Components of updated model exactly match those of suff stats
    self.K = SS.K
    if len(self.comp) != self.K:
      self.comp = [copy.deepcopy(self.obsPrior) for k in xrange(self.K)]
    if Krange is None:
      Krange = xrange(self.K)

    if self.inferType == 'EM':
      self.update_obs_params_EM(SS, Krange)
    elif self.inferType.count('VB') > 0:
      if rho is None or rho == 1.0:
        self.update_obs_params_VB(SS, Krange)
      else:
        self.update_obs_params_soVB(SS, rho, Krange)
  
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
  
  def calc_log_marg_lik_for_component(self, SS, kA, kB=None):
    ''' Calculate the log marginal likelihood of the data assigned
          to the given component (specified by integer ID).
        Requires Data pre-summarized into sufficient stats for each comp.
        If multiple comp IDs are provided, we combine into a "merged" component.
        
        Args
        -------
        SS : bnpy suff stats object
        kA : integer ID of target component to compute likelihood for
        kB : (optional) integer ID of second component.
             If provided, we merge kA, kB into one component for calculation.
        Returns
        -------
        scalar log probability of data assigned to given component(s)
    '''
    if kB is None:
      postDistr = self.obsPrior.get_post_distr(SS.getComp(kA))
    else:
      postDistr = self.obsPrior.get_post_distr(SS.getComp(kA) + SS.getComp(kB))
    return postDistr.get_log_norm_const()


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
    
