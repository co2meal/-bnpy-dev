'''
HModel.py

Generic class for representing hierarchical Bayesian models in bnpy.

Attributes
-------
allocModel : a bnpy.allocmodel.AllocModel subclass
              such as MixModel, DPMixModel, etc.
             model for generating latent structure (such as cluster assignments) 
              
obsModel : a bnpy.obsmodel.ObsCompSet subclass
              such as GaussObsCompSet or ZMGaussObsCompSet
            model for generating observed data given latent structure

Key functions
-------
* calc_local_params
* get_global_suff_stats
* update_global_params
     
'''
from collections import defaultdict
import numpy as np
import os
import copy

import init
from obsmodel import *
from allocmodel import *

# Dictionary map that turns string input at command line into desired bnpy objects
# string --> bnpy object constructor
AllocConstr = {'MixModel':MixModel, 'DPMixModel':DPMixModel, 'AdmixModel': AdmixModel}
ObsConstr = {'ZMGauss':ZMGaussObsCompSet, 'Gauss':GaussObsCompSet, 'Mult':MultObsModel}
                   
class HModel( object ):

  @classmethod
  def CreateEntireModel(cls, inferType, allocModelName, obsModelName, allocPriorDict, obsPriorDict, Data):
    ''' Constructor that assembles HModel and all its submodels (alloc, obs) in one call
    '''
    allocModel = AllocConstr[allocModelName](inferType, allocPriorDict)
    obsModel = ObsConstr[obsModelName].InitFromData(inferType, obsPriorDict, Data)
    return cls(allocModel, obsModel)
  
  def __init__( self, allocModel, obsModel ):
    ''' Constructor that assembles HModel given fully valid subcomponents
    '''
    self.allocModel = allocModel
    self.obsModel = obsModel
    self.inferType = allocModel.inferType
    
  def copy( self ):
    ''' Create a clone of this object with distinct memory allocation
        Any manipulation of clone's internal parameters will NOT reference self
    '''
    return copy.deepcopy( self )
      
  def set_inferType( self, inferType):
    self.inferType = inferType
    self.allocModel.set_inferType(inferType)
    self.obsModel.set_inferType(inferType)

  #########################################################  Local Param update
  #########################################################    
  def calc_local_params( self, Data, LP=None, **kwargs):
    ''' Calculate the local parameters for each data item given global parameters.
        This is the E-step of the EM/VB algorithm.        
    '''
    if LP is None:
      LP = dict()
    # Calculate the "soft evidence" each obsModel component has on each item
    # Fills in LP['E_log_soft_ev']
    LP = self.obsModel.calc_local_params(Data, LP, **kwargs)
    # Combine with allocModel probs of each cluster
    # Fills in LP['resp'], a Data.nObs x K matrix whose rows sum to one
    LP = self.allocModel.calc_local_params(Data, LP, **kwargs)
    return LP

  #########################################################  Suff Stat Calc
  #########################################################   
  def get_global_suff_stats( self, Data, LP, Ntotal=None, **kwargs):
    ''' Calculate sufficient statistics for global parameters, given data and local responsibilities
        This is necessary prep for the M-step of EM/VB.
    '''
    SS = self.allocModel.get_global_suff_stats( Data, LP, **kwargs )
    SS = self.obsModel.get_global_suff_stats( Data, SS, LP, **kwargs )
    # Change effective scale (nObs) of the suff stats 
    # (useful for stochastic variational)
    if hasattr(Data,"nDoc"):
      ampF = Data.nDocTotal / Data.nDoc
      SS.applyAmpFactor(ampF)
    elif Ntotal is not None:
      ampF = Ntotal / Data.nObsTotal
      SS.applyAmpFactor(ampF)
    return SS

  #########################################################  
  #########################################################  Global Param Update
  #########################################################   
  def update_global_params( self, SS, rho=None, **kwargs):
    ''' Update (in-place) global parameters given provided sufficient statistics.
        This is the M-step of EM/VB.
    '''
    self.allocModel.update_global_params(SS, rho, **kwargs)
    self.obsModel.update_global_params( SS, rho, **kwargs)
  
  #########################################################  
  #########################################################  Evidence/Obj. Func. Calc
  #########################################################     
  def calc_evidence( self, Data=None, SS=None, LP=None):
    ''' Compute the evidence lower bound (ELBO) of the objective function.
    '''
    if Data is not None and LP is None and SS is None:
      LP = self.calc_local_params( Data )
      SS = self.get_global_suff_stats( Data, LP)
    evA = self.allocModel.calc_evidence( Data, SS, LP)
    evObs = self.obsModel.calc_evidence( Data, SS, LP)
    return evA + evObs
  
  #########################################################  
  #########################################################  Global Param initialization
  #########################################################    
  def init_global_params(self, Data, **initArgs):
    ''' Initialize (in-place) global parameters
    '''
    initname = initArgs['initname']
    if initname.count('truth') > 0:
      init.FromScratchMult.init_global_params(self, Data, **initArgs)
    elif initname.count(os.path.sep) > 0:
      init.FromSaved.init_global_params(self, Data, **initArgs)
    elif str(type(self.obsModel)).count('Gauss') > 0:
      init.FromScratchGauss.init_global_params(self, Data, **initArgs)
    elif str(type(self.obsModel)).count('Mult') > 0:
      init.FromScratchMult.init_global_params(self, Data, **initArgs)
    else:
      # TODO: more observation types!
      raise NotImplementedError("to do")
    
  #########################################################  
  #########################################################  Print to stdout
  ######################################################### 
  def get_model_info( self ):
    s =  'Allocation Model:  %s\n'  % (self.allocModel.get_info_string())
    s += 'Obs. Data  Model:  %s\n' % (self.obsModel.get_info_string())
    s += 'Obs. Data  Prior:  %s' % (self.obsModel.get_info_string_prior())
    return s
  
  def print_global_params( self ):
    print 'Allocation Model:'
    print  self.allocModel.get_human_global_param_string()
    print 'Obs. Data Model:'
    print  self.obsModel.get_human_global_param_string()


