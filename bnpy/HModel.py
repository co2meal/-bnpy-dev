'''
HModel.py
Represents hierarchical Bayesian model with conditional distributions in exponential family.     
'''
from IPython import embed

from collections import defaultdict
import numpy as np
import os
import copy
import time

from obsmodel import *
from allocmodel import *

AllocConstr = dict( MixModel=MixModel,
                    DPMixModel=DPMixModel,
                   )

ObsConstr = dict( ZMGauss=ZMGaussObsCompSet,
                  #Gauss=GaussObsCompSet,
                   )
                   
class HModel( object ):

  @classmethod
  def InitFromData(cls, inferType, allocModelName, obsModelName, allocPriorDict, obsPriorDict, Data):
    '''
    Constructor that assembles HModel in one call
    '''
    allocModel = AllocConstr[allocModelName](inferType, allocPriorDict)
    obsModel = ObsConstr[obsModelName].InitFromData(inferType, obsPriorDict, Data)
    return cls(allocModel, obsModel)
  
  def __init__( self, allocModel, obsModel ):
    self.inferType = allocModel.inferType
    self.allocModel = allocModel
    self.obsModel = obsModel
    
  def copy( self ):
    return copy.deepcopy( self )
      
  def set_inferType( self, inferType):
    self.inferType = inferType
    self.allocModel.set_inferType(inferType)
    self.obsModel.set_inferType(inferType)

  def reset(self):
    self.allocModel.reset()
    self.obsModel.reset()

  #########################################################  
  #########################################################  Suff Stat Calc
  #########################################################   
  def get_global_suff_stats( self, Data, LP, **kwargs):
    SS = self.allocModel.get_global_suff_stats( Data, LP, **kwargs )
    SS = self.obsModel.get_global_suff_stats( Data, SS, LP, **kwargs )
    return SS

  #########################################################  
  #########################################################  Global Param Update
  #########################################################   
  def update_global_params( self, SS, rho=None, **kwargs):
    self.allocModel.update_global_params(SS, rho, **kwargs)
    self.obsModel.update_global_params( SS, rho, **kwargs)
  
  #########################################################  
  #########################################################  Evidence/Obj. Func. Calc
  #########################################################     
  def calc_evidence( self, Data=None, SS=None, LP=None):
    if Data is not None and LP is None and SS is None:
      LP = self.calc_local_params( Data )
      SS = self.get_global_suff_stats( Data, LP)
    evA = self.allocModel.calc_evidence( Data, SS, LP)
    evObs = self.obsModel.calc_evidence( Data, SS, LP)
    return evA + evObs
  
  
  #########################################################  
  #########################################################  Local Param update
  #########################################################    
  def calc_local_params( self, Data, LP=None, **kwargs):
    if LP is None:
      LP = dict()
    LP = self.obsModel.calc_local_params(Data, LP, **kwargs)
    LP = self.allocModel.calc_local_params(Data, LP, **kwargs)
    return LP

  
  #########################################################  
  #########################################################  Global Param initialization
  #########################################################    
  def init_global_params(self, Data, **initArgs):
    initname = initArgs['initname']
    if initname.count('truth') > 0:
      InitEngine = FromTruthInitializer(**initArgs)
    elif initname.count(os.path.sep) > 0:
      InitEngine = FromSavedInitializer(**initArgs)
    elif str(type(self.obsModel)).count('Gauss') > 0:
      InitEngine = GaussObsSetInitializer(**initArgs)
    else:
      # TODO: more observation types!
      raise NotImplementedError("to do")
    InitEngine.init_global_params(self, Data)      
    
  #########################################################  
  #########################################################  Print to stdout
  ######################################################### 
  def print_model_info( self ):
    print 'Allocation Model:  %s'%  (self.allocModel.get_info_string() )
    print 'Obs. Data  Model:  %s'%  (self.obsModel.get_info_string() )
    print 'Obs. Data  Prior:  %s'%  (self.obsModel.get_info_string_prior() )
  
  def print_global_params( self ):
    print 'Allocation Model:'
    print  self.allocModel.get_human_global_param_string()
    print 'Obs. Data Model:'
    print  self.obsModel.get_human_global_param_string()


