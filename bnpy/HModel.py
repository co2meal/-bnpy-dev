'''
HModel.py

Generic class for representing hierarchical Bayesian models in bnpy.

Attributes
-------
allocModel : a bnpy.allocmodel.AllocModel subclass
              such as MixModel, DPMixModel, etc.
             model for generating latent structure (such as cluster assignments) 
              
obsModel : a list of bnpy.obsmodel.ObsCompSet subclass
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
from bnpy.ioutil import BNPYArgParser
# Dictionary map
#    turns string input at command line into desired bnpy objects
# string --> bnpy object constructor
AllocConstr = {'MixModel':MixModel, 'DPMixModel':DPMixModel}
ObsConstr = {'Gauss':GaussObsModel,'ZMGauss':ZMGaussObsModel}
                   
class HModel( object ):

  ######################################################### Constructors
  #########################################################    
  def __init__( self, allocModel, obsModel ):
    ''' Constructor assembles HModel given fully valid subcomponents
    '''
    self.allocModel = allocModel
    self.obsModel = obsModel
    self.inferType = allocModel.inferType

  @classmethod
  def CreateEntireModel(cls, inferType, allocModelName, obsModelName, allocPriorDict, obsPriorDict, Data):
    ''' Constructor assembles HModel and all its submodels in one call
    '''
    allocModel = AllocConstr[allocModelName](inferType, allocPriorDict)
    # support for multiple observation models
    obsModel = []
    obsModelNames = BNPYArgParser.parseObsModelName(obsModelName)
    #######
    import pdb
    pdb.set_trace()
    ########
    if(str(type(Data)).count('DiverseData')>0):
        assert len(obsModelNames)==len(Data.DataList)
        for obsModelPartName,DataPart in zip(obsModelNames,Data.DataList):
            obsModel.append(ObsConstr[obsModelPartName].CreateWithPrior(
                                            inferType, obsPriorDict, DataPart))
    else:
        # only one type of data and observation model
        assert len(obsModelNames)==1,'Data List and Model List must match'
        obsModel.append(ObsConstr[obsModelNames[0]].CreateWithPrior(
                                            inferType, obsPriorDict, Data))                                        
    return cls(allocModel, obsModel)
  
    
  def copy( self ):
    ''' Create a clone of this object with distinct memory allocation
        Any manipulation of clone's parameters will NOT affect self
    '''
    return copy.deepcopy( self )

  ######################################################### Local Params
  #########################################################    
  def calc_local_params( self, Data, LP=None, **kwargs):
    ''' Calculate the local parameters specific to each data item,
          given global parameters.
        This is the E-step of the EM algorithm.        
    '''
    if LP is None:
      LP = dict()
    # Calculate the "soft evidence" each obsModel component has on each item
    # Fills in LP['E_log_soft_ev']
    LP['E_log_soft_ev'] = np.zeros( (Data.nObs, self.obsModel[0].K) )
    if(str(type(Data)).count('DiverseData')>0):
        for obsModelPart,dataPart in zip(self.obsModel,Data.DataList):
            LP['E_log_soft_ev'] += obsModelPart.calc_local_params(dataPart, LP, **kwargs)
        Data = Data.DataList[0]    
    else:
        LP['E_log_soft_ev'] += self.obsModel[0].calc_local_params(Data, LP, **kwargs)
    
    # Combine with allocModel probs of each cluster
    # Fills in LP['resp'], a Data.nObs x K matrix whose rows sum to one
    LP = self.allocModel.calc_local_params(Data, LP, **kwargs)
    return LP

  ######################################################### Suff Stats
  #########################################################   
  def get_global_suff_stats( self, Data, LP, doAmplify=False, **kwargs):
    ''' Calculate sufficient statistics for each component,
          given data and local responsibilities
        This is necessary prep for the M-step of EM algorithm.
    '''
    SS = []
    for i,obsModel in enumerate(self.obsModel):
        SS.append(self.allocModel.get_global_suff_stats(Data.DataList[i], LP, **kwargs))
        SS[i] = self.obsModel[i].get_global_suff_stats(Data.DataList[i], SS[i], LP, obsModelId=i, **kwargs)
    if doAmplify:
      # Change effective scale of the suff stats, for soVB learning
      if hasattr(Data,"nDoc"):
        ampF = Data.nDocTotal / Data.nDoc
        for i,obsModel in enumerate(self.obsModel):
            SS[i].applyAmpFactor(ampF)
      else:
        ampF = Data.nObsTotal / Data.nObs
        for i,obsModel in enumerate(self.obsModel):
            SS[i].applyAmpFactor(ampF)
    return SS
  
  ######################################################### Global Params
  #########################################################   
  def update_global_params( self, SS, rho=None, **kwargs):
    ''' Update (in-place) global parameters given provided suff stats.
        This is the M-step of EM.
    '''
    self.allocModel.update_global_params(SS[0], rho, **kwargs)
    for i,obsModel in enumerate(self.obsModel):
        self.obsModel[i].update_global_params(SS[i], rho, obsModelId=i, **kwargs)
  
  ######################################################### Evidence
  #########################################################     
  def calc_evidence( self, Data=None, SS=None, LP=None):
    ''' Compute the evidence lower bound (ELBO) of the objective function.
    '''
    if Data is not None and LP is None and SS is None:
      LP = self.calc_local_params(Data)
      SS = self.get_global_suff_stats(Data, LP)
    ##### Could compute alloc evidence using *any* SS object from the SS list.  
    evA = self.allocModel.calc_evidence(Data, SS[0], LP)
    evObs = 0.0
    for i,obsModel in enumerate(self.obsModel):
       evObs += self.obsModel[i].calc_evidence(Data, SS[i], LP, obsModelId=i)
    return evA + evObs
  
  ######################################################### Init Global Params
  #########################################################    
  def init_global_params(self, Data, **initArgs):
    ''' Initialize (in-place) global parameters
        TODO: Only supports initialization from scratch. FIX THIS.
    '''
    ####
    import pdb
    pdb.set_trace()
    #####
    initname = initArgs['initname']
    if initname.count('true') > 0:
      #init.FromTruth.init_global_params(self, Data, **initArgs)
      raise NotImplementedError("TODO")
    elif initname.count(os.path.sep) > 0:
      #init.FromSaved.init_global_params(self, Data, **initArgs)
      raise NotImplementedError("TODO")
    if len(self.obsModel)==1: 
        obsModelPart = obsModel[0] 
        if str(type(obsModelPart)).count('Gauss') > 0:
            init.FromScratchGauss.init_global_params(self, Data, **initArgs)
        elif str(type(obsModelPart)).count('Mult') > 0:
            init.FromScratchMult.init_global_params(self, Data, **initArgs)
        else:
            raise NotImplementedError("TODO")
    else:
        init.FromScratchDiverse.init_global_params(self, Data, **initArgs)
              

  ######################################################### I/O Utils
  ######################################################### 
  def getAllocModelName(self):
    return self.allocModel.__class__.__name__

  def getObsModelName(self):
    return self.obsModel.__class__.__name__  

  def get_model_info( self ):
    s =  'Allocation Model:  %s\n'  % (self.allocModel.get_info_string())
    s += 'Obs. Data  Model:  %s\n' % ([obs.get_info_string() for obs in self.obsModel])
    s += 'Obs. Data  Prior:  %s' % ([obs.get_info_string_prior() for obs in self.obsModel])
    return s