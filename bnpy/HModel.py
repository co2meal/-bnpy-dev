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
import SharedMemWorker

import init
from allocmodel import AllocModelConstructorsByName
from obsmodel import ObsModelConstructorsByName
import multiprocessing
from multiprocessing import Process, Queue

                   
class HModel(object):

  ######################################################### Constructors
  #########################################################    
  def __init__(self, allocModel, obsModel):
    ''' Constructor assembles HModel given fully valid subcomponents
    '''
    self.allocModel = allocModel
    self.obsModel = obsModel
    self.inferType = allocModel.inferType
    self.initParams = None
    if hasattr(obsModel, 'setupWithAllocModel'):
      # Tell the obsModel whether to model docs or words
      obsModel.setupWithAllocModel(allocModel)



  @classmethod
  def CreateEntireModel(cls, inferType, allocModelName, obsModelName, 
                             allocPriorDict, obsPriorDict, Data):
    ''' Constructor assembles HModel and all its submodels in one call
    '''
    AllocConstr = AllocModelConstructorsByName[allocModelName]
    allocModel = AllocConstr(inferType, allocPriorDict)

    ObsConstr = ObsModelConstructorsByName[obsModelName]
    obsModel = ObsConstr(inferType, Data=Data, **obsPriorDict)

    return cls(allocModel, obsModel)
  
    
  def setupMemory(self,Data):
        # Create a JobQ (to hold tasks to be done)
    # and a ResultsQ (to hold results of completed tasks)
    manager = multiprocessing.Manager()
    self.JobQ = manager.Queue()
    self.ResultQ = manager.Queue()

    self.nWorkers=5 #TODO: change this

    for uid in range(self.nWorkers):
        SharedMemWorker.SharedMemWorker(
            uid, self.JobQ, self.ResultQ, 
            Data=Data,
            verbose=1).start() #TODO: change the verbose
        
  def copy(self):
    ''' Create a clone of this object with distinct memory allocation
        Any manipulation of clone's parameters will NOT affect self
    '''
    return copy.deepcopy(self)

  ######################################################### Local Params
  #########################################################    
  def calc_local_params(self, Data, LP=None, **kwargs):
    ''' Calculate the local parameters specific to each data item,
          given global parameters.
        This is the E-step of the EM algorithm.        
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
       
  ######################################################### Suff Stats
  #########################################################   
  def get_global_suff_stats(self, Data, LP, doAmplify=False, **kwargs):
    ''' Calculate sufficient statistics for each component,
          given data and local responsibilities
        This is necessary prep for the M-step of EM algorithm.
    '''
    SS = self.allocModel.get_global_suff_stats(Data, LP, **kwargs)
    SS = self.obsModel.get_global_suff_stats(Data, SS, LP, **kwargs)
    if doAmplify:
      # Change effective scale of the suff stats, for soVB learning
      if hasattr(Data, 'nDoc'):
        ampF = Data.nDocTotal / float(Data.nDoc)
        SS.applyAmpFactor(ampF)
      else:
        ampF = Data.nObsTotal / float(Data.nObs)
        SS.applyAmpFactor(ampF)
    return SS
  
  def calcLocalParamsAndSummarize(self, Data, LP, doAmplify=False, **kwargs):
    ''' This does both the E-step and M-step of the algorithm.
        Calculates local parameters specific to each data item and then Calculates
        sufficient statistics
    '''
    if LP is None:
      LP = dict()
    # Calculate the "soft evidence" each obsModel component has on each item
    # Fills in LP['E_log_soft_ev']
    LP = self.obsModel.calc_local_params(Data, LP, **kwargs)

    # Combine with allocModel probs of each cluster
    # Fills in LP['resp'], a Data.nObs x K matrix whose rows sum to one
    LP = self.allocModel.calc_local_params(Data, LP, **kwargs)

    SS = self.allocModel.get_global_suff_stats(Data, LP, **kwargs)
    SS = self.obsModel.get_global_suff_stats(Data, SS, LP, **kwargs)

    ##TWO OPTIONS

    ##COULD DO +=
    #In the joining step, simply do LP+=job.get()
    #Problem is that this would require changing the methods...or just changing the data object passed? So it'd be all zeros everywhere except where relevant
    #But then if all zeros, how to specify that it only matters on a specific row? Could add in arguments that specify which rows to work on...




    ##COULD DO concatenate arrays
    #But then we need to worry about the order, don't we? Or does it not matter...
    #Or does it go in the same order after we call join?
    #^will not be in same order because different speeds, but could we maintain a pointer to all these different ones

    #Still need to work out the memory issues involved
    #Could we instead of joining, simply have no returns and just modify that memory portion directly?


    if doAmplify:
      # Change effective scale of the suff stats, for soVB learning
      if hasattr(Data, 'nDoc'):
        ampF = Data.nDocTotal / float(Data.nDoc)
        SS.applyAmpFactor(ampF)
      else:
        ampF = Data.nObsTotal / float(Data.nObs)
        SS.applyAmpFactor(ampF)
    return LP, SS


  ######################################################### Global Params
  #########################################################   
  def update_global_params(self, SS, rho=None, **kwargs):
    ''' Update (in-place) global parameters given provided suff stats.
        This is the M-step of EM.
    '''
    self.allocModel.update_global_params(SS, rho, **kwargs)
    self.obsModel.update_global_params(SS, rho, **kwargs)
  
  def set_global_params(self, **kwargs):
    self.allocModel.set_global_params(**kwargs)
    self.obsModel.set_global_params(**kwargs)

  def insert_global_params(self, **kwargs):
    self.allocModel.insert_global_params(**kwargs)
    self.obsModel.insert_global_params(**kwargs)

  def reorderComps(self, order):
    self.allocModel.reorderComps(order)
    self.obsModel.reorderComps(order)

  ######################################################### Evidence
  #########################################################     
  def calc_evidence(self, Data=None, SS=None, LP=None, 
                          scaleFactor=None, todict=False, **kwargs):
    ''' Compute the evidence lower bound (ELBO) of the objective function.
    '''
    if Data is not None and LP is None and SS is None:
      LP = self.calc_local_params(Data, **kwargs)
      SS = self.get_global_suff_stats(Data, LP)

    evA = self.allocModel.calc_evidence(Data, SS, LP, todict=todict, **kwargs)
    evObs = self.obsModel.calc_evidence(Data, SS, LP, todict=todict, **kwargs)

    if scaleFactor is None:
      scaleFactor = self.obsModel.getDatasetScale(SS)
    if todict:
      evA.update(evObs)
      return evA
    else:
      return (evA + evObs) / scaleFactor
    
  def calcLogLikCollapsedSamplerState(self, SS):
      ''' 
      '''
      return self.obsModel.calcMargLik(SS) \
             + self.allocModel.calcMargLik(SS)

  ######################################################### Init params
  #########################################################
  def init_global_params(self, Data, **initArgs):
    ''' Initialize (in-place) global parameters

        Keyword Args
        -------
        K : number of components
        initname : string name of routine for initialization
    '''
    initname = initArgs['initname']
    if initname.count(os.path.sep) > 0:
      init.FromSaved.init_global_params(self, Data, **initArgs)
    elif initname.count('true') > 0:
      init.FromTruth.init_global_params(self, Data, **initArgs)
    elif initname.count('LP') > 0:
      init.FromLP.init_global_params(self, Data, **initArgs)
    else:
      # Set hmodel global parameters "from scratch", in two stages
      # * init allocmodel to "uniform" prob over comps
      # * init obsmodel in likelihood-specific, data-driven fashion
      if str(type(self.obsModel)).count('Gauss') > 0:
        init.FromScratchGauss.init_global_params(self.obsModel, 
                                                 Data, **initArgs)
      elif str(type(self.obsModel)).count('Mult') > 0:
        init.FromScratchMult.init_global_params(self.obsModel,
                                                Data, **initArgs)
      elif str(type(self.obsModel)).count('Bern') > 0:
        init.FromScratchBern.init_global_params(self.obsModel,
                                                Data, **initArgs)
      else:
        raise NotImplementedError('Unrecognized initname procedure.')

      if 'K' in initArgs:
        # Make sure K is exactly same for both alloc and obs models
        # Needed because obsModel init can sometimes yield K < Kinput
        initArgs['K'] = self.obsModel.K
      self.allocModel.init_global_params(Data, **initArgs)
   
  ######################################################### I/O Utils
  ######################################################### 
  def getAllocModelName(self):
    return self.allocModel.__class__.__name__

  def getObsModelName(self):
    return self.obsModel.__class__.__name__

  def get_model_info( self ):
    s =  'Allocation Model:  %s\n'  % (self.allocModel.get_info_string())
    s += 'Obs. Data  Model:  %s\n' % (self.obsModel.get_info_string())
    s += 'Obs. Data  Prior:  %s' % (self.obsModel.get_info_string_prior())
    return s
