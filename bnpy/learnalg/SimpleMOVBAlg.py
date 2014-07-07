'''
SimpleMOVBAlg.py

Bare-bones implementation of Memoized Online VB (moVB) learn alg for bnpy models
'''
import numpy as np
import joblib
import os
import logging
from collections import defaultdict
import copy

Log = logging.getLogger('bnpy')
from LearnAlg import LearnAlg
from bnpy.suffstats import SuffStatBag
from bnpy.util import isEvenlyDivisibleFloat

class SimpleMOVBAlg(LearnAlg):

  def __init__( self, **kwargs):
    ''' Creates memoized VB learning algorithm object,
          including specialized internal fields to hold "memoized" statistics
    '''
    super(type(self),self).__init__(**kwargs)
    self.SSmemory = dict()
    self.LPmemory = dict()

  def fit(self, hmodel, DataIterator):
    ''' Run moVB learning algorithm, fit parameters of hmodel to Data,
          traversed one batch at a time from DataIterator

        Returns
        --------
        LP : None type, cannot fit all local params in memory
        Info : dict of run information, with fields
              evBound : final ELBO evidence bound
              status : str message indicating reason for termination
                        {'converged', 'max passes exceeded'}
    
    '''
    origmodel = hmodel
    # Define how much of data we see at each mini-batch
    nBatch = float(DataIterator.nBatch)
    self.lapFracInc = 1.0/nBatch
    self.nBatch = nBatch
    # Set-up progress-tracking variables
    iterid = -1
    lapFrac = np.maximum(0, self.algParams['startLap'] - 1.0/nBatch)
    if lapFrac > 0:
      # When restarting an existing run,
      #  need to start with last update for final batch from previous lap
      DataIterator.lapID = int(np.ceil(lapFrac)) - 1
      DataIterator.curLapPos = nBatch - 2
      iterid = int(nBatch * lapFrac) - 1

    # memoLPkeys : keep list of params that should be retained across laps
    self.memoLPkeys = hmodel.allocModel.get_keys_for_memoized_local_params()

    SS = None
    isConverged = False
    prevBound = -np.inf
    numConvergedInARow = 0
    self.set_start_time_now()
    while DataIterator.has_next_batch():

      # Grab new data
      Dchunk = DataIterator.get_next_batch()
      batchID = DataIterator.batchID
      
      # Update progress-tracking variables
      iterid += 1
      lapFrac = (iterid + 1) * self.lapFracInc
      self.set_random_seed_at_lap(lapFrac)

      # M step
      if self.algParams['doFullPassBeforeMstep']:
        if SS is not None and lapFrac > 1.0:
          hmodel.update_global_params(SS)
      else:
        if SS is not None:
          hmodel.update_global_params(SS)
      
      # E step
      if batchID in self.LPmemory:
        oldLPchunk = self.load_batch_local_params_from_memory(batchID)
        LPchunk = hmodel.calc_local_params(Dchunk, oldLPchunk,
                                           **self.algParamsLP)
      else:
        LPchunk = hmodel.calc_local_params(Dchunk, **self.algParamsLP)

      # Suff Stat step
      if batchID in self.SSmemory:
        oldSSchunk = self.load_batch_suff_stat_from_memory(batchID, SS.K)
        SS -= oldSSchunk

      SSchunk = hmodel.get_global_suff_stats(Dchunk, LPchunk,
                                                     doPrecompEntropy=1)

      if SS is None:
        SS = SSchunk.copy()
      else:
        assert SSchunk.K == SS.K
        SS += SSchunk

      # ELBO Update!
      evBound = hmodel.calc_evidence(SS=SS)

      # Store batch-specific stats to memory
      if self.algParams['doMemoizeLocalParams']:
        self.save_batch_local_params_to_memory(batchID, LPchunk)          
      self.save_batch_suff_stat_to_memory(batchID, SSchunk)

      # Save and display progress
      self.add_nObs(Dchunk.nObs)
      self.save_state(hmodel, iterid, lapFrac, evBound)
      self.print_state(hmodel, iterid, lapFrac, evBound)
      self.eval_custom_func(lapFrac, hmodel=hmodel, SS=SS, Dchunk=Dchunk, 
                                     LPchunk=LPchunk, batchID=batchID,
                                     SSchunk=SSchunk, learnAlg=self,
                                     evBound=evBound,
                                     )

      # Check for Convergence!
      #  evBound will increase monotonically AFTER completing first lap of the data 
      #  verify_evidence will warn if bound isn't increasing monotonically
      if lapFrac > self.algParams['startLap'] + 1.0 + 1.0/nBatch:
        isConvergedCurrent = self.verify_evidence(evBound, prevBound, lapFrac)
        if isConvergedCurrent:
          numConvergedInARow += 1.0 / nBatch
        else:
          numConvergedInARow = 0
 
        if lapFrac > 5 and numConvergedInARow > self.algParams['convergeDuration']:
          isConverged = 1

      if isConverged:
        break
      prevBound = evBound
      #.................................................... end loop over data

    # Finally, save, print and exit
    if isConverged:
      msg = "converged."
    else:
      msg = "max passes thru data exceeded."
    self.save_state(hmodel, iterid, lapFrac, evBound, doFinal=True) 
    self.print_state(hmodel, iterid, lapFrac,evBound,doFinal=True,status=msg)

    self.SS = SS # hack so we can examine global suff stats
    return None, self.buildRunInfo(evBound, msg)

  ######################################################### Load from memory
  #########################################################
  def load_batch_suff_stat_from_memory(self, batchID, doCopy=0):
    ''' Load the suff stats stored in memory for provided batchID
        Returns
        -------
        SSchunk : bnpy SuffStatDict object for batchID,
                  Contains stored values from the last visit to batchID,
                   updated to reflect any moves that happened since that visit.
    '''
    SSchunk = self.SSmemory[batchID]
    if doCopy:
      # Duplicating to avoid changing the raw data stored in SSmemory
      #  this is done usually when debugging.
      SSchunk = SSchunk.copy()
    return SSchunk  

  def load_batch_local_params_from_memory(self, batchID):
    ''' Load local parameter dict stored in memory for provided batchID
        Ensures "fast-forward" so that all recent merges/births
          are accounted for in the returned LP
        Returns
        -------
        LPchunk : bnpy local parameters dictionary for batchID
    '''
    LPchunk = self.LPmemory[batchID]
    return LPchunk

  ######################################################### Save to memory
  #########################################################
  def save_batch_suff_stat_to_memory(self, batchID, SSchunk):
    ''' Store the provided suff stats into the "memory" for later retrieval
    '''
    self.SSmemory[batchID] = SSchunk

  def save_batch_local_params_to_memory(self, batchID, LPchunk):
    ''' Store certain fields of the provided local parameters dict
          into "memory" for later retrieval.
        Fields to save determined by the memoLPkeys attribute of this alg.
    '''
    LPchunk = dict(**LPchunk) # make a copy
    allkeys = LPchunk.keys()
    for key in allkeys:
      if key not in self.memoLPkeys:
        del LPchunk[key]
    if len(LPchunk.keys()) > 0:
      self.LPmemory[batchID] = LPchunk
    else:
      self.LPmemory[batchID] = None
