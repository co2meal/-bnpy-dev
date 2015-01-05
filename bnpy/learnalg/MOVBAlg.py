'''
MOVBAlg.py

Implementation of Memoized Online VB (moVB) learn alg for bnpy models
'''
import os
import copy
import numpy as np
import logging
Log = logging.getLogger('bnpy')

from LearnAlg import LearnAlg
from LearnAlg import makeDictOfAllWorkspaceVars

class MOVBAlg(LearnAlg):

  def __init__( self, **kwargs):
    ''' Construct memoized variational learning algorithm instance.

        Includes specialized internal fields to hold "memoized" statistics
    '''
    LearnAlg.__init__(self, **kwargs)
    self.SSmemory = dict()
    self.LPmemory = dict()

  def doDebug(self):
    debug = self.algParams['debug']
    return debug.count('q') or debug.count('on') or debug.count('interact')
          
  def doDebugVerbose(self):
    return self.doDebug() and self.algParams['debug'].count('q') == 0


  ######################################################### fit
  ######################################################### 
  def fit(self, hmodel, DataIterator):
    ''' Run learning algorithm that fits parameters of hmodel to Data.

        Returns
        --------
        Info : dict of run information.
  
        In-place updates
        --------
        hmodel.
    '''
    ## Initialize Progress Tracking vars like nBatch, lapFrac, etc.
    iterid, lapFrac = self.initProgressTrackVars(DataIterator)

    ## Keep list of params that should be retained across laps
    self.memoLPkeys = hmodel.allocModel.get_keys_for_memoized_local_params()

    ## Save initial state
    self.saveParams(lapFrac, hmodel)

    ## Begin loop over batches of data...
    SS = None
    isConverged = False
    self.set_start_time_now()
    while DataIterator.has_next_batch():

      # Grab new data
      Dchunk = DataIterator.get_next_batch()
      batchID = DataIterator.batchID
      Dchunk.batchID = batchID
      
      # Update progress-tracking variables
      iterid += 1
      lapFrac = (iterid + 1) * self.lapFracInc
      self.lapFrac = lapFrac
      nLapsCompleted = lapFrac - self.algParams['startLap']
      self.set_random_seed_at_lap(lapFrac)
      if self.doDebugVerbose():
        self.print_msg('========================== lap %.2f batch %d' \
                       % (lapFrac, batchID))

      ## Local convergence thr
      if self.isFirstBatch(lapFrac) and 'convThrLP' in self.algParamsLP:
        if lapFrac <= 1:
          finalConvThr = self.algParamsLP['convThrLP']
          initConvThr = self.algParamsLP['initconvThrLP']          
        if initConvThr > 0:
          assert initConvThr >= finalConvThr
          fracComplete = self.lapFrac / self.algParamsLP['plateauLapLP']
          convThr = finalConvThr + initConvThr * np.exp(-7 * fracComplete)
          self.algParamsLP['convThrLP'] = convThr

      ## Local/E step
      self.algParamsLP['lapFrac'] = lapFrac ## logging
      self.algParamsLP['batchID'] = batchID
      LPchunk = self.memoizedLocalStep(hmodel, Dchunk, batchID)
      
      self.saveDebugStateAtBatch('Estep', batchID, Dchunk=Dchunk,
                                 SS=SS, hmodel=hmodel, LPchunk=LPchunk)

      ## Summary step
      SS, SSchunk = self.memoizedSummaryStep(hmodel, SS,
                                             Dchunk, LPchunk, batchID)
      ## Global step
      self.GlobalStep(hmodel, SS, lapFrac)

      ## ELBO calculation
      evBound = hmodel.calc_evidence(SS=SS)
      if nLapsCompleted > 1.0:
        # evBound will increase monotonically AFTER first lap of the data 
        # verify_evidence will warn if bound isn't increasing monotonically
        self.verify_evidence(evBound, prevBound, lapFrac)

      if self.doDebug() and lapFrac >= 1.0:
        self.verifyELBOTracking(hmodel, SS, evBound)

      self.saveDebugStateAtBatch('Mstep', batchID, Dchunk=Dchunk, SSchunk=SSchunk,
                                 SS=SS, hmodel=hmodel, LPchunk=LPchunk)


      ## Assess convergence
      countVec = SS.getCountVec()
      if lapFrac > 1.0:
        isConverged = self.isCountVecConverged(countVec, prevCountVec)
        self.setStatus(lapFrac, isConverged)

      ## Display progress
      self.updateNumDataProcessed(Dchunk.get_size())
      if self.isLogCheckpoint(lapFrac, iterid):
        self.printStateToLog(hmodel, evBound, lapFrac, iterid)

      ## Save diagnostics and params
      if self.isSaveDiagnosticsCheckpoint(lapFrac, iterid):
        self.saveDiagnostics(lapFrac, SS, evBound)
      if self.isSaveParamsCheckpoint(lapFrac, iterid):
        self.saveParams(lapFrac, hmodel, SS)

      ## Custom func hook
      self.eval_custom_func(**makeDictOfAllWorkspaceVars(**vars()))

      if isConverged and self.isLastBatch(lapFrac) \
         and nLapsCompleted >= self.algParams['minLaps']:
        break
      prevCountVec = countVec.copy()
      prevBound = evBound
      #.................................................... end loop over data

    # Finished! Save, print and exit
    self.printStateToLog(hmodel, evBound, lapFrac, iterid, isFinal=1)
    self.saveParams(lapFrac, hmodel, SS)
    self.eval_custom_func(isFinal=1, **makeDictOfAllWorkspaceVars(**vars()))

    return self.buildRunInfo(evBound=evBound, SS=SS,
                             LPmemory=self.LPmemory,
                             SSmemory=self.SSmemory)

  ######################################################### Local step
  #########################################################
  def memoizedLocalStep(self, hmodel, Dchunk, batchID):
    ''' Execute local step on data chunk.

        Returns
        --------
        LPchunk : dict of local params for current batch
    '''
    if batchID in self.LPmemory:
      oldLPchunk = self.load_batch_local_params_from_memory(batchID)
    else:
      oldLPchunk = None
    LPchunk = hmodel.calc_local_params(Dchunk, oldLPchunk,
                                       **self.algParamsLP)
    if self.algParams['doMemoizeLocalParams']:
      self.save_batch_local_params_to_memory(batchID, LPchunk) 
    return LPchunk

  def load_batch_local_params_from_memory(self, batchID, doCopy=0):
    ''' Load local parameter dict stored in memory for provided batchID
        Ensures "fast-forward" so that all recent merges/births
          are accounted for in the returned LP
        Returns
        -------
        LPchunk : bnpy local parameters dictionary for batchID
    '''
    LPchunk = self.LPmemory[batchID]
    if doCopy:
      # Duplicating to avoid changing the raw data stored in LPmemory
      # Usually for debugging only
      LPchunk = copy.deepcopy(LPchunk)
    return LPchunk

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

  ######################################################### Summary step
  #########################################################
  def memoizedSummaryStep(self, hmodel, SS, Dchunk, LPchunk, batchID):
    ''' Execute summary step on current batch and update aggregated SS.

        Returns
        --------
        SS : updated aggregate suff stats
        SSchunk : updated current-batch suff stats
    '''
    if batchID in self.SSmemory:
      oldSSchunk = self.load_batch_suff_stat_from_memory(batchID) 
      assert oldSSchunk.K == SS.K
      SS -= oldSSchunk
    SSchunk = hmodel.get_global_suff_stats(Dchunk, LPchunk,
                                           doPrecompEntropy=1)
    if SS is None:
      SS = SSchunk.copy()
    else:
      assert SSchunk.K == SS.K
      SS += SSchunk
    self.save_batch_suff_stat_to_memory(batchID, SSchunk)

    ## Force aggregated suff stats to obey required constraints.
    # This avoids numerical issues caused by incremental updates
    if hasattr(hmodel.allocModel, 'forceSSInBounds'):
      hmodel.allocModel.forceSSInBounds(SS)
    if hasattr(hmodel.obsModel, 'forceSSInBounds'):
      hmodel.obsModel.forceSSInBounds(SS)
    return SS, SSchunk

  def load_batch_suff_stat_from_memory(self, batchID, doCopy=0,
                                             **kwargs):
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
      # Usually for debugging only
      SSchunk = SSchunk.copy()
    return SSchunk  


  def save_batch_suff_stat_to_memory(self, batchID, SSchunk):
    ''' Store the provided suff stats into the "memory" for later retrieval
    '''
    if SSchunk.hasSelectionTerms():
      del SSchunk._SelectTerms
    self.SSmemory[batchID] = SSchunk


  ######################################################### Global step
  #########################################################
  def GlobalStep(self, hmodel, SS, lapFrac):
    ''' Execute global update of hmodel params, if appropriate at current lap.

        Returns
        --------
        None. hmodel updated in-place.
    '''
    doFullPass = self.algParams['doFullPassBeforeMstep']

    if self.algParams['doFullPassBeforeMstep'] == 1:
      if lapFrac >= 1.0:
        hmodel.update_global_params(SS)
    elif doFullPass > 1.0:
      if lapFrac >= 1.0 or (doFullPass < SS.nDoc):
        # update if we've seen specified num of docs, not before
        hmodel.update_global_params(SS)
    else:
      hmodel.update_global_params(SS)


  ######################################################### Init nBatch, etc.
  #########################################################
  def initProgressTrackVars(self, DataIterator):
    ''' Initialize internal attributes like nBatch, lapFracInc, etc.
    '''
    # Define how much of data we see at each mini-batch
    nBatch = float(DataIterator.nBatch)
    self.nBatch = nBatch
    self.lapFracInc = 1.0/nBatch

    # Set-up progress-tracking variables
    iterid = -1
    lapFrac = np.maximum(0, self.algParams['startLap'] - 1.0/nBatch)
    if lapFrac > 0:
      # When restarting an existing run,
      #  need to start with last update for final batch from previous lap
      DataIterator.lapID = int(np.ceil(lapFrac)) - 1
      DataIterator.curLapPos = nBatch - 2
      iterid = int(nBatch * lapFrac) - 1
    return iterid, lapFrac
  
  ######################################################### Verify memoized ELBO
  #########################################################
  def verifyELBOTracking(self, hmodel, SS, evBound=None, **kwargs):
    ''' Verify that current aggregated SS consistent with sum over all batches
    '''
    if self.doDebugVerbose():
      self.print_msg('>>>>>>>> BEGIN double-check @ lap %.2f' % (self.lapFrac))

    if evBound is None:
      evBound = hmodel.calc_evidence(SS=SS)

    for batchID in range(len(self.SSmemory.keys())):
      SSchunk = self.load_batch_suff_stat_from_memory(batchID, doCopy=1)
      if batchID == 0:
        SS2 = SSchunk.copy()
      else:
        SS2 += SSchunk
    evCheck = hmodel.calc_evidence(SS=SS2)

    if self.algParams['debug'].count('quiet') == 0:
      print '% 14.8f evBound from agg SS' % (evBound)
      print '% 14.8f evBound from sum over SSmemory' % (evCheck)
    if self.algParams['debug'].count('interactive'):
      isCorrect = np.allclose(SS.getCountVec(), SS2.getCountVec()) \
                  and np.allclose(evBound, evCheck)
      if not isCorrect:
        from IPython import embed; embed()
    else:
      assert np.allclose(SS.getCountVec(), SS2.getCountVec())
      assert np.allclose(evBound, evCheck)


    if self.doDebugVerbose():
      self.print_msg('<<<<<<<< END   double-check @ lap %.2f' % (self.lapFrac))

