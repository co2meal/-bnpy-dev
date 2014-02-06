'''
MemoizedOnlineVBLearnAlg.py

Implementation of Memoized Online VB (moVB) learn alg for bnpy models
'''
import numpy as np
from LearnAlg import LearnAlg
from bnpy.util import isEvenlyDivisibleFloat
import logging
from collections import defaultdict
import BirthMove
import MergeMove
import joblib
import os

Log = logging.getLogger('bnpy')


class MemoizedOnlineVBLearnAlg(LearnAlg):

  def __init__( self, **kwargs):
    ''' Creates memoized VB learning algorithm object,
          including specialized internal fields to hold "memoized" statistics
    '''
    super(type(self),self).__init__(**kwargs)
    self.SSmemory = dict()
    self.LPmemory = dict()
    if self.hasMove('merge'):
      self.MergeLog = list()
    if self.hasMove('birth'):
      # Track subsampled data aggregated across batches
      self.targetDataList = list()
      # Track the components freshly added in current lap
      self.BirthCompIDs = list()
      self.BirthInfoCurLap = list()
      # Track the number of laps since birth last attempted
      #  at each component, to encourage trying diversity
      self.LapsSinceLastBirth = defaultdict(int)

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
    # Define how much of data we see at each mini-batch
    nBatch = float(DataIterator.nBatch)
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

    # memoLPkeys : keep list of params that should be retained across laps
    self.memoLPkeys = hmodel.allocModel.get_keys_for_memoized_local_params()
    mPairIDs = None

    SS = None
    isConverged = False
    prevBound = -np.inf
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
      
      # Birth moves!
      if self.hasMove('birth') and iterid > 0:
        hmodel, SS = self.run_birth_move(hmodel, Dchunk, SS, lapFrac)

      if self.hasMove('merge') and not self.algParams['merge']['doAllPairs']:
        if self.isFirstBatch(lapFrac):
          if self.hasMove('birth'):
            compIDs = self.BirthCompIDs
          else:
            compIDs = []
          mPairIDs = MergeMove.preselect_all_merge_candidates(hmodel, SS, 
                           randstate=self.PRNG, compIDs=compIDs,
                           **self.algParams['merge'])

      # E step
      if batchID in self.LPmemory:
        oldLPchunk = self.load_batch_local_params_from_memory(batchID)
        LPchunk = hmodel.calc_local_params(Dchunk, oldLPchunk, **self.algParamsLP)
      else:
        LPchunk = hmodel.calc_local_params(Dchunk, **self.algParamsLP)

      # Collection of target data for birth
      if self.hasMove('birth'):
        self.subsample_data_for_birth(Dchunk, LPchunk, lapFrac)

      # Suff Stat step
      if batchID in self.SSmemory:
        SSchunk = self.load_batch_suff_stat_from_memory(batchID, SS.K)
        SS -= SSchunk

      SSchunk = hmodel.get_global_suff_stats(Dchunk, LPchunk,
                       doPrecompEntropy=True, 
                       doPrecompMergeEntropy=self.hasMove('merge'),
                       mPairIDs=mPairIDs,
                       )
      
      if SS is None:
        SS = SSchunk.copy()
      else:
        assert SSchunk.K == SS.K
        SS += SSchunk

      # Store batch-specific stats to memory
      if self.algParams['doMemoizeLocalParams']:
        self.save_batch_local_params_to_memory(batchID, LPchunk)          
      self.save_batch_suff_stat_to_memory(batchID, SSchunk)  

      # Handle removing "extra mass" of fresh components
      #  to make SS have size exactly consistent with entire dataset
      if self.hasMove('birth') and self.isLastBatch(lapFrac):
        hmodel, SS = self.onBirthLastBatchRemoveExtraMass(hmodel, SS)

      # ELBO calc
      self.verify_suff_stats(Dchunk, SS, lapFrac)
      evBound = hmodel.calc_evidence(SS=SS)

      # Merge move!      
      if self.hasMove('merge') and isEvenlyDivisibleFloat(lapFrac, 1.):
        hmodel, SS, evBound = self.run_merge_move(hmodel, SS, evBound, mPairIDs)

      # Save and display progress
      self.add_nObs(Dchunk.nObs)
      self.save_state(hmodel, iterid, lapFrac, evBound)
      self.print_state(hmodel, iterid, lapFrac, evBound)

      # Check for Convergence!
      #  evBound will increase monotonically AFTER first lap of the data 
      #  verify_evidence will warn if bound isn't increasing monotonically
      if lapFrac > self.algParams['startLap'] + 1.0:
        isConverged = self.verify_evidence(evBound, prevBound, lapFrac)
        if isConverged and lapFrac > 5 and not self.hasMove('birth'):
          break
      prevBound = evBound

    # Finally, save, print and exit
    if isConverged:
      msg = "converged."
    else:
      msg = "max passes thru data exceeded."
    self.save_state(hmodel, iterid, lapFrac, evBound, doFinal=True) 
    self.print_state(hmodel, iterid, lapFrac,evBound,doFinal=True,status=msg)
    return None, self.buildRunInfo(evBound, msg)

  def verify_suff_stats(self, Dchunk, SS, lap):
    ''' Run-time checks to make sure the suff stats
        have expected values
    '''
    SSfile = os.path.join(self.savedir, 'SSdump-Lap%03d.dat' % (lap))
    if self.isLastBatch(lap):
      joblib.dump(SS, SSfile)
    if hasattr(Dchunk, 'nDocTotal') and Dchunk.nDocTotal < 4000:
      if self.hasMove('birth') and len(self.BirthCompIDs) > 0:
        if self.algParams['birth']['earlyLap'] > 0:
          pass
        elif lap < np.ceil(lap):
          assert SS.nDoc - Dchunk.nDocTotal > -0.001
        else:
          if abs(SS.nDoc - Dchunk.nDocTotal) > 0.01:
            print "WARNING @ lap %.2f | SS.nDoc=%d, nDocTotal=%d" % (lap, SS.nDoc, Dchunk.nDocTotal)
          assert abs(SS.nDoc - Dchunk.nDocTotal) < 0.01
      elif lap >= 1.0:
        assert abs(SS.nDoc - Dchunk.nDocTotal) < 0.01

    if hasattr(SS, 'N'):
      if not np.all(SS.N >= -1e-9):
        raise ValueError('N should be >= 0!')
      SS.N[SS.N < 0] = 0

  ######################################################### Load from memory
  #########################################################
  def load_batch_suff_stat_from_memory(self, batchID, K):
    ''' Load the suff stats stored in memory for provided batchID
        Returns
        -------
        SSchunk : bnpy SuffStatDict object for batchID
    '''
    SSchunk = self.SSmemory[batchID]
    # Successful merges from the previous lap must be "replayed"
    #  on the memoized suff stats
    if self.hasMove('birth'):
      Kextra = len(self.BirthCompIDs)
      if Kextra > 0:
        SSchunk.insertEmptyComps(Kextra)      
    if self.hasMove('merge'): 
      for MInfo in self.MergeLog:
        kA = MInfo['kA']
        kB = MInfo['kB']
        if kA < SSchunk.K and kB < SSchunk.K:
          SSchunk.mergeComps(kA, kB)
    if self.hasMove('birth') and self.algParams['birth']['earlyLap'] > 0:
      Kextra = K - SSchunk.K
      if Kextra > 0: # early births
        SSchunk.insertEmptyComps(Kextra)
    if self.hasMove('merge'): 
      if SSchunk.hasMergeTerms():
        SSchunk.setMergeFieldsToZero()
    assert SSchunk.K == K
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
    if self.hasMove('birth') and LPchunk is not None:
      if len(self.BirthCompIDs) > 0:
        LPchunk = None # Forget the old LPchunk!
    if self.hasMove('merge') and LPchunk is not None:
      for MInfo in self.MergeLog:
        kA = MInfo['kA']
        kB = MInfo['kB']
        for key in self.memoLPkeys:
          LPchunk[key][:,kA] = LPchunk[key][:,kA] + LPchunk[key][:,kB]
          LPchunk[key] = np.delete(LPchunk[key], kB, axis=1)
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
    allkeys = LPchunk.keys()
    for key in allkeys:
      if key not in self.memoLPkeys:
        del LPchunk[key]
    if len(LPchunk.keys()) > 0:
      self.LPmemory[batchID] = LPchunk
    else:
      self.LPmemory[batchID] = None


  ######################################################### Birth moves!
  #########################################################
  def run_birth_move(self, hmodel, Dchunk, SS, lapFrac):
    ''' Run birth moves on hmodel.
        Internally handles 
          * creating new components, 
          * suff stat bookkeeping, etc.

        On first batch of data in current lap, 
          * new components created, using data in self.targetDataList
          * target components for births on next lap selected

        On last batch in current lap,
          * suff stats adjusted to remove 'extra mass'

        Returns
        -------
        hmodel : bnpy HModel, with (possibly) new components
        SS : bnpy SuffStatBag, with (possibly) new components
    '''
    # Determine whether to run birth now (at current lap)
    if not self.do_birth_at_lap(lapFrac):
      if self.isFirstBatch(lapFrac):
        Log.info("BIRTH skipped. Exceeded target fraction of total laps.")
        self.BirthCompIDs = list()
        self.BirthInfoCurLap = list()
      return hmodel, SS

    if self.isFirstBatch(lapFrac):
      self.BirthCompIDs = list()
      self.BirthInfoCurLap = list()
      hmodel, SS = self.onBirthFirstBatchCreateNewComps(hmodel, SS)
      # Prepare for the next lap birth moves,
      #   by choosing the next target components
      if self.do_birth_at_lap(lapFrac + 1):
        self.targetDataList = self.onBirthFirstBatchPrepForNextLapBirth(SS)
      else:
        self.targetDataList = list()

    elif lapFrac <= self.algParams['birth']['earlyLap']:
      hmodel, SS = self.onBirthEarlyBatchCreateNewComps(hmodel, SS, Dchunk)

    return hmodel, SS

  def onBirthEarlyBatchCreateNewComps(self, hmodel, SS, Dchunk):
    ''' Returns hmodel, SS with (potentially) several new components

       Internal Updates
       ----------------
         BirthInfoCurLap : list of MoveInfo from each birth attempted
         BirthCompIDs : list of all comp IDs added from all births
         LapsSinceLastBirth : dict, compID -> num. of laps since last birth

       Returns
       -------
         hmodel : 
         SS : bnpy suff stats, with same number of components as hmodel
              if new components were added, SS will have extra mass
              because it contains 2 interpretations of targetData:
               1) mostly assigned to single component ktarget
               2) assigned to brand-new fresh components 
    '''
    if SS is None:
      return hmodel, SS

    Dmax = self.algParams['birth']['maxTargetSize']
    docMask = self.PRNG.permutation(Dchunk.nDoc)[:Dmax]
    TargetData = Dchunk.select_subset_by_mask(docMask=docMask,
                                                doTrackFullSize=False)
    assert TargetData.nDoc <= Dmax

    if TargetData.nDoc < self.algParams['birth']['minTargetSize']:
      msg = "target data too small"
    else:
      hmodel, SS, MoveInfo = BirthMove.run_birth_move(
                 hmodel, TargetData, SS, randstate=self.PRNG, 
                 **self.algParams['birth'])
      msg = MoveInfo['msg']
      if MoveInfo['didAddNew']:
        self.BirthInfoCurLap.append(MoveInfo)
        for kk in MoveInfo['birthCompIDs']:
          self.LapsSinceLastBirth[kk] = -1
        self.BirthCompIDs.extend(MoveInfo['birthCompIDs'])

    self.print_msg( "EARLY %s" % (msg) )
    return hmodel, SS

  def onBirthFirstBatchCreateNewComps(self, hmodel, SS):
    ''' Returns hmodel, SS with (potentially) several new components

       Internal Updates
       ----------------
         BirthInfoCurLap : list of MoveInfo from each birth attempted
         BirthCompIDs : list of all comp IDs added from all births
         LapsSinceLastBirth : dict, compID -> num. of laps since last birth

       Returns
       -------
         hmodel : 
         SS : bnpy suff stats, with same number of components as hmodel
              if new components were added, SS will have extra mass
              because it contains 2 interpretations of targetData:
               1) mostly assigned to single component ktarget
               2) assigned to brand-new fresh components 
    '''
    # Run birth moves on target data!
    nMoves = len(self.targetDataList)
    for moveID, tInfoDict in enumerate(self.targetDataList):
      # Unpack data for current move
      ktarget = tInfoDict['ktarget']
      targetData = tInfoDict['Data']
      if targetData is not None:
        pass
        #self.print_msg("------------------ Target Dataset")
        #self.print_msg(targetData.get_text_summary())
      if ktarget is None or targetData is None:
        msg = tInfoDict['msg']
      # Verify targetData large enough that birth would be productive
      elif targetData.nObs < self.algParams['birth']['minTargetObs']:
        msg = "BIRTH skipped. Target data too small (size %d)"
        msg = msg % (targetData.nObs)
      elif hasattr(targetData, 'nDoc') and targetData.nDoc < self.algParams['birth']['minTargetSize']:
          msg = "BIRTH skipped. Target data too small (size %d)"
          msg = msg % (targetData.nDoc)

      else:
        hmodel, SS, MoveInfo = BirthMove.run_birth_move(
                 hmodel, targetData, SS, randstate=self.PRNG, 
                 ktarget=ktarget, **self.algParams['birth'])
        msg = MoveInfo['msg']
        if MoveInfo['didAddNew']:
          self.BirthInfoCurLap.append(MoveInfo)
          for kk in MoveInfo['birthCompIDs']:
            self.LapsSinceLastBirth[kk] = -1
          self.BirthCompIDs.extend(MoveInfo['birthCompIDs'])
      self.print_msg( "%d/%d %s" % (moveID+1, nMoves, msg) )
    return hmodel, SS

  def onBirthLastBatchRemoveExtraMass(self, hmodel, SS):
    ''' Returns updated suff stats with extra mass removed.
    '''
    didChangeSS = False
    for MoveInfo in self.BirthInfoCurLap:
      freshSS = MoveInfo['freshSS']
      birthCompIDs = MoveInfo['birthCompIDs']
      for bb in birthCompIDs:
        assert bb in self.BirthCompIDs
      SS.subtractSpecificComps(freshSS, birthCompIDs)
      didChangeSS = True
    if didChangeSS:
      hmodel.update_global_params(SS)
    return hmodel, SS


  def onBirthFirstBatchPrepForNextLapBirth(self, SS):
    ''' Return list of dicts, each one has info for one birth move to attempt

        Prepares for next lap's birth moves, by
          selecting which current model component to target

        Returns
        --------
        targetList : list of dictionaries
    '''
    # Create empty list to aggregate subsampled data
    targetList = list()

    # Update counter for which components haven't been updated in a while
    for kk in range(SS.K):
      self.LapsSinceLastBirth[kk] += 1

    # Ignore components that have just been added to the model.
    excludeList = [kk for kk in self.BirthCompIDs]

    # For each birth move, select the target comp
    for posID in range(self.algParams['birth']['birthPerLap']):
      try:
        ktarget = BirthMove.select_birth_component(SS, randstate=self.PRNG,
                          excludeList=excludeList, doVerbose=False,
                          lapsSinceLastBirth=self.LapsSinceLastBirth,
                          **self.algParams['birth'])
        self.LapsSinceLastBirth[ktarget] = 0
        excludeList.append(ktarget)
        tInfoDict = dict(ktarget=ktarget, Data=None)
        targetList.append(tInfoDict)
      except BirthMove.BirthProposalError, e:
        tInfoDict = dict(ktarget=None, Data=None, msg=str(e))
        targetList.append(tInfoDict)
    return targetList

  def subsample_data_for_birth(self, Dchunk, LPchunk, lapFrac):
    ''' Incrementally build-up a target dataset to use as basis for BirthMove
        Calling this method updates the internal data objects.

        Args
        -------
        Dchunk : data object to subsample from
        LPchunk : local parameters for Dchunk

        Internal Updates
        -------
        targetDataList : list of dicts, each with info for one intended birth 
            each entry has fields:
                * ktarget : integer id of component targeted by this move
                * Data : bnpy.data.Data object, target dataset for this move

        Returns
        -------
        None (all updates happen to internal data structures)
    '''
    import BirthMove
    if not self.do_birth_at_lap(lapFrac):
      return
    for tInfoDict in self.targetDataList:

      # Skip this move if component selection failed
      if tInfoDict['ktarget'] is None:
        continue

      birthParams = dict(**self.algParams['birth'])
      # Skip this move if enough data has been collected
      if tInfoDict['Data'] is not None:
        if hasattr(tInfoDict['Data'], 'nDoc'):
          if tInfoDict['Data'].nDoc >= self.algParams['birth']['maxTargetSize']:
            continue
          birthParams['maxTargetSize'] -= tInfoDict['Data'].nDoc
        else:
          if tInfoDict['Data'].nObs >= self.algParams['birth']['maxTargetObs']:
            continue

      # Sample data from current batch, if more is needed
      targetData = BirthMove.subsample_data(Dchunk, LPchunk,
                          tInfoDict['ktarget'], randstate=self.PRNG,
                          **birthParams)
      # Update Data for current entry in self.targetDataList
      if targetData is None:
        if tInfoDict['Data'] is None:
          tInfoDict['msg'] = "TargetData: No samples for target comp found."
      else:
        if tInfoDict['Data'] is None:
          tInfoDict['Data'] = targetData
        else:
          tInfoDict['Data'].add_data(targetData)
        tInfoDict['msg'] = "TargetData: nObs %d" % (tInfoDict['Data'].nObs)


  ######################################################### Merge moves!
  #########################################################
  def run_merge_move(self, hmodel, SS, evBound, mPairIDs=None):
    if self.algParams['merge']['version'] > 0:
      return self.run_merge_move_NEW(hmodel, SS, evBound, mPairIDs)
    else:
      return self.run_merge_move_OLD(hmodel, None, SS, evBound)

  def run_merge_move_NEW(self, hmodel, SS, evBound, mPairIDs=None):
    ''' Run (potentially many) merge moves on hmodel,
          performing necessary bookkeeping to
            (1) avoid trying the same merge twice
            (2) avoid merging a component that has already been merged,
                since the precomputed entropy will no longer be correct.
        Returns
        -------
        hmodel : bnpy HModel, with (possibly) some merged components
        SS : bnpy SuffStatBag, with (possibly) merged components
        evBound : correct ELBO for returned hmodel
                  guaranteed to be at least as large as input evBound    
    '''
    from MergeMove import run_many_merge_moves

    if self.hasMove('birth') and len(self.BirthCompIDs) > 0:
      compList = [x for x in self.BirthCompIDs] # need a copy
    else:
      compList = list()

    nMergeTrials = self.algParams['merge']['mergePerLap']

    hmodel, SS, newEvBound, MTracker = run_many_merge_moves(
                        hmodel, None, SS, evBound=evBound,
                        mPairIDs=mPairIDs, 
                        randstate=self.PRNG, nMergeTrials=nMergeTrials,
                        compList=compList, savedir=self.savedir,
                        **self.algParams['merge'])

    msg = 'MERGE: %3d/%3d accepted.' % (MTracker.nSuccess, MTracker.nTrial)
    if MTracker.nSuccess > 0:
      msg += ' ev improved + %.3e' % (newEvBound - evBound)
    if self.algParams['merge']['doVerbose'] >= 0:
      self.print_msg(msg)
    
    if self.algParams['merge']['doVerbose'] > 0:
      for msg in MTracker.InfoLog:
        self.print_msg(msg)

    # ------ Adjust indexing for counter that determines which comp to target
    if self.hasMove('birth'):
      for kA, kB in MTracker.acceptedIDs:
        self._adjustLapsSinceLastBirthForMerge(MTracker, kA, kB)
    # ------ Record accepted moves, so can adjust memoized stats later
    self.MergeLog = list()
    for kA, kB in MTracker.acceptedIDs:
      self.MergeLog.append(dict(kA=kA, kB=kB))
    # ------ Reset all precalculated merge terms
    if SS.hasMergeTerms():
      SS.setMergeFieldsToZero()

    return hmodel, SS, newEvBound

  def _adjustLapsSinceLastBirthForMerge(self, MTracker, kA, kB):
    ''' Adjust internal tracking of laps since birth
    '''
    compList = self.LapsSinceLastBirth.keys()
    newDict = defaultdict(int)
    for kk in compList:
      if kk == kA:
        newDict[kA] = np.maximum(self.LapsSinceLastBirth[kA], self.LapsSinceLastBirth[kB])
      elif kk < kB:
        newDict[kk] = self.LapsSinceLastBirth[kk]
      elif kk > kB:
        newDict[kk-1] = self.LapsSinceLastBirth[kk]
    self.LapsSinceLastBirth = newDict


  def run_merge_move_OLD(self, hmodel, Data, SS, evBound):
    ''' Run (potentially many) merge moves on hmodel,
          performing necessary bookkeeping to
            (1) avoid trying the same merge twice
            (2) avoid merging a component that has already been merged,
                since the precomputed entropy will no longer be correct.
        Returns
        -------
        hmodel : bnpy HModel, with (possibly) some merged components
        SS : bnpy SuffStatBag, with (possibly) merged components
        evBound : correct ELBO for returned hmodel
                  guaranteed to be at least as large as input evBound    
    '''
    import OldMergeMove
    self.MergeLog = list() # clear memory of recent merges!
    excludeList = list()
    excludePairs = defaultdict(lambda:set())
    nMergeAttempts = self.algParams['merge']['mergePerLap']
    trialID = 0
    while trialID < nMergeAttempts:

      # Synchronize contents of the excludeList and excludePairs
      # So that comp excluded in excludeList (due to accepted merge)
      #  is automatically contained in the set of excluded pairs 
      for kx in excludeList:
        for kk in excludePairs:
          excludePairs[kk].add(kx)
          excludePairs[kx].add(kk)

      for kk in excludePairs:
        if len(excludePairs[kk]) > hmodel.obsModel.K - 2:
          if kk not in excludeList:
            excludeList.append(kk)

      if len(excludeList) > hmodel.obsModel.K - 2:
        Log.info('Merge Done. No more options to try!')
        break # when we don't have any more comps to merge
        
      if self.hasMove('birth') and len(self.BirthCompIDs) > 0:
        kA = self.BirthCompIDs.pop()
        if kA in excludeList:
          continue
      else:
        kA = None

      hmodel, SS, evBound, MoveInfo = OldMergeMove.run_merge_move(
                 hmodel, None, SS, evBound, randstate=self.PRNG,
                 excludeList=excludeList, excludePairs=excludePairs,
                 kA=kA, **self.algParams['merge'])
      trialID += 1
      self.print_msg(MoveInfo['msg'])

      # Begin Bookkeeping!
      if 'kA' in MoveInfo and 'kB' in MoveInfo:
        kA = MoveInfo['kA']
        kB = MoveInfo['kB']
        excludePairs[kA].add(kB)
        excludePairs[kB].add(kA)

      if MoveInfo['didAccept']:
        self.MergeLog.append(dict(kA=kA, kB=kB))

        # Adjust excluded lists since components kB+1, kB+2, ... K
        #  have been shifted down by one due to removal of kB
        for kk in range(len(excludeList)):
          if excludeList[kk] > kB:
            excludeList[kk] -= 1

        # Exclude new merged component kA from future attempts        
        #  since precomputed entropy terms involving kA aren't good
        excludeList.append(kA)
    
        # Adjust excluded pairs to remove kB and shift down kB+1, ... K
        newExcludePairs = defaultdict(lambda:set())
        for kk in excludePairs.keys():
          ksarr = np.asarray(list(excludePairs[kk]))
          ksarr[ksarr > kB] -= 1
          if kk > kB:
            newExcludePairs[kk-1] = set(ksarr)
          elif kk < kB:
            newExcludePairs[kk] = set(ksarr)
        excludePairs = newExcludePairs

        # Adjust internal tracking of laps since birth
        if self.hasMove('birth'):
          compList = self.LapsSinceLastBirth.keys()
          newDict = defaultdict(int)
          for kk in compList:
            if kk == kA:
              newDict[kA] = np.maximum(self.LapsSinceLastBirth[kA], self.LapsSinceLastBirth[kB])
            elif kk < kB:
              newDict[kk] = self.LapsSinceLastBirth[kk]
            elif kk > kB:
              newDict[kk-1] = self.LapsSinceLastBirth[kk]
          self.LapsSinceLastBirth = newDict

    if SS.hasMergeTerms():
      SS.setMergeFieldsToZero()
    return hmodel, SS, evBound
  
