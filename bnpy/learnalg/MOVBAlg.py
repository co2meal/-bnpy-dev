'''
MOVBAlg.py

Implementation of Memoized Online VB (moVB) learn alg for bnpy models
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
from bnpy.birthmove import TargetPlanner, TargetDataSampler, BirthMove
from bnpy.birthmove import BirthLogger, TargetPlannerWordFreq
from bnpy.mergemove import MergeMove, MergePlanner

class MOVBAlg(LearnAlg):

  def __init__( self, **kwargs):
    ''' Creates memoized VB learning algorithm object,
          including specialized internal fields to hold "memoized" statistics
    '''
    super(type(self),self).__init__(**kwargs)
    self.SSmemory = dict()
    self.LPmemory = dict()
    if self.hasMove('merge') or self.hasMove('softmerge'):
      self.MergeLog = list()
    if self.hasMove('birth'):
      # Track the components freshly added in current lap
      self.BirthCompIDs = list()
      self.ModifiedCompIDs = list()
      # Track the number of laps since birth last attempted
      #  at each component, to encourage trying diversity
      self.LapsSinceLastBirth = defaultdict(int)
    if self.hasMove('delete'):
      self.DelMoveSSmemory = dict()
      self.DelMoveLPmemory = dict()

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
    self.ActiveIDVec = np.arange(hmodel.obsModel.K)

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
    mPairIDs = None

    if self.hasMove('birth'):
      doQ = self.algParams['birth']['targetSelectName'].lower().count('anchor')
    else:
      doQ = False
    DeleteInfo = dict()
    PrunePlans = list()
    PruneResults = list()

    BirthPlans = list()
    BirthResults = None
    prevBirthResults = None
    preselectroutine = None
    order = None
    SS = None
    isConverged = False
    prevBound = -np.inf
    self.set_start_time_now()
    doDelete = self.hasMove('delete')
    doPrecompMergeEntropy = 0
    MM = None # Score matrix for merge candidate pairs
    numStuck = 0 # Num consecutive merge attempts without any accepts
    numConvergedInARow = 0
    while DataIterator.has_next_batch():

      # Grab new data
      Dchunk = DataIterator.get_next_batch()
      batchID = DataIterator.batchID
      
      # Update progress-tracking variables
      iterid += 1
      lapFrac = (iterid + 1) * self.lapFracInc
      self.set_random_seed_at_lap(lapFrac)

      # Rearrange the order
      if self.isFirstBatch(lapFrac) and self.hasMove('shuffle'):
        if SS is not None:
          order = np.argsort(-1*SS.N)
          SS.reorderComps(order)
          self.ActiveIDVec = self.ActiveIDVec[order]

      # M step
      if self.isFirstBatch(lapFrac):
        if SS is not None and hasattr(hmodel.obsModel, 'forceSSInBounds'):
          hmodel.obsModel.forceSSInBounds(SS)

      if self.algParams['doFullPassBeforeMstep']:
        if SS is not None and lapFrac > 1.0:
          hmodel.update_global_params(SS)
      else:
        if SS is not None:
          hmodel.update_global_params(SS)
      
      # Birth move : track birth info from previous lap
      if self.isFirstBatch(lapFrac):
        if self.hasMove('birth') and self.do_birth_at_lap(lapFrac - 1.0):
          prevBirthResults = BirthResults
        else:
          prevBirthResults = list()

      if self.hasMove('birth') and doQ:
        if lapFrac <= 1.0:
          A, b, c = Dchunk.to_wordword_cooccur_building_blocks()
          if self.isFirstBatch(lapFrac):
            Q = A
            bb = b
            cc = c
          else:
            Q += A
            bb += b
            cc += c
          if self.isLastBatch(lapFrac):
            Q = Dchunk._calc_wordword_cooccur(Q, bb, cc)        
        else:
          Dchunk.Q = Q

      # Birth move : create new components
      if self.hasMove('birth') and self.do_birth_at_lap(lapFrac):
        if self.doBirthWithPlannedData(lapFrac):
          hmodel, SS, BirthResults = self.birth_create_new_comps(
                                            hmodel, SS, BirthPlans,
                                                        lapFrac=lapFrac)

        if self.doBirthWithDataFromCurrentBatch(lapFrac):
          hmodel, SS, BirthRes = self.birth_create_new_comps(
                                            hmodel, SS, Data=Dchunk,
                                                        lapFrac=lapFrac)
          BirthResults.extend(BirthRes)

        self.BirthCompIDs = self.birth_get_all_new_comps(BirthResults)
        self.ModifiedCompIDs = self.birth_get_all_modified_comps(BirthResults)
      else:
        BirthResults = list()
        self.BirthCompIDs = list() # no births = no new components
        self.ModifiedCompIDs = list()


      # Prep for Merge
      if self.hasMove('merge'):
        preselectroutine = self.algParams['merge']['preselectroutine']
        mergeELBOTrackMethod = self.algParams['merge']['mergeELBOTrackMethod']
        mergeStartLap = self.algParams['merge']['mergeStartLap']
        if self.isFirstBatch(lapFrac) and lapFrac > 1:
          SS.setMergeFieldsToZero()

        if self.isFirstBatch(lapFrac) and lapFrac >= mergeStartLap:
          if mergeELBOTrackMethod == 'exact':
            # Update tracked
            if preselectroutine == 'wholeELBO' and MM is not None:
              for Info in self.MergeLog:
                MM = np.delete(MM, Info['kB'], axis=0)
                MM = np.delete(MM, Info['kB'], axis=1)
                MM[Info['kA'], :] = 0
                MM[:, Info['kA']] = 0
              if len(BirthResults) > 0:
                Korig = MM.shape[0]
                Mnew = np.zeros((SS.K, SS.K))
                Mnew[:Korig, :Korig] = MM
                MM = Mnew
              rLap = self.algParams['merge']['mergeScoreRefreshLap']
              if MM is not None and np.floor(lapFrac) % rLap == 0:
                MM.fill(0) # Refresh!
            mPairIDs, MM = MergePlanner.preselect_candidate_pairs(hmodel, SS,
                                            randstate=self.PRNG,
                                            returnScoreMatrix=1,
                                            M=MM,
                                            **self.algParams['merge'])
            doPrecompMergeEntropy = 1 # explicitly precomp all O(K^2) pairs
          else:
            doPrecompMergeEntropy = 2 # need only precomp O(K) stats
      elif self.hasMove('softmerge'):
        if self.isFirstBatch(lapFrac) and lapFrac > 1:
          SS.setMergeFieldsToZero()
        mergeStartLap = self.algParams['softmerge']['mergeStartLap']
        doPrecompMergeEntropy = 2

      ## Reset selection terms to zero
      if self.isFirstBatch(lapFrac):
        if SS is not None and SS.hasSelectionTerms():
          SS._SelectTerms.setAllFieldsToZero()

      # E step
      if batchID in self.LPmemory:
        oldLPchunk = self.load_batch_local_params_from_memory(
                                           batchID, prevBirthResults,    
                                           PruneResults)
        LPchunk = hmodel.calc_local_params(Dchunk, oldLPchunk,
                                           **self.algParamsLP)
      else:
        LPchunk = hmodel.calc_local_params(Dchunk, **self.algParamsLP)

      # Collect target data for birth
      if self.hasMove('birth') and self.do_birth_at_lap(lapFrac+1.0):
        if self.isFirstBatch(lapFrac):
          BirthPlans = self.birth_plan_targets_for_next_lap(
                                Dchunk, hmodel, SS, LPchunk, BirthResults)
        BirthPlans = self.birth_collect_target_subsample(
                                Dchunk, hmodel, LPchunk, BirthPlans, lapFrac)
      else:
        BirthPlans = list()

      # Suff Stat step
      if batchID in self.SSmemory:
        oldSSchunk = self.load_batch_suff_stat_from_memory(batchID, SS.K, 
                                                        prevBirthResults,
                                                        BirthResults,
                                                        PruneResults,
                                                        order)
        SS -= oldSSchunk
      else:
        # Record this batch as updated to reflect all current birth moves
        for MInfo in BirthResults:
          if 'bchecklist' not in MInfo:
            MInfo['bchecklist'] = np.zeros(self.nBatch)
          MInfo['bchecklist'][batchID] = 2 # mark as resolved

      SSchunk = hmodel.get_global_suff_stats(Dchunk, LPchunk,
                       doPrecompEntropy=True, 
                       doPrecompMergeEntropy=doPrecompMergeEntropy,
                       mPairIDs=mPairIDs,
                       preselectroutine=preselectroutine
                       )
      
      if SS is None:
        SS = SSchunk.copy()
      else:
        assert SSchunk.K == SS.K
        SS += SSchunk

      # Handle removing "extra mass" of fresh components
      #  to make SS have size exactly consistent with entire dataset
      if self.hasMove('birth') and self.isLastBatch(lapFrac):
        hmodel, SS = self.birth_remove_extra_mass(hmodel, SS, BirthResults)
      
      # Store batch-specific stats to memory
      if self.algParams['doMemoizeLocalParams']:
        self.save_batch_local_params_to_memory(batchID, LPchunk)          
      self.save_batch_suff_stat_to_memory(batchID, SSchunk)


      # Merge move!      
      if self.hasMove('merge') and self.isLastBatch(lapFrac) \
                               and lapFrac > mergeStartLap:       
        hmodel.update_global_params(SS)
        evBound = hmodel.calc_evidence(SS=SS)
        if mergeELBOTrackMethod == 'fastBound':
          mPairIDs, MM = MergePlanner.preselect_candidate_pairs(hmodel, SS,
                           randstate=self.PRNG, doLimitNumPairs=0,
                           returnScoreMatrix=1,
                           **self.algParams['merge'])
        assert mPairIDs is not None
        hmodel, SS, evBound = self.run_many_merge_moves(hmodel, SS, evBound,
                                                        mPairIDs, MM, lapFrac)
      elif self.hasMove('softmerge') and self.isLastBatch(lapFrac) \
                                     and lapFrac > mergeStartLap:
        hmodel.update_global_params(SS)
        evBound = hmodel.calc_evidence(SS=SS)
        hmodel, SS, evBound = self.run_softmerge_moves(hmodel, SS, evBound,
                                                       lapFrac, LPchunk)
          
      else:
        evBound = hmodel.calc_evidence(SS=SS)

      # Save and display progress
      hmodel.ActiveIDVec = self.ActiveIDVec
      self.add_nObs(Dchunk.get_size())
      self.save_state(hmodel, SS, iterid, lapFrac, evBound)
      self.print_state(hmodel, SS, iterid, lapFrac, evBound)
      self.eval_custom_func(lapFrac, hmodel=hmodel, SS=SS, Dchunk=Dchunk, 
                                     LPchunk=LPchunk, batchID=batchID,
                                     SSchunk=SSchunk, learnAlg=self,
                                     evBound=evBound,
                                     BirthResults=BirthResults,
                                     prevBirthResults=prevBirthResults)

      # Check for Convergence!
      #  evBound will increase monotonically AFTER first lap of the data 
      #  verify_evidence will warn if bound isn't increasing monotonically
      if lapFrac > self.algParams['startLap'] + 1.0 + 1.0 / nBatch:
        isConvergedCurrent = self.verify_evidence(evBound, prevBound, lapFrac)
        if isConvergedCurrent:
          numConvergedInARow += 1.0 / nBatch
        else:
          numConvergedInARow = 0
 
        if lapFrac > 5 and numConvergedInARow > self.algParams['convergeDuration']:
          isConverged = 1

      if self.hasMove('birth'):
        doQuit = False # never quit early for births
      elif self.hasMove('merge') or self.hasMove('softmerge'):
        doQuit = False
        if self.hasMove('merge'):
          nStuckBeforeQuit = self.algParams['merge']['mergeNumStuckBeforeQuit']
        else:
          nStuckBeforeQuit = self.algParams['softmerge']['mergeNumStuckBeforeQuit']

        if self.isLastBatch(lapFrac) and lapFrac > mergeStartLap:
          if len(self.MergeLog) == 0:
            numStuck += 1
          else:
            numStuck = 0
          doQuit = isConverged and numStuck >= nStuckBeforeQuit
      else:
        doQuit = isConverged
      if doQuit:
        break
      prevBound = evBound
      #.................................................... end loop over data

    # Finally, save, print and exit
    if isConverged:
      msg = "converged."
    else:
      msg = "max passes thru data exceeded."
    self.save_state(hmodel, SS, iterid, lapFrac, evBound, doFinal=True) 
    self.print_state(hmodel, SS, iterid, lapFrac, evBound, 
                     doFinal=True, status=msg)

    self.eval_custom_func(lapFrac, hmodel=hmodel, SS=SS, Dchunk=Dchunk,
                                   LPchunk=LPchunk, batchID=batchID,
                                   SSchunk=SSchunk, learnAlg=self,
                                   evBound=evBound,
                                   isFinal=True,
                                   )

    self.SS = SS # hack so we can examine global suff stats
    # Births and merges require copies of original model object
    #  we need to make sure original reference has updated parameters, etc.
    if id(origmodel) != id(hmodel):
      origmodel.allocModel = hmodel.allocModel
      origmodel.obsModel = hmodel.obsModel
    return None, self.buildRunInfo(evBound, msg)

  ######################################################### Load from memory
  #########################################################
  def load_batch_suff_stat_from_memory(self, batchID, K, 
                                       prevBirthResults=None, 
                                       BirthResults=None,
                                       PruneResults=list(),
                                       order=None,
                                       doCopy=0):
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

    # "Replay" accepted softmerges from end of previous lap 
    if self.hasMove('softmerge'): 
      for MInfo in self.MergeLog:
        SSchunk.multiMergeComps(MInfo['kdel'], MInfo['alph'])
    
    # "Replay" accepted merges from end of previous lap 
    if self.hasMove('merge'): 
      for MInfo in self.MergeLog:
        kA = MInfo['kA']
        kB = MInfo['kB']
        if kA < SSchunk.K and kB < SSchunk.K and SSchunk.K == MInfo['Korig']:
          SSchunk.mergeComps(kA, kB)
    if SSchunk.hasMergeTerms():
      SSchunk.setMergeFieldsToZero()

    # "replay" any shuffling/reordering that happened
    if self.hasMove('shuffle') and order is not None:
      SSchunk.reorderComps(order)

    # "replay" generic and batch-specific births from this lap
    if self.hasMove('birth'):   
      Kextra = K - SSchunk.K
      if Kextra > 0:
        SSchunk.insertEmptyComps(Kextra)
    assert SSchunk.K == K

    # Adjust / replace terms related to expansion
    MoveInfoList = list()
    if prevBirthResults is not None:
      MoveInfoList.extend(prevBirthResults)
    if BirthResults is not None:
      MoveInfoList.extend(BirthResults)
    for MInfo in MoveInfoList:
      if 'AdjustInfo' in MInfo and MInfo['AdjustInfo'] is not None:
        if 'bchecklist' not in MInfo:
          MInfo['bchecklist'] = np.zeros(int(self.nBatch))
        bchecklist = MInfo['bchecklist']
        if bchecklist[batchID] > 0:
          continue
        # Do the adjustment work
        for key in MInfo['AdjustInfo']:
          if hasattr(SSchunk, key):
            Kmax = MInfo['AdjustInfo'][key].size
            arr = getattr(SSchunk, key)
            arr[:Kmax] += SSchunk.nDoc *  MInfo['AdjustInfo'][key]
            SSchunk.setField(key, arr, dims=SSchunk._FieldDims[key])
          elif SSchunk.hasELBOTerm(key):
            Kmax = MInfo['AdjustInfo'][key].size
            arr = SSchunk.getELBOTerm(key)
            arr[:Kmax] += SSchunk.nDoc *  MInfo['AdjustInfo'][key]
            SSchunk.setELBOTerm(key, arr, dims='K')

        # Record visit, so adjustment is only done once
        bchecklist[batchID] = 1
    # Run backwards through results to find most recent ReplaceInfo
    for MInfo in reversed(MoveInfoList):
      if 'ReplaceInfo' in MInfo and MInfo['ReplaceInfo'] is not None:
        if MInfo['bchecklist'][batchID] > 1:
          break # this batch has had replacements done already
        for key in MInfo['ReplaceInfo']:
          if hasattr(SSchunk, key):
            arr = SSchunk.nDoc * MInfo['ReplaceInfo'][key]
            SSchunk.setField(key, arr, dims=SSchunk._FieldDims[key])
          elif SSchunk.hasELBOTerm(key):
            arr = SSchunk.nDoc * MInfo['ReplaceInfo'][key]
            SSchunk.setELBOTerm(key, arr, dims=None)

        MInfo['bchecklist'][batchID] = 2
        break # Stop after the first ReplaceInfo
    return SSchunk  

  def load_batch_local_params_from_memory(self, batchID, 
                                                BirthResults,
                                                PruneResults=list()):
    ''' Load local parameter dict stored in memory for provided batchID
        Ensures "fast-forward" so that all recent merges/births
          are accounted for in the returned LP
        Returns
        -------
        LPchunk : bnpy local parameters dictionary for batchID
    '''
    LPchunk = self.LPmemory[batchID]
    if self.hasMove('prune') and LPchunk is not None:
      for Plan in PruneResults:
        kk = Plan['ktarget']
        for key in self.memoLPkeys:
          LPchunk[key] = np.delete(LPchunk[key], kk, axis=1)
    if self.hasMove('merge') and LPchunk is not None:
      for MInfo in self.MergeLog:
        kA = MInfo['kA']
        kB = MInfo['kB']
        for key in self.memoLPkeys:
          ## Not sure about this old code. Disabled for now...
          #if kB >= LPchunk[key].shape[1]:
          #  # Birth occured in previous lap, after this batch was visited.
          #  return None
          kB_column = LPchunk[key][:,kB]
          LPchunk[key] = np.delete(LPchunk[key], kB, axis=1)
          LPchunk[key][:,kA] = LPchunk[key][:,kA] + kB_column

    # any shuffling/reordering is handled within allocmodel
    return LPchunk

  ######################################################### Save to memory
  #########################################################
  def save_batch_suff_stat_to_memory(self, batchID, SSchunk):
    ''' Store the provided suff stats into the "memory" for later retrieval
    '''
    if SSchunk.hasSelectionTerms():
      del SSchunk._SelectTerms
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


  ######################################################### Birth moves!
  #########################################################
  def doBirthWithPlannedData(self, lapFrac):
    return self.isFirstBatch(lapFrac)

  def doBirthWithDataFromCurrentBatch(self, lapFrac):
    if self.isLastBatch(lapFrac):
      return False
    rem = lapFrac - np.floor(lapFrac)
    isWithinFrac = rem <= self.algParams['birth']['birthBatchFrac'] + 1e-6
    isWithinLimit = lapFrac <= self.algParams['birth']['birthBatchLapLimit'] 
    return isWithinFrac and isWithinLimit

  def birth_create_new_comps(self, hmodel, SS, BirthPlans=list(), Data=None,
                                               lapFrac=0):
    ''' Create new components 

        Returns
        -------
        hmodel : bnpy HModel, either existing model or one with more comps
        SS : bnpy SuffStatBag, either existing SS or one with more comps
        BirthResults : list of dicts, one entry per birth move
    '''
    kwargs = dict(**self.algParams['birth'])
    kwargs.update(**self.algParamsLP)

    if 'birthRetainExtraMass' not in kwargs:
      kwargs['birthRetainExtraMass'] = 1

    if Data is not None:
      targetData = TargetDataSampler.sample_target_data(
                                        Data, model=hmodel, LP=None,
                                        randstate=self.PRNG,
                                        **kwargs)
      Plan = dict(Data=targetData, ktarget=-1, targetWordIDs=[-1])
      BirthPlans = [Plan]
      kwargs['birthRetainExtraMass'] = 0

    nMoves = len(BirthPlans)
    BirthResults = list()

    def isInPlan(Plan, key):
      return key in Plan and Plan[key] is not None

    for moveID, Plan in enumerate(BirthPlans):
      # Unpack data for current move
      ktarget = Plan['ktarget']
      targetData = Plan['Data']
      targetSize = TargetDataSampler.getSize(targetData) 
      # Remember, targetData may be None

      if isInPlan(Plan, 'targetWordIDs'):
        isBad = len(Plan['targetWordIDs']) == 0
      elif isInPlan(Plan, 'targetWordFreq'):
        isBad = False
      else:
        isBad = ktarget is None

      BirthLogger.logStartMove(lapFrac, moveID + 1, len(BirthPlans))
      if isBad or targetData is None:
        msg = Plan['msg']
        BirthLogger.log(msg)
        BirthLogger.log('SKIPPED. TargetData bad.')
      elif targetSize < kwargs['targetMinSize']:
        msg = "SKIPPED. Target data too small. Size %d."
        BirthLogger.log(msg % (targetSize))
      else:
        hmodel, SS, MoveInfo = BirthMove.run_birth_move(
                                           hmodel, SS, targetData, 
                                           randstate=self.PRNG, 
                                           Plan=Plan,
                                           **kwargs)
        if MoveInfo['didAddNew']:
          BirthResults.append(MoveInfo)
          for kk in MoveInfo['birthCompIDs']:
            self.LapsSinceLastBirth[kk] = -1

    return hmodel, SS, BirthResults

  def birth_remove_extra_mass(self, hmodel, SS, BirthResults):
    ''' Adjust hmodel and suff stats to remove the "extra mass"
          added during a birth move to brand-new components.
        After this call, SS should have scale exactly consistent with 
          the entire dataset (all B batches).

        Returns
        -------
        hmodel : bnpy HModel
        SS : bnpy SuffStatBag
    '''
    didChangeSS = False
    for MoveInfo in BirthResults:
      if MoveInfo['didAddNew'] and 'extraSS' in MoveInfo:
        extraSS = MoveInfo['extraSS']
        compIDs = MoveInfo['modifiedCompIDs']
        assert extraSS.K == len(compIDs)
        SS.subtractSpecificComps(extraSS, compIDs)
        didChangeSS = True
    if didChangeSS:
      hmodel.update_global_params(SS)
    return hmodel, SS

  def birth_plan_targets_for_next_lap(self, Data, hmodel, SS, LP,
                                            BirthResults):
    ''' Create plans for next lap's birth moves
    
        Returns
        -------
        BirthPlans : list of dicts, 
                     each entry represents the plan for one future birth move
    '''
    if SS is not None:
      assert hmodel.allocModel.K == SS.K
    K =  hmodel.allocModel.K
    nBirths = self.algParams['birth']['birthPerLap']
    if self.algParams['birth']['targetSelectName'].lower().count('word'):
      Plans = TargetPlanner.select_target_words_MultipleSets(
                            model=hmodel, Data=Data, LP=LP, 
                            nSets=nBirths, randstate=self.PRNG,
                            **self.algParams['birth'])
      return Plans
    elif self.algParams['birth']['targetSelectName'].lower().count('freq'):
      Plans = TargetPlannerWordFreq.MakePlans(
                            Data, hmodel, LP, 
                            nPlans=nBirths, randstate=self.PRNG,
                            **self.algParams['birth'])
      return Plans
    # Update counter for duration since last targeted-birth for each comp
    for kk in range(K):
      self.LapsSinceLastBirth[kk] += 1
    # Ignore components that have just been added to the model.
    excludeList = self.birth_get_all_new_comps(BirthResults)

    # For each birth move, create a "plan"
    BirthPlans = list()
    for posID in range(nBirths):
      try:
        ktarget = TargetPlanner.select_target_comp(
                             K, SS=SS, Data=Data, model=hmodel,
                             randstate=self.PRNG,
                             excludeList=excludeList,
                             lapsSinceLastBirth=self.LapsSinceLastBirth,
                              **self.algParams['birth'])
        self.LapsSinceLastBirth[ktarget] = 0
        excludeList.append(ktarget)
        Plan = dict(ktarget=ktarget,
                    Data=None,
                    targetWordIDs=None, 
                    targetWordFreq=None)
      except BirthMove.BirthProposalError, e:
        # Happens when no component is eligible for selection (all excluded)
        Plan = dict(ktarget=None, 
                    Data=None,
                    targetWordIDs=None, 
                    targetWordFreq=None, 
                    msg=str(e),
                    )
      BirthPlans.append(Plan)
    return BirthPlans

  def birth_collect_target_subsample(self, Dchunk, model, LPchunk, 
                                           BirthPlans, lapFrac):
    ''' Collect subsample of the data in Dchunk, and add that subsample
          to overall targeted subsample stored in input list BirthPlans
        This overall sample is aggregated across many batches of data.
        Data from Dchunk is only collected if more data is needed.

        Returns
        -------
        BirthPlans : list of planned births for the next lap,
                      updated to include data from Dchunk if needed
    '''
    for Plan in BirthPlans:
      # Skip this move if component selection failed
      if Plan['ktarget'] is None and Plan['targetWordIDs'] is None \
                                 and Plan['targetWordFreq'] is None:
        continue

      birthParams = dict(**self.algParams['birth'])

      # Skip collection if have enough data already
      if Plan['Data'] is not None:
        targetSize = TargetDataSampler.getSize(Plan['Data'])
        if targetSize >= birthParams['targetMaxSize']:
            continue
        birthParams['targetMaxSize'] -= targetSize
        # TODO: worry about targetMaxSize when we always keep topK datapoints

      # Sample data from current batch, if more is needed
      targetData, targetInfo = TargetDataSampler.sample_target_data(
                          Dchunk, model=model, LP=LPchunk,
                          targetCompID=Plan['ktarget'],
                          targetWordIDs=Plan['targetWordIDs'],
                          targetWordFreq=Plan['targetWordFreq'],
                          randstate=self.PRNG,
                          return_Info=True,
                          **birthParams)

      # Update Data for current entry in self.targetDataList
      if targetData is None:
        if Plan['Data'] is None:
          Plan['msg'] = "TargetData: No samples for target comp found."
      else:
        if Plan['Data'] is None:
          Plan['Data'] = targetData
          Plan['Info'] = targetInfo
        else:
          Plan['Data'].add_data(targetData)
          if 'dist' in Plan['Info']:
            Plan['Info']['dist'] = np.append(Plan['Info']['dist'],
                                             targetInfo['dist'])
        size = TargetDataSampler.getSize(Plan['Data'])
        Plan['msg'] = "TargetData: size %d" % (size)

      if self.isLastBatch(lapFrac) and 'Info' in Plan:
        if 'dist' in Plan['Info']:
          dist = Plan['Info']['dist']
          sortIDs = np.argsort(dist)[:self.algParams['birth']['targetMaxSize']]
          Plan['Data'] = Plan['Data'].select_subset_by_mask(sortIDs)
          size = TargetDataSampler.getSize(Plan['Data'])
          Plan['msg'] = "TargetData: size %d" % (size)
    return BirthPlans

  def birth_get_all_new_comps(self, BirthResults):
    ''' Returns list of integer ids of all new components added by
          birth moves summarized in BirthResults

        Returns
        -------
        birthCompIDs : list of integers, each entry is index of a new component
    '''
    birthCompIDs = list()
    for MoveInfo in BirthResults:
      birthCompIDs.extend(MoveInfo['birthCompIDs'])
    return birthCompIDs

  def birth_get_all_modified_comps(self, BirthResults):
    ''' Returns list of integer ids of all new components added by
          birth moves summarized in BirthResults

        Returns
        -------
        mCompIDs : list of integers, each entry is index of modified comp
    '''
    mCompIDs = list()
    for MoveInfo in BirthResults:
      mCompIDs.extend(MoveInfo['modifiedCompIDs'])
    return mCompIDs

  def birth_count_new_comps(self, BirthResults):
    ''' Returns total number of new components added by moves 
          summarized by BirthResults
    
        Returns
        -------
        Kextra : int number of components added by given list of moves
    '''
    Kextra = 0
    for MoveInfo in BirthResults:
      Kextra += len(MoveInfo['birthCompIDs'])
    return Kextra


  ######################################################### Merge moves!
  #########################################################

  def run_many_merge_moves(self, hmodel, SS, evBound, mPairIDs, M, lapFrac):
    ''' Run (potentially many) merge moves on hmodel,
          performing necessary bookkeeping to
            (1) avoid trying the same merge twice
            (2) avoid merging a component that has already been merged,
                since the precomputed entropy will no longer be correct.
        Returns
        -------
        hmodel : bnpy HModel, with (possibly) merged components
        SS : bnpy SuffStatBag, with (possibly) merged components
        evBound : correct ELBO for returned hmodel
                  guaranteed to be at least as large as input evBound    
    '''
    if self.algParams['merge']['mergeLogVerbose']:
      import bnpy.mergemove.MergeLogger as MergeLogger
      MergeLogger.logStartMove(lapFrac)

    Korig = SS.K
    hmodel, SS, newEvBound, Info = MergeMove.run_many_merge_moves(
                                       hmodel, SS, evBound, mPairIDs, M=M,
                                       **self.algParams['merge'])

    # ------ Adjust indexing for counter that determines which comp to target
    if self.hasMove('birth'):
      for kA, kB in Info['AcceptedPairs']:
        self._resetLapsSinceLastBirthAfterMerge(kA, kB)

    # ------ Record accepted moves, so can adjust memoized stats later
    self.MergeLog = list()
    for kA, kB in Info['AcceptedPairs']:
      self.ActiveIDVec = np.delete(self.ActiveIDVec, kB, axis=0)
      self.MergeLog.append(dict(kA=kA, kB=kB, Korig=Korig))
      Korig -= 1

    # ------ Reset all precalculated merge terms
    if SS.hasMergeTerms():
      SS.setMergeFieldsToZero()
    return hmodel, SS, newEvBound

  def _resetLapsSinceLastBirthAfterMerge(self, kA, kB):
    ''' Update internal list of LapsSinceLastBirth to reflect accepted merge

        Returns
        ---------
        None. Updates to self.LapsSinceLastBirth happen in-place.
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



  ######################################################### Soft Merge moves
  #########################################################
  def run_softmerge_moves(self, hmodel, SS, evBound, lapFrac, LPchunk):
    ''' Run (potentially many) softmerge moves on hmodel,

        Returns
        -------
        hmodel : bnpy HModel, with (possibly) merged components
        SS : bnpy SuffStatBag, with (possibly) merged components
        evBound : correct ELBO for returned hmodel
                  guaranteed to be at least as large as input evBound    
    '''
    import bnpy.mergemove.MergeLogger as MergeLogger

    MergeLogger.logStartMove(lapFrac)
    self.MergeLog = list()

    for kdel in reversed(xrange(SS.K)):
      aFunc = hmodel.allocModel.calcSoftMergeGap_alph
      oFunc = hmodel.obsModel.calcSoftMergeGap_alph
      ## Find optimal alph redistribution vector for candidate kdel
      from bnpy.mergemove.OptimizerMultiwayMerge import find_optimum
      try:
        alph, f, Info = find_optimum(SS, kdel, aFunc, oFunc)
      except ValueError as e:
        if str(e).lower().count('failure') > 0:
          MergeLogger.log(str(e))
          continue
        raise e

      ## Evaluate total evidence improvement using optimal alpha
      HgapLB = hmodel.allocModel.calcSoftMergeEntropyGap(SS, kdel, alph)
      ELBOgap = hmodel.allocModel.calcSoftMergeGap(SS, kdel, alph) \
                + hmodel.obsModel.calcSoftMergeGap(SS, kdel, alph) \
                + HgapLB

      MergeLogger.log('--------- kdel %d.  N %.1f' 
                            % (kdel, SS.N[kdel]))

      if np.allclose(SS.N.sum(), LPchunk['resp'].shape[0]) \
         and SS.K == LPchunk['resp'].shape[1]:
        from bnpy.util.NumericUtil import calcRlogR
        R = LPchunk['resp']
        R2 = np.delete(R, kdel, axis=1)
        R2[:, kdel:] += R[:, kdel][:,np.newaxis] * alph[kdel+1:][np.newaxis,:]
        R2[:, :kdel] += R[:, kdel][:,np.newaxis] * alph[:kdel][np.newaxis,:]
        assert np.allclose(R2.sum(axis=1), 1.0)
        HgapExact = -1 * np.sum(calcRlogR(R2+1e-100)) \
                     +   np.sum(calcRlogR(R+1e-100))
        ELBOgapExact = ELBOgap - HgapLB + HgapExact

        MergeLogger.log(' HgapLB    % 7.1f' % (HgapLB))
        MergeLogger.log(' HgapExact % 7.1f' % (HgapExact))
        if ELBOgapExact > 0 and ELBOgap < 0:
          msg = '******'
        else:
          msg = ''
        MergeLogger.log(' ELBOgapExact % 7.1f %s' % (ELBOgapExact, msg))
      MergeLogger.log(' ELBOgapLB    % 7.1f' % (ELBOgap))

      MergeLogger.log('Alph')
      MergeLogger.logPosVector(alph)

      if ELBOgap > 0:
        MergeLogger.log('ACCEPTED!')
        ## Accepted!
        SS.multiMergeComps(kdel, alph)
        hmodel.update_global_params(SS)
        evBound += ELBOgap
        curInfo = dict(ELBOgap=ELBOgap, kdel=kdel, alph=alph)
        self.MergeLog.append(curInfo)
        self.verifyELBOTracking(hmodel, SS, evBound)

    return hmodel, SS, evBound


  def verifyELBOTracking(self, hmodel, SS, evBound):
    for batchID in range(len(self.SSmemory.keys())):
      SSchunk = self.load_batch_suff_stat_from_memory(batchID, SS.K, doCopy=1)
      if batchID == 0:
        SS2 = SSchunk.copy()
      else:
        SS2 += SSchunk
    evCheck = hmodel.calc_evidence(SS=SS2)
    #print '% 9.3f' % (evBound)
    #print '% 9.3f' % (evCheck)
    assert np.allclose(SS.N, SS2.N)
    assert np.allclose(evBound, evCheck)
