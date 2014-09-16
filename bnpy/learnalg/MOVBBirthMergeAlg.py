'''
MOVBAlg.py

Implementation of Memoized Online VB (moVB) learn alg for bnpy models
'''
from collections import defaultdict
import numpy as np
import os
import logging
Log = logging.getLogger('bnpy')

from MOVBAlg import MOVBAlg, makeDictOfAllWorkspaceVars

from bnpy.suffstats import SuffStatBag
from bnpy.util import isEvenlyDivisibleFloat
from bnpy.birthmove import TargetPlanner, TargetDataSampler, BirthMove
from bnpy.birthmove import BirthLogger, TargetPlannerWordFreq
from bnpy.mergemove import MergeMove, MergePlanner, MergeLogger
from bnpy.birthmove.TargetDataSampler import hasValidKey

class MOVBBirthMergeAlg(MOVBAlg):

  def __init__(self, **kwargs):
    ''' Construct memoized algorithm instance that can do births/merges.

        Includes specialized internal fields to hold "memoized" statistics
    '''
    MOVBAlg.__init__(self, **kwargs)

    if self.hasMove('merge') or self.hasMove('softmerge'):
      self.MergeLog = list()
      self.lapLastAcceptedMerge = self.algParams['startLap']

    if self.hasMove('birth'):
      # Track the number of laps since birth last attempted
      #  at each component, to encourage trying diversity
      self.LapsSinceLastBirth = defaultdict(int)


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
    origmodel = hmodel
    self.ActiveIDVec = np.arange(hmodel.obsModel.K)
    self.maxUID = self.ActiveIDVec.max()

    ## Initialize progress tracking vars like nBatch, lapFrac, etc.
    iterid, lapFrac = self.initProgressTrackVars(DataIterator)

    ## Keep list of params that should be retained across laps
    self.memoLPkeys = hmodel.allocModel.get_keys_for_memoized_local_params()

    ## Save initial state
    self.saveParams(lapFrac, hmodel)

    ## Prep for birth 
    BirthPlans = list()
    BirthResults = list()
    prevBirthResults = list()

    ## Prep for merge
    MergePlanInfo = dict()
    if self.hasMove('merge'):
      mergeStartLap = self.algParams['merge']['mergeStartLap']
    else:
      mergeStartLap = 0
    order = None

    ## Begin loop over batches of data...
    SS = None
    isConverged = False
    self.set_start_time_now()
    while DataIterator.has_next_batch():

      ## Grab new data
      Dchunk = DataIterator.get_next_batch()
      batchID = DataIterator.batchID
      
      ## Update progress-tracking variables
      iterid += 1
      lapFrac = (iterid + 1) * self.lapFracInc
      nLapsCompleted = lapFrac - self.algParams['startLap']
      self.set_random_seed_at_lap(lapFrac)

      ## Birth move : track birth info from previous lap
      if self.isFirstBatch(lapFrac):
        if self.hasMove('birth') and self.do_birth_at_lap(lapFrac - 1.0):
          prevBirthResults = BirthResults
        else:
          prevBirthResults = list()

      ## Birth move : create new components
      if self.hasMove('birth') and self.do_birth_at_lap(lapFrac):
        if self.doBirthWithPlannedData(lapFrac):
          hmodel, SS, BirthResults = self.birth_create_new_comps(
                                            hmodel, SS, BirthPlans,
                                                        lapFrac=lapFrac)

        if self.doBirthWithDataFromCurrentBatch(lapFrac):
          hmodel, SS, curResults = self.birth_create_new_comps(
                                            hmodel, SS, Data=Dchunk,
                                                        lapFrac=lapFrac)
          BirthResults.extend(curResults)
      else:
        BirthResults = list()

      ## Prepare for merges
      if self.hasMove('merge') and self.doMergePrepAtLap(lapFrac):
        MergePrepInfo = self.preparePlansForMerge(hmodel, SS, MergePrepInfo,
                                                  order=order,
                                                  BirthResults=BirthResults,
                                                  lapFrac=lapFrac)
      elif self.isFirstBatch(lapFrac):
        MergePrepInfo = dict()

      ## Local/E step
      LPchunk = self.memoizedLocalStep(hmodel, Dchunk, batchID)
      
      ## Birth move : collect target data
      if self.hasMove('birth') and self.do_birth_at_lap(lapFrac+1.0):
        if self.isFirstBatch(lapFrac):
          BirthPlans = self.birth_plan_targets_for_next_lap(
                                Dchunk, hmodel, SS, LPchunk, BirthResults)
        BirthPlans = self.birth_collect_target_subsample(
                                Dchunk, hmodel, LPchunk, BirthPlans, lapFrac)
      else:
        BirthPlans = list()

      ## Summary step
      SS, SSchunk = self.memoizedSummaryStep(hmodel, SS,
                                             Dchunk, LPchunk, batchID,
                                             MergePrepInfo=MergePrepInfo,
                                             order=order)

      ## Birth : Handle removing "extra mass" of fresh components
      if self.hasMove('birth') and self.isLastBatch(lapFrac):
        hmodel, SS = self.birth_remove_extra_mass(hmodel, SS, BirthResults)
        # SS now has size exactly consistent with entire dataset
      
      ## Global/M step
      self.GlobalStep(hmodel, SS, lapFrac)

      ## ELBO calculation
      evBound = hmodel.calc_evidence(SS=SS)

      ## Merge move!
      if self.hasMove('merge') and self.isLastBatch(lapFrac) \
                               and lapFrac > mergeStartLap:
        hmodel, SS, evBound = self.run_many_merge_moves(hmodel, SS, evBound,
                                                        lapFrac, MergePrepInfo)

      ## Shuffle : Rearrange topic order (big to small)
      if self.hasMove('shuffle') and self.isLastBatch(lapFrac):
        order = np.argsort(-1*SS.getCountVec())
        sortedalready = np.arange(SS.K)
        if np.allclose(order, sortedalready):
          order = None # Already sorted, do nothing!
        else:
          self.ActiveIDVec = self.ActiveIDVec[order]
          SS.reorderComps(order)
          assert np.allclose(SS.uIDs, self.ActiveIDVec)

          hmodel.update_global_params(SS)
          evBound = hmodel.calc_evidence(SS=SS)

      if nLapsCompleted > 1.0 and len(BirthResults) == 0:
        # evBound will increase monotonically AFTER first lap of the data 
        # verify_evidence will warn if bound isn't increasing monotonically
        self.verify_evidence(evBound, prevBound, lapFrac)

      if self.doDebug() and lapFrac >= 1.0:
        self.verifyELBOTracking(hmodel, SS, evBound, order=order,
                                BirthResults=BirthResults)
            
      ## Assess convergence
      countVec = SS.getCountVec()
      if lapFrac > 1.0:
        isConverged = self.isCountVecConverged(countVec, prevCountVec)
        isConverged = isConverged and not self.hasMoreReasonableMoves(lapFrac)
        self.setStatus(lapFrac, isConverged)

      ## Display progress
      self.updateNumDataProcessed(Dchunk.get_size())
      if self.isLogCheckpoint(lapFrac, iterid):
        self.printStateToLog(hmodel, evBound, lapFrac, iterid)

      ## Save diagnostics and params
      if self.isSaveDiagnosticsCheckpoint(lapFrac, iterid):
        self.saveDiagnostics(lapFrac, SS, evBound, self.ActiveIDVec)
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

    # Births and merges require copies of original model object
    #  we need to make sure original reference has updated parameters, etc.
    if id(origmodel) != id(hmodel):
      origmodel.allocModel = hmodel.allocModel
      origmodel.obsModel = hmodel.obsModel
    return self.buildRunInfo(evBound=evBound, SS=SS,
                             LPmemory=self.LPmemory,
                             SSmemory=self.SSmemory)


  def hasMoreReasonableMoves(self, lapFrac):
    ''' Decide if more moves will feasibly change current configuration. 
    '''
    if lapFrac - self.algParams['startLap'] >= self.algParams['nLap']:
      ## Time's up, so doesn't matter what other moves are possible.
      return False

    if self.hasMove('birth') and self.do_birth_at_lap(lapFrac):
      return True # Never quit early for birth

    elif self.hasMove('merge'):
      nStuckBeforeQuit = self.algParams['merge']['mergeNumStuckBeforeQuit']
      if (lapFrac - self.lapLastAcceptedMerge) > nStuckBeforeQuit:
        return False
      return True

    return False

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

  def load_batch_local_params_from_memory(self, batchID):
    ''' Load local parameter dict stored in memory for provided batchID

        Ensures "fast-forward" so that all recent merges/births
        are accounted for in the returned LP

        Returns
        -------
        LPchunk : bnpy local parameters dictionary for batchID
    '''
    LPchunk = self.LPmemory[batchID]
    if self.hasMove('merge') and LPchunk is not None:
      K =  LPchunk[self.memoLPkeys[0]].shape[1]
      for MInfo in self.MergeLog:
        kA = MInfo['kA']
        kB = MInfo['kB']
        for key in self.memoLPkeys:
          if kA >= K or kB >= K:
            # Stored LPchunk is outdated... forget it.
            return None
          kB_column = LPchunk[key][:,kB]
          LPchunk[key] = np.delete(LPchunk[key], kB, axis=1)
          LPchunk[key][:,kA] = LPchunk[key][:,kA] + kB_column

    return LPchunk


  def save_batch_local_params_to_memory(self, batchID, LPchunk, doCopy=0):
    ''' Store certain local params into internal LPmemory cache.

        Fields to save determined by the memoLPkeys attribute of this alg.

        Returns
        ---------
        None. self.LPmemory updated in-place.
    '''
    keepLPchunk = dict()
    for key in LPchunk.keys():
      if key in self.memoLPkeys:
        if doCopy:
          keepLPchunk[key] = copy.deepcopy(LPchunk[key])
        else:
          keepLPchunk[key] = LPchunk[key]

    if len(keepLPchunk.keys()) > 0:
      self.LPmemory[batchID] = keepLPchunk
    else:
      self.LPmemory[batchID] = None

  ######################################################### Summary step
  #########################################################
  def memoizedSummaryStep(self, hmodel, SS, Dchunk, LPchunk, batchID,
                                order=None, MergePrepInfo=None):
    ''' Execute summary step on current batch and update aggregated SS

        Returns
        --------
        SS : updated aggregate suff stats
        SSchunk : updated current-batch suff stats
    '''
    if MergePrepInfo is None:
      MergePrepInfo = dict()

    if batchID in self.SSmemory:
      ## Decrement old value of SSchunk from aggregated SS
      oldSSchunk = self.load_batch_suff_stat_from_memory(batchID, doCopy=0,
                                                         Kfinal=SS.K,
                                                         order=order)
      assert oldSSchunk.K == SS.K
      assert np.allclose(SS.uIDs, oldSSchunk.uIDs)
      SS -= oldSSchunk

    ## Calculate fresh suff stats for current batch
    SSchunk = hmodel.get_global_suff_stats(Dchunk, LPchunk, 
                                           doPrecompEntropy=1,
                                           **MergePrepInfo)
    SSchunk.setUIDs(self.ActiveIDVec.copy())

    ## Increment aggregated SS by adding in SSchunk
    if SS is None:
      SS = SSchunk.copy()
    else:
      assert SSchunk.K == SS.K
      assert np.allclose(SS.uIDs, self.ActiveIDVec)
      assert np.allclose(SSchunk.uIDs, self.ActiveIDVec)
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
                                             Kfinal=0, order=None):
    ''' Load (fast-forwarded) suff stats stored from previous visit to batchID.

        Any merges, shuffles, or births which happened since last visit
        are automatically applied.

        Returns
        -------
        SSchunk : bnpy SuffStatDict object for batchID,
    '''
    SSchunk = self.SSmemory[batchID]
    if doCopy:
      # Duplicating to avoid changing the raw data stored in SSmemory
      #  this is done usually when debugging.
      SSchunk = SSchunk.copy()

    # Check to see if we've fast-forwarded this chunk already
    # If so, we can just use it as is.
    if SSchunk.K == self.ActiveIDVec.size:
      if np.allclose(SSchunk.uIDs, self.ActiveIDVec):
        return SSchunk

    # Fast-forward accepted softmerges from end of previous lap 
    if self.hasMove('softmerge'): 
      for MInfo in self.MergeLog:
        SSchunk.multiMergeComps(MInfo['kdel'], MInfo['alph'])
    
    # Fast-forward accepted merges from end of previous lap 
    if self.hasMove('merge'): 
      for MInfo in self.MergeLog:
        kA = MInfo['kA']
        kB = MInfo['kB']
        if kA < SSchunk.K and kB < SSchunk.K and SSchunk.K == MInfo['Korig']:
          SSchunk.mergeComps(kA, kB)
    if SSchunk.hasMergeTerms():
      SSchunk.setMergeFieldsToZero()

    # Fast-forward any shuffling/reordering that happened
    if self.hasMove('shuffle') and order is not None:
      if SSchunk.K == self.ActiveIDVec.size:
        if not np.allclose(SSchunk.uIDs, self.ActiveIDVec):
          SSchunk.reorderComps(order)
      else:
        SSchunk.reorderComps(order)

    isGood = np.allclose(SSchunk.uIDs, self.ActiveIDVec[:SSchunk.K])
    if not isGood:
      if self.algParams['debug'] == 'interactive':
        from IPython import embed; embed()
    assert isGood    

    # Fast-forward births from this lap
    if self.hasMove('birth') and Kfinal > 0 and SSchunk.K < Kfinal:
      Kextra = Kfinal - SSchunk.K
      if Kextra > 0:
        SSchunk.insertEmptyComps(Kextra)
      assert SSchunk.K == Kfinal
      SSchunk.setUIDs(self.ActiveIDVec.copy())

    assert np.allclose(SSchunk.uIDs, self.ActiveIDVec)
    return SSchunk  

  def save_batch_suff_stat_to_memory(self, batchID, SSchunk):
    ''' Store the provided suff stats into the "memory" for later retrieval
    '''
    if SSchunk.hasSelectionTerms():
      del SSchunk._SelectTerms
    self.SSmemory[batchID] = SSchunk




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
        newmodel, newSS, MoveInfo = BirthMove.run_birth_move(
                                           hmodel, SS, targetData, 
                                           randstate=self.PRNG, 
                                           Plan=Plan,
                                           **kwargs)
        hmodel = newmodel
        SS = newSS

        if MoveInfo['didAddNew']:
          BirthResults.append(MoveInfo)
          for kk in MoveInfo['birthCompIDs']:
            self.LapsSinceLastBirth[kk] = -1
            self.maxUID += 1
            self.ActiveIDVec = np.append(self.ActiveIDVec, self.maxUID)
          SS.setUIDs(self.ActiveIDVec.copy())
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
        MoveInfo['extraSSDone'] = 1
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
  def preparePlansForMerge(self, hmodel, SS, prevPrepInfo=None,
                                             order=None,
                                             BirthResults=list(),
                                             lapFrac=0):
    if prevPrepInfo is None:
      prevPrepInfo = dict()
    if 'PairScoreMat' not in prevPrepInfo:
      prevPrepInfo['PairScoreMat'] = None

    if SS is not None:
      # Remove any merge terms left over from previous lap
      SS.setMergeFieldsToZero()

    mergeStartLap = self.algParams['merge']['mergeStartLap']
    preselectroutine = self.algParams['merge']['preselectroutine']
    mergeELBOTrackMethod = self.algParams['merge']['mergeELBOTrackMethod']
    refreshInterval = self.algParams['merge']['mergeScoreRefreshLap']

    PrepInfo = dict()
    PrepInfo['doPrecompMergeEntropy'] = 1
    PrepInfo['preselectroutine'] = preselectroutine
    PrepInfo['mPairIDs'] = list()
    PrepInfo['PairScoreMat'] = None

    ## Short-cut if we use fastBound to compute elbo for merge candidate
    if mergeELBOTrackMethod == 'fastBound':
      PrepInfo['doPrecompMergeEntropy'] = 2
      PrepInfo['preselectroutine'] = None
      return PrepInfo

    ## Update stored ScoreMatrix to account for recent births/merges
    if hasValidKey('PairScoreMat', prevPrepInfo):
      MM = prevPrepInfo['PairScoreMat']

      ## Replay any shuffles
      if order is not None:
        Ktmp = len(order)
        assert Ktmp == MM.shape[0]
        Mnew = np.zeros_like(MM)
        for kA in xrange(Ktmp):
          nA = np.flatnonzero(order == kA)
          for kB in xrange(kA+1, Ktmp):
            nB = np.flatnonzero(order == kB)
            mA = np.minimum(nA, nB)
            mB = np.maximum(nA, nB)
            Mnew[mA, mB] = MM[kA, kB]
        MM = Mnew

      ## Replay any recent birth moves!
      if len(BirthResults) > 0:
        Korig = MM.shape[0]
        Mnew = np.zeros((SS.K, SS.K))
        Mnew[:Korig, :Korig] = MM
        MM = Mnew
      if np.floor(lapFrac) % refreshInterval == 0:
        MM.fill(0) # Refresh!
      prevPrepInfo['PairScoreMat'] = MM

    ## Determine which merge pairs we will track in the upcoming lap 
    if preselectroutine == 'wholeELBObetter':
      mPairIDs, PairScoreMat = MergePlanner.preselectPairs(hmodel, SS, lapFrac,
                                 prevScoreMat=prevPrepInfo['PairScoreMat'])
    else:
      mPairIDs, PairScoreMat = MergePlanner.preselect_candidate_pairs(hmodel,
                                            SS,
                                            randstate=self.PRNG,
                                            returnScoreMatrix=1,
                                            M=prevPrepInfo['PairScoreMat'],
                                            **self.algParams['merge'])

    PrepInfo['mPairIDs'] = mPairIDs
    PrepInfo['PairScoreMat'] = PairScoreMat
    MergeLogger.log('MERGE Num pairs selected: %d' % (len(mPairIDs)),
                    level='debug')

    degree = MergePlanner.calcDegreeFromEdgeList(mPairIDs, SS.K)
    if np.sum( degree > 0 ) > 0:
      degree = degree[degree > 0]
      MergeLogger.log('Degree distribution among selected pairs', 'debug')
      for p in [10, 50, 90, 100]:
        MergeLogger.log('   %d: %d' % (p, np.percentile(degree, p)), 'debug')

    ## Reset selection terms to zero
    if SS is not None and SS.hasSelectionTerms():
      SS._SelectTerms.setAllFieldsToZero()
    return PrepInfo

  def run_many_merge_moves(self, hmodel, SS, evBound, lapFrac, MergePrepInfo):
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

    if 'mPairIDs' not in MergePrepInfo or MergePrepInfo['mPairIDs'] is None:
      MergePrepInfo['mPairIDs'] = list()

    if 'PairScoreMat' not in MergePrepInfo:
      MergePrepInfo['PairScoreMat'] = None

    Korig = SS.K
    hmodel, SS, newEvBound, Info = MergeMove.run_many_merge_moves(
                                       hmodel, SS, evBound,
                                       mPairIDs=MergePrepInfo['mPairIDs'],
                                       M=MergePrepInfo['PairScoreMat'],
                                       **self.algParams['merge'])
    afterMat = Info['ScoreMat'].copy()
    assert afterMat.shape[0] == SS.K

    # ------ Adjust indexing for counter that determines which comp to target
    if self.hasMove('birth'):
      for kA, kB in Info['AcceptedPairs']:
        self._resetLapsSinceLastBirthAfterMerge(kA, kB)

    # ------ Record accepted moves, so can adjust memoized stats later
    self.MergeLog = list()
    for kA, kB in Info['AcceptedPairs']:
      self.ActiveIDVec = np.delete(self.ActiveIDVec, kB, axis=0)      
      self.MergeLog.append(dict(kA=kA, kB=kB, Korig=Korig))
      self.lapLastAcceptedMerge = lapFrac
      Korig -= 1

    # ------ Reset all precalculated merge terms
    if SS.hasMergeTerms():
      SS.setMergeFieldsToZero()

    MergePrepInfo['PairScoreMat'] = Info['ScoreMat']
    MergePrepInfo['mPairIDs'] = list()
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
        newDict[kA] = np.maximum(self.LapsSinceLastBirth[kA], 
                                 self.LapsSinceLastBirth[kB])
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

  ######################################################### Verify ELBO
  #########################################################
  def verifyELBOTracking(self, hmodel, SS, evBound=None, order=None,
                               BirthResults=list(),
                               **kwargs):
    ''' Verify that current aggregated SS consistent with sum over all batches
    '''
    if evBound is None:
      evBound = hmodel.calc_evidence(SS=SS)

    ## Reconstruct aggregate SS explicitly by sum over all stored batches
    for batchID in range(len(self.SSmemory.keys())):
      SSchunk = self.load_batch_suff_stat_from_memory(batchID, doCopy=1,
                                                      order=order,
                                                      Kfinal=SS.K)
      if batchID == 0:
        SS2 = SSchunk.copy()
      else:
        SS2 += SSchunk

    ## Add in extra mass from birth moves
    for MoveInfo in BirthResults:
      if MoveInfo['didAddNew'] and 'extraSS' in MoveInfo:
        if not 'extraSSDone' in MoveInfo:
          extraSS = MoveInfo['extraSS'].copy()
          if extraSS.K < SS2.K:
            extraSS.insertEmptyComps(SS2.K - extraSS.K)
          SS2 += extraSS

    evCheck = hmodel.calc_evidence(SS=SS2)
    if self.algParams['debug'].count('quiet') == 0:
      print '% 14.8f evBound from agg SS' % (evBound)
      print '% 14.8f evBound from sum over SSmemory' % (evCheck)
    if self.algParams['debug'].count('interactive'):
      isCorrect = np.allclose(SS.uIDs, SS2.uIDs) \
                  and np.allclose(SS.getCountVec(), SS2.getCountVec()) \
                  and np.allclose(evBound, evCheck)
      if not isCorrect:
        from IPython import embed; embed()
    else:
      assert np.allclose(SS.getCountVec(), SS2.getCountVec())
      assert np.allclose(evBound, evCheck)
      assert np.allclose(SS.uIDs, SS2.uIDs)

"""
DEPRECATED CODE from load_suff_stats_for_batch

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
"""
