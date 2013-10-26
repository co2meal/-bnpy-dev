'''
MemoizedOnlineVBLearnAlg.py

Implementation of Memoized Online VB (moVB) learn alg for bnpy models
'''
import numpy as np
from bnpy.learn import LearnAlg
import logging
from collections import defaultdict

Log = logging.getLogger('bnpy')


class MemoizedOnlineVBLearnAlg(LearnAlg):

  def __init__( self, **kwargs):
    ''' Creates moVB object
    '''
    super(type(self),self).__init__(**kwargs)
    self.SSmemory = dict()
    self.LPmemory = dict()
    if self.hasMove('merge'):
      self.MergeLog = list()
    if self.hasMove('birth'):
      self.BirthCompIDs = list()
      # Track the number of laps since birth last attempted
      #  at each component, to encourage trying diversity
      self.LapsSinceLastBirth = defaultdict(int)

  def fit(self, hmodel, DataIterator):
    ''' fit hmodel to the provided dataset in DataIterator
    '''
    # memoLPkeys : list of keys for LP that should be retained across laps
    self.memoLPkeys = hmodel.allocModel.get_keys_for_memoized_local_params()
    self.set_start_time_now()
    prevBound = -np.inf
    self.lapFracInc = DataIterator.nObsBatch / float(DataIterator.nObsTotal)
    iterid = -1
    lapFrac = 0
    while DataIterator.has_next_batch():
      # Grab new data and update counts
      Dchunk = DataIterator.get_next_batch()
      batchID = DataIterator.batchID
      iterid += 1
      lapFrac = (iterid + 1) * self.lapFracInc

      # M step
      if iterid > 0:
        hmodel.update_global_params(SS)
      
        # Birth moves!
        if self.hasMove('birth') and lapFrac > 1.0:
          hmodel, SS = self.run_birth_move(
                                    hmodel, Dchunk, SS, LPchunk, lapFrac)

      # E step
      if batchID in self.LPmemory:
        oldLPchunk = self.load_batch_local_params_from_memory(batchID)
        LPchunk = hmodel.calc_local_params(Dchunk, oldLPchunk)
      else:
        LPchunk = hmodel.calc_local_params(Dchunk)

      if self.hasMove('birth'):
        self.subsample_data_for_birth(Dchunk, LPchunk)

      # SS step
      if batchID in self.SSmemory:
        SSchunk = self.load_batch_suff_stat_from_memory(batchID)
        assert SSchunk.K == SS.K
        SS -= SSchunk

      SSchunk = hmodel.get_global_suff_stats(Dchunk, LPchunk,
                       doPrecompEntropy=True, 
                       doPrecompMergeEntropy=self.hasMove('merge')
                       )

      if iterid == 0:
        SS = SSchunk.copy()
      else:
        assert SSchunk.K == SS.K
        SS += SSchunk
        if 'N' in SS.__compkeys__ and np.any(SS.N < 0):        
          SS.N[SS.N < 0] = 0

      self.save_batch_local_params_to_memory(batchID, LPchunk)          
      self.save_batch_suff_stat_to_memory(batchID, SSchunk)  

      # ELBO calc
      evBound = hmodel.calc_evidence(SS=SS)

      # Merge move!      
      if self.hasMove('merge') and lapFrac % 1 == 0:
        hmodel, SS, evBound = self.run_merge_move(hmodel, None, SS, evBound)

      # Save and display progress
      self.add_nObs(Dchunk.nObs)
      lap = iterid
      self.save_state(hmodel, iterid, lapFrac, evBound)
      self.print_state(hmodel, iterid, lapFrac, evBound)

      # Check for Convergence!
      #  evBound will increase monotonically AFTER first lap of the data 
      #  verify_evidence will warn if bound isn't increasing monotonically
      if lapFrac > 1.0:
        isConverged = self.verify_evidence(evBound, prevBound)
        if isConverged:
          break
      prevBound = evBound

    # Finally, save, print and exit
    if isConverged:
      msg = "converged."
    else:
      msg = "max passes thru data exceeded."
    self.save_state(hmodel, iterid, lapFrac, evBound, doFinal=True) 
    self.print_state(hmodel, iterid, lapFrac, evBound, doFinal=True, status=msg)
    return None, evBound

  ######################################################### Load from memory
  #########################################################
  def load_batch_suff_stat_from_memory(self, batchID):
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
        SSchunk.insertEmptyComponents(Kextra)
    if self.hasMove('merge'): 
      for MInfo in self.MergeLog:
        kA = MInfo['kA']
        kB = MInfo['kB']
        SSchunk.mergeComponents(kA, kB)
      # After any accepted merges are done
      if SSchunk.hasPrecompMergeEntropy():
        SSchunk.setToZeroPrecompMergeEntropy()
      elif SSchunk.hasPrecompMerge():
        SSchunk.setToZeroAllPrecompMergeTerms()
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
      raise NotImplementedError('TODO')
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

  #####################################################################
  #####################################################################
  def run_birth_move(self, hmodel, Dchunk, SS, LPchunk, lapFrac):
    ''' Run birth moves on hmodel.
        Internally handles subsampling data, suff stat bookkeeping, etc.

        Returns
        -------
        hmodel : bnpy HModel, with (possibly) new components
        SS : bnpy SuffStatDict, with (possibly) new components
    '''
    import BirthMove

    if not hasattr(self, 'targetDataList'):
      self.targetDataList = list()
    if lapFrac > (0.8 * self.algParams['nLap']):
      return hmodel, SS

    isFirstBatch = np.allclose(lapFrac - np.floor(lapFrac), self.lapFracInc)
    isLastBatch = lapFrac % 1 == 0

    if isFirstBatch:
      self.BirthCompIDs = list()
      self.BirthInfoCurLap = list()

      # Increment birth counter
      for kk in range(SS.K):
        self.LapsSinceLastBirth[kk] += 1

      # Run birth moves on current target data!
      for tInfoDict in self.targetDataList:
        ktarget = tInfoDict['ktarget']
        targetData = tInfoDict['Data']
        if targetData.nObs < self.algParams['birth']['minTargetObs']:
          Log.info("BIRTH Skipped at comp %d : target dataset too small (size %d)" % (tInfoDict['ktarget'], targetData.nObs))
          continue
        hmodel, SS, MoveInfo = BirthMove.run_birth_move(
                 hmodel, targetData, SS, randstate=self.PRNG, 
                 ktarget=ktarget, **self.algParams['birth'])
        
        if MoveInfo['didAddNew']:
          self.BirthInfoCurLap.append(MoveInfo)
          for kk in MoveInfo['birthCompIDs']:
            self.LapsSinceLastBirth[kk] = -1
        self.print_msg(MoveInfo['msg'])
        self.BirthCompIDs.extend(MoveInfo['birthCompIDs'])

      # Sample new components to target for the next birth proposal
      self.targetDataList = list()

      # Ignore components that have just been added to the model,
      #  as well as components that 
      excludeList = [kk for kk in self.BirthCompIDs]

      for posID in range(self.algParams['birth']['birthPerLap']):
        try:
          ktarget = BirthMove.select_birth_component(SS, randstate=self.PRNG,
                          excludeList=excludeList, doVerbose=False,
                          lapsSinceLastBirth=self.LapsSinceLastBirth,
                          **self.algParams['birth'])
          self.LapsSinceLastBirth[ktarget] = 0
          excludeList.append(ktarget)
          tInfoDict = dict(ktarget=ktarget, Data=None)
          self.targetDataList.append(tInfoDict)
        except BirthMove.BirthProposalError, e:
          Log.debug(str(e))

    # Handle removing "artificial mass" of fresh components
    elif isLastBatch:
      Nall = np.sum(SS.N)
      didChangeSS = False
      for MoveInfo in self.BirthInfoCurLap:
        freshSS = MoveInfo['freshSS']
        birthCompIDs = MoveInfo['birthCompIDs']
        SS.subtractSpecificComponents(freshSS, birthCompIDs)
        didChangeSS = True
      if didChangeSS:
        hmodel.update_global_params(SS)
      assert np.abs(np.sum(SS.N) - Dchunk.nObsTotal) < 0.0001

    # Return and exit. That's all folks.
    return hmodel, SS

    
  def subsample_data_for_birth(self, Dchunk, LPchunk):
    ''' Incrementally build-up a target dataset to use as basis for BirthMove!
        Calling this method updates the internal data objects.
        Args
        -------
        Dchunk : data object to subsample from
        LPchunk : local parameters for Dchunk

        Returns
        -------
        None (all updates happen to internal data structures)
    '''
    import BirthMove

    if not hasattr(self, 'targetDataList'):
      self.targetDataList = list()
    for tInfoDict in self.targetDataList:
      ktarget = tInfoDict['ktarget']
      if tInfoDict['Data'] is not None:
        if tInfoDict['Data'].nObs > self.algParams['birth']['maxTargetObs']:
          continue
      # Sample data if more is needed
      targetData = BirthMove.subsample_data(Dchunk, LPchunk, ktarget, 
                          randstate=self.PRNG,
                          **self.algParams['birth'])
      if tInfoDict['Data'] is None:
        tInfoDict['Data'] = targetData
      else:
        tInfoDict['Data'].add_data(targetData)


  #####################################################################
  #####################################################################
  def run_merge_move(self, hmodel, Data, SS, evBound):
    ''' Run (potentially many) merge moves on hmodel,
          performing necessary bookkeeping to
            (1) avoid trying the same merge twice
            (2) avoid merging a component that has already been merged,
                since the precomputed entropy will no longer be correct.
        Returns
        -------
        hmodel : bnpy HModel, with (possibly) some merged components
        SS : bnpy SuffStatDict, with (possibly) merged components
        evBound : correct ELBO for returned hmodel
                  guaranteed to be at least as large as input evBound    
    '''
    import MergeMove
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

      hmodel, SS, evBound, MoveInfo = MergeMove.run_merge_move(
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

    if SS.hasPrecompMergeEntropy():
      SS.setToZeroPrecompMergeEntropy()
    elif SS.hasPrecompMerge():
      SS.setToZeroAllPrecompMergeTerms()
    return hmodel, SS, evBound
