'''
MemoizedOnlineVBLearnAlg.py

Implementation of Memoized Online VB (moVB) learn alg for bnpy models
'''
import copy
import numpy as np
from bnpy.learn import LearnAlg

class MemoizedOnlineVBLearnAlg(LearnAlg):

  def __init__( self, **kwargs):
    super(type(self),self).__init__(**kwargs)
    self.SSmemory = dict()
    if self.hasMove('merge'):
      self.MergeLog = list()

  def fit(self, hmodel, DataIterator):
    self.set_start_time_now()
    prevBound = -np.inf
    LPchunk = None
    lapFracPerBatch = DataIterator.nObsBatch / float(DataIterator.nObsTotal)
    iterid = -1
    lapFrac = 0
    while DataIterator.has_next_batch():
      # Grab new data and update counts
      Dchunk = DataIterator.get_next_batch()
      batchID = DataIterator.batchID
      iterid += 1
      lapFrac = (iterid + 1) * lapFracPerBatch

      # M step
      if iterid > 0:
        hmodel.update_global_params(SS)

      # E step
      LPchunk = hmodel.calc_local_params(Dchunk, LPchunk)

      # SS step
      if batchID in self.SSmemory:
        assert SS.hasPrecompMergeEntropy()
        SSchunk = self.load_batch_suff_stat_from_memory(batchID)
        SS -= SSchunk
        assert SS.hasPrecompMergeEntropy()

      SSchunk = hmodel.get_global_suff_stats(
                       Dchunk, LPchunk,
                       doPrecompEntropy=True, 
                       doPrecompMergeEntropy=self.hasMove('merge')
                       )
      if iterid == 0:
        SS = SSchunk.copy()
      else:
        SS += SSchunk
      self.save_batch_suff_stat_to_memory(batchID, SSchunk)
      
      # ELBO calc
      evBound = hmodel.calc_evidence(SS=SS)
      
      # Attempt merge moves if available      
      if self.hasMove('merge') and lapFrac % 1 == 0:
        hmodel, SS, evBound = self.run_merge_move(hmodel, None, SS, evBound)
      assert SS.hasPrecompMergeEntropy()

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
      status = "converged."
    else:
      status = "max passes thru data exceeded."
    self.save_state(hmodel, iterid, lapFrac, evBound, doFinal=True) 
    self.print_state(hmodel, iterid, lapFrac, evBound, doFinal=True, status=status)
    return None, evBound

  #####################################################################
  #####################################################################
  def load_batch_suff_stat_from_memory(self, batchID):
    SSchunk = self.SSmemory[batchID]
    # Play merge forward
    if self.hasMove('merge'): 
      for MInfo in self.MergeLog:
        kA = MInfo['kA']
        kB = MInfo['kB']
        SSchunk.mergeComponents(kA, kB)
        SSchunk.setToZeroPrecompMergeEntropy()
    return SSchunk  

  def save_batch_suff_stat_to_memory(self, batchID, SSchunk):
    self.SSmemory[batchID] = SSchunk

  #####################################################################
  #####################################################################
  def run_merge_move(self, hmodel, Data, SS, evBound):
    ''' Run merge move on hmodel
    '''
    import MergeMove
    self.MergeLog = list() # clear memory of recent merges!
    excludeList = list()    
    for trialID in range(self.algParams['merge']['mergePerLap']):
      hmodel, SS, evBound, MoveInfo = MergeMove.run_merge_move(
                 hmodel, Data, SS, evBound, randstate=self.PRNG,
                 excludeList=excludeList, **self.algParams['merge'])
      self.print_msg(MoveInfo['msg'])
      if MoveInfo['didAccept']:
        kA = MoveInfo['kA']
        kB = MoveInfo['kB']
        self.MergeLog.append(dict(kA=kA, kB=kB))

        # Adjust excludeList since components kB+1, kB+2, ... K
        #  have been shifted down by one due to removal of kB
        for kk in range(len(excludeList)):
          if excludeList[kk] > kB:
            excludeList[kk] -= 1
        # Exclude new merged component kA from future attempts        
        #  since precomputed entropy terms involving kA aren't good
        excludeList.append(kA)
    
    SS.setToZeroPrecompMergeEntropy()
    return hmodel, SS, evBound
