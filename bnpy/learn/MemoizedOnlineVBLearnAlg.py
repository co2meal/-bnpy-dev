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
        SSchunk = self.load_batch_suff_stat_from_memory(batchID)
        SS -= SSchunk
         
      SSchunk = hmodel.get_global_suff_stats(Dchunk, LPchunk, doPrecompEntropy=True)
      if iterid == 0:
        SS = SSchunk.copy()
      else:
        SS += SSchunk
      self.save_batch_suff_stat_to_memory(batchID, SSchunk)
      
      # ELBO calc
      evBound = hmodel.calc_evidence(SS=SS)
      
      # Save and display progress
      self.add_nObs(Dchunk.nObs)
      lap = iterid
      self.save_state(hmodel, iterid, lapFrac, evBound)
      self.print_state(hmodel, iterid, lapFrac, evBound)

      # Check for Convergence!
      #  report warning if bound isn't increasing monotonically
      isConverged = self.verify_evidence(evBound, prevBound)
      if isConverged:
        break
      prevBound = evBound

    #Finally, save, print and exit
    if isConverged:
      status = "converged."
    else:
      status = "max passes thru data exceeded."
    self.save_state(hmodel, iterid, lapFrac, evBound, doFinal=True) 
    self.print_state(hmodel, iterid, lapFrac, evBound, doFinal=True, status=status)
    return None

  #####################################################################
  #####################################################################
  def load_batch_suff_stat_from_memory(self, batchID):
    return self.SSmemory[batchID]
  
  def save_batch_suff_stat_to_memory(self, batchID, SS):
    self.SSmemory[batchID] = SS
