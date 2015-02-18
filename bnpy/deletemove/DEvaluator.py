"""
Functions that evaluate delete proposals and decide to accept/reject.

- runDeleteMove
"""
import numpy as np
import DeleteLogger
from DCollector import hasValidKey

def runDeleteMoveAndUpdateMemory(curModel, curSS, Plan,
                                 nRefineIters=2,
                                 LPkwargs=None,
                                 SSmemory=None,
                                 Kmax=np.inf,
                                 **kwargs):
  ''' Propose candidate model with fewer comps and accept if ELBO improves.

      Returns
      --------
      bestModel : HModel, with K' states
      bestSS : SuffStatBag with K' states
      SSmemory : dict of SuffStatBags, one per batch, each with K' states
      Plan : dict, with updated fields

      * didAccept
      * acceptedUIDs
      * acceptedIDs

      Post Condition
      --------
      SSmemory represents a valid configuration of every batch under proposed
      model with K' states. If we sum over all batches, the entries of SSmemory
      will be exactly equal to the whole-dataset stats bestSS.

      Both bestSS and each bag in SSmemory have NO ELBO or Merge terms.
  '''
  if SSmemory is None:
      SSmemory = dict()
  if LPkwargs is None:
      LPkwargs = dict()
  if curSS.K == 1:
      Plan['didAccept'] = 0
      Plan['acceptedUIDs'] = list()
      return curModel, curSS, SSmemory, Plan

  # -------------------------     Evaluate current model on target set
  targetData = Plan['DTargetData']
  ctargetLP = curModel.calc_local_params(targetData, **LPkwargs)
  ctargetSS = curModel.get_global_suff_stats(targetData, ctargetLP)
  curELBO = curModel.calc_evidence(targetData, ctargetSS, ctargetLP)

  # -------------------------     bestModel, bestSS represent best so far
  bestModel = curModel
  bestELBO = curELBO
  bestSS = curSS
  besttargetSS = Plan['targetSS']
  assert np.allclose(besttargetSS.uIDs, bestSS.uIDs)

  didAccept = 0
  acceptedUIDs = list()
  for delCompID in Plan['candidateUIDs']:
    if bestSS.K == 1:
      continue # Don't try to remove the final comp!

    # -------------------------    Construct candidate with delCompID removed
    propModel = bestModel.copy()
    propSS = bestSS.copy()
    ptargetSS = besttargetSS.copy()

    k = np.flatnonzero(propSS.uIDs == delCompID)[0]
    propSS.removeComp(k)
    ptargetSS.removeComp(k)  
    propModel.update_global_params(propSS)

    # -------------------------    Refine candidate with local/global steps
    didAcceptCur = 0
    for riter in xrange(nRefineIters):
        ptargetLP = propModel.calc_local_params(targetData, **LPkwargs)
        propSS -= ptargetSS
        ptargetSS = propModel.get_global_suff_stats(targetData, ptargetLP)
        propSS += ptargetSS
        propModel.update_global_params(propSS)

        propELBO = propModel.calc_evidence(targetData, ptargetSS, ptargetLP)

        if not np.isfinite(propELBO):
            break

        if propELBO >= bestELBO or bestSS.K > Kmax:
            didAcceptCur = 1
            didAccept = 1

            break

    # -------------------------    Log result of this proposal
    origk = np.flatnonzero(curSS.uIDs == delCompID)[0]
    if didAcceptCur:
      propname = ' *prop'
    else:
      propname = '  prop'
    curname = '  cur '
    msg = 'comp UID %3d' % (delCompID)
    DeleteLogger.log('%s K=%3d | elbo %.6f | %s | target %12.4f | total %12.4f' 
         % (curname,  besttargetSS.K, bestELBO, msg,
            besttargetSS.getCountVec().sum(), bestSS.getCountVec().sum()))
    DeleteLogger.log('%s K=%3d | elbo %.6f | %s | target %12.4f | total %12.4f' 
         % (propname, ptargetSS.K, propELBO, ' ' * len(msg),
            ptargetSS.getCountVec().sum(), propSS.getCountVec().sum()))

    # -------------------------    Update best model/stats to accepted values
    if didAcceptCur:
       acceptedUIDs.append(delCompID)
       bestELBO = propELBO
       bestModel = propModel

       besttargetLP = ptargetLP
       besttargetSS = ptargetSS

       bestSS = propSS
       bestSS.setELBOFieldsToZero()
       bestSS.setMergeFieldsToZero()

    # << end for loop over each candidate comp

  Plan['didAccept'] = didAccept
  Plan['bestELBO'] = bestELBO
  Plan['acceptedUIDs'] = acceptedUIDs

  # -------------------------    Update SSmemory to reflect accepted deletes
  if didAccept:
      for batchID in SSmemory:
          SSmemory[batchID].setELBOFieldsToZero()
          SSmemory[batchID].setMergeFieldsToZero()

          if hasValidKey(Plan, 'targetSSByBatch'):
              doEditBatch = batchID in Plan['targetSSByBatch']

          # Decrement : subtract old value of targets in this batch
          # Here, SSmemory has K states
          if doEditBatch:
              SSmemory[batchID] -= Plan['targetSSByBatch'][batchID]

          for uID in acceptedUIDs:
              kk = np.flatnonzero(SSmemory[batchID].uIDs == uID)[0]
              SSmemory[batchID].removeComp(kk)

          assert np.allclose(SSmemory[batchID].uIDs, bestSS.uIDs)
          assert SSmemory[batchID].K == besttargetLP['resp'].shape[1]
          assert SSmemory[batchID].K == bestModel.allocModel.K
          assert SSmemory[batchID].K == bestSS.K

          # Increment : add in new value of targets in this batch
          # Here, SSmemory has K-1 states        
          if doEditBatch:
              relUnitIDs = np.flatnonzero(Plan['batchIDs'] == batchID)
              Data_b = targetData.select_subset_by_mask(relUnitIDs,
                                                        doTrackFullSize=False)
              targetLP_b = bestModel.allocModel.selectSubsetLP(
                                                      targetData,
                                                      besttargetLP,
                                                      relUnitIDs)
              targetSS_b = bestModel.get_global_suff_stats(Data_b, targetLP_b)
              SSmemory[batchID] += targetSS_b

          SSmemory[batchID].setELBOFieldsToZero()
          SSmemory[batchID].setMergeFieldsToZero()

  return bestModel, bestSS, SSmemory, Plan
