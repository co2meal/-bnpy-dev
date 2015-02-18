"""
Functions that evaluate delete proposals and decide to accept/reject.

- runDeleteMove
"""

def runDeleteMove(curModel, curSS, Plan,
                  nRefineIters=2,
                  LPkwargs=None,
                  SSmemory=None,
                  Kmax=np.inf,
                  **kwargs):
  ''' Propose candidate model with fewer comps and accept if ELBO improves.

      Returns
      --------
      bestModel
      bestSS
      Plan : dict, with updated fields

      * didAccept
      * acceptedUIDs
      * acceptedIDs
  '''
  if SSmemory is None:
      SSmemory = dict()
  if LPkwargs is None:
      LPkwargs = dict()
  if curSS.K == 1:
      Plan['didAccept'] = 0
      Plan['acceptedUIDs'] = list()
      Plan['acceptedIDs'] = list()
      return curModel, curSS, Plan

  # -------------------------     Evaluate current model on target set
  targetData = Plan['DTargetData']
  targetLP = curModel.calc_local_params(targetData, **LPkwargs)
  ctargetSS = curModel.get_global_suff_stats(targetData, targetLP)
  curELBO = curModel.calc_evidence(targetData, ctargetSS, targetLP)

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
    propSS = bestSS.copy()
    propModel = bestModel.copy()
    proptargetSS = besttargetSS.copy()
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
      if propELBO >= bestELBO or bestSS.K > Kmax:
          didAcceptCur = 1
          didAccept = 1
          bestModel = propModel
          bestSS = propSS
          besttargetSS = ptargetSS
          besttargetLP = ptargetLP
          bestELBO = propELBO
          break

    origk = np.flatnonzero(curSS.uIDs == delCompID)[0]
    if didAcceptCur:
      acceptedUIDs.append(delCompID)
      propname = ' *prop'
    else:
      propname = '  prop'
    curname = '  cur '
    msg = 'comp UID %3d' % (delCompID)
    DeleteLogger.log('%s K=%3d | elbo %.6f | %s | target %12.4f | total=%12.4f' 
         % (curname,  besttargetSS.K, bestELBO, msg,
            besttargetSS.getCountVec().sum(), bestSS.getCountVec().sum()))
    DeleteLogger.log('%s K=%3d | elbo %.6f | %s | target %12.4f | total=%12.4f' 
         % (propname, ptargetSS.K, propELBO, ' ' * len(msg),
            ptargetSS.getCountVec().sum(), propSS.getCountVec().sum()))

  Plan['didAccept'] = didAccept
  Plan['bestELBO'] = bestELBO
  Plan['acceptedUIDs'] = acceptedUIDs
  if didAccept:
      bestSS.setELBOFieldsToZero()
      bestSS.setMergeFieldsToZero()

      for batchID in SSmemory:
          SSmemory[batchID].setELBOFieldsToZero()
          SSmemory[batchID].setMergeFieldsToZero()
          if hasValidKey(Plan, 'targetSSByBatch'):
              if batchID in Plan['targetSSByBatch']:
                  doEditBatch = True

          if doEditBatch:
              SSmemory[batchID] -= Plan['targetSSByBatch'][batchID]

          for uID in acceptedUIDs:
              kk = np.flatnonzero(SSmemory[batchID].uIDs == uID)[0]
              SSmemory[batchID].removeComp(kk)

          assert SSmemory[batchID].K == besttargetLP['resp'].shape[1]
          assert SSmemory[batchID].K == bestModel.allocModel.K
        
          # Update batch-specific stored memory, if it changed
          if doEditBatch:
              relUnitIDs = np.flatnonzero(Plan['batchIDs'] == batchID)
              Data_b = targetData.select_subset_by_mask(relUnitIDs,
                                                        doTrackFullSize=False)
              targetLP_b = bestModel.allocModel.selectSubsetLP(
                                                      targetData,
                                                      targetLP,
                                                      relUnitIDs)
              targetSS_b = bestModel.get_global_suff_stats(Data_b, targetLP_b)
              SSmemory[batchID] += targetSS_b

      # Be very sure we've set everything to zero
      SSmemory[batchID].setELBOFieldsToZero()
      SSmemory[batchID].setMergeFieldsToZero()

  return bestModel, bestSS, Plan