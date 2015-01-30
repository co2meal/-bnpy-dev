import numpy as np
from matplotlib import pylab

from DeleteLPUtil import deleteCompFromResp_Renorm
from DeleteLPUtil import deleteCompFromResp_SoftEv
from DeleteLPUtil import deleteCompFromResp_SoftEvOverlap

from DeleteLogger import log, logPosVector, logProbVector
from bnpy.viz import BarsViz

def runDeleteMove_SingleSequence(n, Data, curSS, SS_n, curModel,
                                 **kwargs):
  ''' Evaluate candidate model that removes unique states in sequence n.

      Thin wrapper around function runDeleteMove_Target defined below,
      which builds a plan for a current sequence, evaluates it,
      and returns the result.

      Returns
      --------
      newModel
      newSS
      Plan
  '''
  curModel.update_global_params(curSS)  
  
  # Create targeted LP and SS bags for this sequence only
  start = Data.doc_range[n]
  stop = Data.doc_range[n]
  Data_n = Data.select_subset_by_mask([n])
  #LP_n = curModel.allocModel.selectSubsetLP(Data, curLP, [n])
  #SS_n = curModel.get_global_suff_stats(Data_n, LP_n)

  curSS.uIDs = np.arange(curSS.K)
  SS_n.uIDs = curSS.uIDs.copy()

  # Identify states to delete, which are unique to this sequence
  # Attempt them from smallest to largest (in terms of size in this sequence
  N_total = curSS.N
  N_n = SS_n.N
  delCompIDs = np.flatnonzero( (N_total - N_n) < 0.001 )
  delCompSizes = [N_n[k] for k in delCompIDs]
  sort_ids = np.argsort(delCompSizes) # smallest to largest
  delCompIDs = delCompIDs[sort_ids]
  Plan = dict(DTargetData=Data_n,
              targetSS=SS_n,
              uIDs=delCompIDs,
             )
  newModel, newSS, Plan = runDeleteMove_Target(curModel, curSS, Plan, **kwargs)
  delattr(newSS, 'uIDs')
  newSS.removeELBOandMergeTerms()

  return newModel, newSS, Plan


def runDeleteMove_Target(curModel, curSS, Plan,
                         nRefineIters=2,
                         Kmax=np.inf,
                         LPkwargs=None,
                         doVizDelete=False,
                         NmaxDisplay=10,
                         SSmemory=None,
                         **kwargs):
  ''' Propose candidate model with fewer comps and accept if ELBO improves.

      Returns
      --------
      newModel
      newSS
      Plan (with updated fields)
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

  ## Baseline ELBO calculation
  targetData = Plan['DTargetData']
  targetLP = curModel.calc_local_params(targetData, **LPkwargs)
  ctargetSS = curModel.get_global_suff_stats(targetData, targetLP)
  curELBO = curModel.calc_evidence(targetData, ctargetSS, targetLP)

  ## Construct candidate model with components removed!
  newModel = curModel
  newSS = curSS
  newELBO = curELBO
  targetSS = Plan['targetSS']

  assert np.allclose(targetSS.uIDs, newSS.uIDs)
  didAccept = 0
  acceptedUIDs = list()
  acceptedIDs = list()
  for delCompID in Plan['uIDs']:
    if newSS.K == 1:
      continue # Don't try to remove the final comp!

    propSS = newSS.copy()
    propModel = newModel.copy()
    ptargetSS = targetSS.copy()

    k = np.flatnonzero(propSS.uIDs == delCompID)[0]
    propSS.removeComp(k)
    ptargetSS.removeComp(k)  
    
    propModel.update_global_params(propSS)

    newNvec = newSS.getCountVec()

    didAcceptCur = 0
    for riter in xrange(nRefineIters):
      ptargetLP = propModel.calc_local_params(targetData, **LPkwargs)

      propSS -= ptargetSS
      ptargetSS = propModel.get_global_suff_stats(targetData, ptargetLP)
      propSS += ptargetSS
      propModel.update_global_params(propSS)
      
      propELBO = propModel.calc_evidence(targetData, ptargetSS, ptargetLP)
      if propELBO >= newELBO or newSS.K > Kmax:
        didAcceptCur = 1
        didAccept = 1
        newModel = propModel
        newSS = propSS
        targetSS = ptargetSS
        targetLP = ptargetLP
        newELBO = propELBO
        break

    origk = np.flatnonzero(curSS.uIDs == delCompID)[0]
    if didAcceptCur:
      acceptedUIDs.append(delCompID)
      acceptedIDs.append(origk)
      name = ' *prop'
    else:
      name = '  prop'
    curname = '  cur '
    msg = 'compID %3d  size %8.2f' % (delCompID, newNvec[k])
    log('%s K=%3d | elbo %.6f | %s | target %12.4f | total=%12.4f' 
         % (curname, targetSS.K, curELBO, msg,
            targetSS.getCountVec().sum(), newNvec.sum()))
    log('%s K=%3d | elbo %.6f | %s | target %12.4f | total=%12.4f' 
         % (name, ptargetSS.K, propELBO, ' ' * len(msg),
            ptargetSS.getCountVec().sum(), propSS.getCountVec().sum()))
    curELBO = newELBO

  if doVizDelete:
    BarsViz.plotBarsFromHModel(curModel, compsToHighlight=acceptedIDs)
    BarsViz.plotBarsFromHModel(newModel)
    pylab.show(block=False)
    raw_input(">>")
    pylab.close('all')

  Plan['targetLP'] = targetLP
  Plan['didAccept'] = didAccept
  Plan['ELBO'] = newELBO
  Plan['acceptedUIDs'] = acceptedUIDs
  if didAccept:
    ## If improved, adjust the sufficient stats!
    newSS.setELBOFieldsToZero()
    newSS.setMergeFieldsToZero()

    aggSumLogPiRem = 0
    for batchID in SSmemory:
      SSmemory[batchID].setELBOFieldsToZero()
      SSmemory[batchID].setMergeFieldsToZero()
      if 'targetSSByBatch' in Plan and batchID in Plan['targetSSByBatch']:
        SSmemory[batchID] -= Plan['targetSSByBatch'][batchID]

      for uID in acceptedUIDs:
        kk = np.flatnonzero(SSmemory[batchID].uIDs == uID)[0]
        SSmemory[batchID].removeComp(kk)

      docIDs = np.flatnonzero(Plan['batchIDs'] == batchID)

      ## Update batch-specific stored memory, if it changed
      if len(docIDs) > 0:
        Data_b = targetData.select_subset_by_mask(docIDs, doTrackFullSize=False)

        assert SSmemory[batchID].K == targetLP['resp'].shape[1]
        selectSubsetLP = newModel.allocModel.selectSubsetLP
        targetLP_b = selectSubsetLP(targetData, targetLP, docIDs)

        targetSS_b = newModel.get_global_suff_stats(Data_b, targetLP_b)
        SSmemory[batchID] += targetSS_b

      # Be very sure we've set everything to zero
      SSmemory[batchID].setELBOFieldsToZero()
      SSmemory[batchID].setMergeFieldsToZero()

      if hasattr(SSmemory[batchID], 'sumLogPiRem'):
        aggSumLogPiRem += SSmemory[batchID].sumLogPiRem
    if hasattr(newSS, 'sumLogPiRem'):
      assert np.allclose(aggSumLogPiRem, newSS.sumLogPiRem, atol=1e-5)

  return newModel, newSS, Plan
