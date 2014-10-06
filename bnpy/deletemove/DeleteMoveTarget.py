import numpy as np

from DeleteLPUtil import deleteCompFromResp_Renorm
from DeleteLPUtil import deleteCompFromResp_SoftEv
from DeleteLPUtil import deleteCompFromResp_SoftEvOverlap

from DeleteLogger import log, logPosVector, logProbVector

def runDeleteMove_Target(curModel, curSS, Plan,
                         nRefineIters=5,
                         doQuitEarly=1,
                         LPkwargs=None,
                         **kwargs):
  ''' Propose candidate model with fewer comps and accept if ELBO improves.

      Returns
      --------
      hmodel
      SS
      Info
  '''
  if LPkwargs is None:
    LPkwargs = dict()

  ## Baseline ELBO calculation
  targetData = Plan['DTargetData']
  ctargetLP = curModel.calc_local_params(targetData, **LPkwargs)
  ctargetSS = curModel.get_global_suff_stats(targetData, ctargetLP)
  curELBO = curModel.calc_evidence(targetData, ctargetSS, ctargetLP)

  ## Construct candidate model with components removed!
  deleteIDs = Plan['selectIDs']
    
  propSS = curSS.copy()
  ptargetSS = Plan['targetSS']
  curNvec = curSS.getCountVec()
  remcurNvec = curNvec.copy()
  for delCompID in Plan['uIDs']:
    k = np.flatnonzero(propSS.uIDs == delCompID)[0]
    propSS.removeComp(k)
    ptargetSS.removeComp(k)  
    remcurNvec = np.delete(remcurNvec, k)

  propModel = curModel.copy()
  propModel.update_global_params(propSS)


  ## Prep for diagnostics
  Nmax = 10

  tracepropELBO = list()
  for riter in xrange(nRefineIters):
    ptargetLP = propModel.calc_local_params(targetData, **LPkwargs)

    propSS -= ptargetSS
    ptargetSS = propModel.get_global_suff_stats(targetData, ptargetLP)
    propSS += ptargetSS
    propModel.update_global_params(propSS)

    propNvec = propSS.getCountVec()
    if riter == 0:
      propBestIDs = np.argsort(-1 * (propNvec - remcurNvec))[:Nmax]
      log('CURRENT  | K=%d | target %12.4f | total=%12.4f' 
          % (ctargetSS.K, ctargetSS.getCountVec().sum(), curNvec.sum()) )
      logPosVector(remcurNvec[propBestIDs], Nmax=Nmax)
      log('PROPOSED | K=%d | target %12.4f | total=%12.4f' 
          % (ptargetSS.K, ptargetSS.getCountVec().sum(), propNvec.sum()))

    logPosVector(propNvec[propBestIDs], Nmax=Nmax)

    propELBO = propModel.calc_evidence(targetData, ptargetSS, ptargetLP)
    tracepropELBO.append(propELBO)
    if propELBO >= curELBO:
      break

  didAccept = 0
  if propELBO >= curELBO:
    didAccept = 1

  Plan['didAccept'] = didAccept
  Plan['tracepropELBO'] = tracepropELBO
  if didAccept:
    newModel = propModel
    newSS = propSS
    Plan['targetSS'] = ptargetSS
    Plan['ELBO'] = propELBO

    ## Make batch-specific adjustments
    Plan['ptargetSSByBatch'] = dict()
    for batchID in Plan['targetSSByBatch']:
      docIDs = np.flatnonzero(Plan['batchIDs'] == batchID)
      Data_b = targetData.select_subset_by_mask(docIDs, doTrackFullSize=False)
      ptargetLP_b = propModel.allocModel.selectSubsetLP(targetData, ptargetLP, docIDs)
      ptargetSS_b = propModel.get_global_suff_stats(Data_b, ptargetLP_b)
      Plan['ptargetSSByBatch'][batchID] = ptargetSS_b

  else:
    Plan['ELBO'] = curELBO
    newModel = curModel
    newSS = curSS

  return newModel, newSS, Plan
