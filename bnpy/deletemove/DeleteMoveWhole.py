import numpy as np

from DeleteLPUtil import deleteCompFromResp_Renorm
from DeleteLPUtil import deleteCompFromResp_SoftEv
from DeleteLPUtil import deleteCompFromResp_SoftEvOverlap

from DeleteLogger import log, logPosVector, logProbVector

def runDeleteMove_Whole(Data, hmodel, SS, LP, curELBO, 
                        deleteCompID=None,
                        deleteRespStrategy='softev',
                        Plan=None,
                        nRefineIters=0,
                        **kwargs):
  ''' Propose candidate model with one less comp and accept if ELBO improves.

      Returns
      --------
      hmodel
      SS
      LP
      Info
  '''
  intromsg = 'Proposal: del comp %d of size %d via %s' \
                % (deleteCompID, 
                   SS.getCountVec()[deleteCompID],
                   deleteRespStrategy)
  log(intromsg)

  assert deleteCompID is not None

  ## Construct resp matrix with deleted comp
  if deleteRespStrategy == 'softevoverlap':
    obsLP = hmodel.obsModel.calc_local_params(Data)
    SoftEv = obsLP['E_log_soft_ev']
    respDel = deleteCompFromResp_SoftEvOverlap(LP['resp'], SoftEv, Data, SS,
                                        deleteCompID)
  elif deleteRespStrategy == 'softev':
    obsLP = hmodel.obsModel.calc_local_params(Data)
    SoftEv = obsLP['E_log_soft_ev']
    respDel = deleteCompFromResp_SoftEv(LP['resp'], SoftEv, deleteCompID)
  elif deleteRespStrategy == 'renorm':
    respDel = deleteCompFromResp_Renorm(LP['resp'], deleteCompID)
  else:
    raise ValueError('Unrecognized deleteRespStrategy: '+ deleteRespStrategy)

  ## Construct propLP dict
  propLP = dict(resp=respDel)
  if hasattr(hmodel.allocModel, 'initLPFromResp'):
    propLP = hmodel.allocModel.initLPFromResp(Data, propLP,
                                              deleteCompID=deleteCompID)

  ## Run update steps to make proposed model  
  propSS = hmodel.get_global_suff_stats(Data, propLP, doPrecompEntropy=1)
  propModel = hmodel.copy()
  propModel.update_global_params(propSS)
  propELBO = propModel.calc_evidence(SS=propSS)

  tracepropELBO = list()
  tracepropELBO.append(propELBO)

  Nmax = 10
  curNvec = SS.getCountVec()
  remcurNvec = np.delete(curNvec, deleteCompID)
  propNvec = propSS.getCountVec()

  propBestIDs = np.argsort(-1 * (propNvec - remcurNvec))[:Nmax]
  curBestIDs = propBestIDs.copy()
  curBestIDs[propBestIDs >= deleteCompID] + 1

  logPosVector(propBestIDs, Nmax=Nmax)

  curBeta = hmodel.allocModel.get_active_comp_probs()
  remcurBeta = np.delete(curBeta, deleteCompID)
  propBeta = propModel.allocModel.get_active_comp_probs()
  log('CURRENT PROBABILITIES')
  logProbVector(remcurBeta[propBestIDs], Nmax=Nmax)
  log('PROPOSED')
  logProbVector(propBeta[propBestIDs], Nmax=Nmax)
  logProbVector((propBeta-remcurBeta)[propBestIDs], Nmax=Nmax)

  log('CURRENT COUNT VEC')
  logPosVector(remcurNvec[propBestIDs], Nmax=Nmax)

  log('PROPOSED')
  logPosVector(propNvec[propBestIDs], Nmax=Nmax)
  logPosVector((propNvec-remcurNvec)[propBestIDs], Nmax=Nmax)
  assert np.allclose(propSS.getCountVec().sum(), 
                     SS.getCountVec().sum())


  log('ELBO ---------')
  log('   cur %.6f' % (curELBO))
  log('  prop %.6f  change %.6f' % (propELBO, propELBO-curELBO))

  scaleF = hmodel.obsModel.getDatasetScale(SS)
  curELBOdata = hmodel.obsModel.calc_evidence(Data, SS, LP)/scaleF
  propELBOdata = propModel.obsModel.calc_evidence(Data, propSS, propLP)/scaleF

  log('data ELBO -----')
  log('   cur %.6f' % (curELBOdata))
  log('  prop %.6f  change %.6f' % (propELBOdata, propELBOdata-curELBOdata))

  didAccept = 0
  if propELBO >= curELBO:
    didAccept = 1

  elif nRefineIters > 0:
    for rr in range(nRefineIters):
      propLP = propModel.calc_local_params(Data, propLP)
      propSS = propModel.get_global_suff_stats(Data, propLP, doPrecompEntropy=1)

      propModel.update_global_params(propSS)
      propELBO = propModel.calc_evidence(SS=propSS)

      logPosVector(propSS.getCountVec()[propBestIDs], Nmax=Nmax)
      log('prop ELBO %.6e' % (propELBO))
      tracepropELBO.append(propELBO)
      if propELBO >= curELBO:
        didAccept = 1
        break

  Info = dict(didAccept=didAccept)
  Info['tracepropELBO'] = tracepropELBO
  if didAccept:
    hmodel = propModel
    SS = propSS
    LP = propLP
    Info['ELBO'] = propELBO
  
  else:
    Info['ELBO'] = curELBO

  return hmodel, SS, LP, Info
