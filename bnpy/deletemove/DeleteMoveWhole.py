from DeleteLPUtil import deleteCompFromResp


def runDeleteMove_Whole(Data, hmodel, SS, LP, curELBO, 
                        deleteCompID=None,
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
  assert deleteCompID is not None

  ## Construct resp matrix with deleted comp
  respDel = deleteCompFromResp(LP['resp'], deleteCompID)

  ## Construct propLP dict
  propLP = dict(resp=respDel)
  if hasattr(hmodel.allocModel, 'initLPFromResp'):
    propLP = hmodel.initLPFromResp(propLP, deleteCompID=deleteCompID)

  ## Run update steps to make proposed model  
  propSS = hmodel.get_global_suff_stats(Data, propLP, doPrecompEntropy=1)
  propModel = hmodel.copy()
  propModel.update_global_params(propSS)
  propELBO = propModel.calc_evidence(SS=propSS)

  didAccept = 0
  if propELBO >= curELBO:
    didAccept = 1

  elif nRefineIters > 0:
    for rr in range(nRefineIters):
      propLP = propModel.calc_local_params(Data, propLP)
      propSS = propModel.get_global_suff_stats(Data, propLP, doPrecompEntropy=1)
      propModel.update_global_params(propSS)
      propELBO = propModel.calc_evidence(SS=propSS)
      if propELBO >= curELBO:
        didAccept = 1
        break

  Info = dict(didAccept=didAccept)
  if didAccept:
    hmodel = propModel
    SS = propSS
    LP = propLP
    Info['ELBO'] = propELBO
  
  else:
    Info['ELBO'] = curELBO

  return hmodel, SS, LP, Info