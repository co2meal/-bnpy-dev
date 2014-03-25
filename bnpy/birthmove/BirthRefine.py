from BirthProposalError import BirthProposalError
import BirthCleanup

def expand_then_refine(freshModel, freshSS, freshData,
                         bigModel, bigSS,
                         **kwargs):
  ''' Create expanded model with K + K' comps, 
        then refine components K+1, K+2, ... K+K' via several VB iterations
  '''
  xSS = bigSS.copy()
  xSS.insertComps(freshSS)
  xModel = freshModel # no need to copy!

  xModel.allocModel.update_global_params(xSS)
  xModel.obsModel.update_global_params(xSS, comps=range(bigSS.K, xSS.K))

  xModel, xSS, xLP = refine_with_multiple_VB_iters(
                              xModel, freshData, Korig=bigSS.K, **kwargs)

  if kwargs['cleanupDeleteToImprove']:
    xModel, xSS, xLP, xELBO = BirthCleanup.delete_comps_to_improve_ELBO(
                                  freshData, xModel, 
                                  LP=xLP, SS=xSS, ELBO=xELBO,
                                  Korig=bigSS.K)

  # TODO 
  # Compare to competitor model using only K existing comps for freshData
  return xModel, xSS


def refine_with_multiple_VB_iters(model, freshData, Korig=0, **kwargs):
  ''' Execute multiple local/global update steps for the current model
      
      If Korig provided, only components Korig+1, ... K-1, K are altered.
  '''
  for _ in xrange(kwargs['refineNumIters']):
    LP = model.calc_local_params(freshData)
    SS = model.get_global_suff_stats(freshData, LP)

    if kwargs['cleanupDeleteEmpty']:
      for k in reversed(range(Korig, SS.K)):
        if SS.N[k] < kwargs['cleanupMinSize']:
          SS.removeComp(k)
          del model.obsModel.comp[k]

      if SS.K < model.allocModel.K:
        model.obsModel.K = SS.K

    if SS.K == Korig:
      msg = "BIRTH failed. No new comps above cleanupMinSize."
      raise BirthProposalError(msg)

    model.allocModel.update_global_params(SS)
    model.obsModel.update_global_params(SS, comps=range(Korig, SS.K))

  LP = model.calc_local_params(freshData)
  SS = model.get_global_suff_stats(freshData, LP, doPrecompEntropy=True)
  return model, SS, LP
