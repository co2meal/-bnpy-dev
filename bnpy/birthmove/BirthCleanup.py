from BirthProposalError import BirthProposalError

def delete_comps_to_improve_ELBO(Data, model, 
                                  SS=None, LP=None, ELBO=None, 
                                  Korig=0, **kwargs):
  ''' Attempts deleting components K, K-1, K-2, ... Korig,
       keeping (and building on) any proposals that improve the ELBO

      * does change allocmodel global params
      * does not alter any obsmodel global params for any comps.

     Returns
     ---------
      model : HModel with Knew comps
      SS : SuffStatBag with Knew comps
      LP : dict of Local params, with Knew comps
      ELBO : evidence lower bound for the returned model
  '''

  if LP is None:
    LP = model.calc_local_params(Data)
  if SS is None:
    SS = model.get_global_suff_stats(Data, LP, doPrecompEntropy=True)
  if ELBO is None:
    ELBO = model.calc_evidence(SS=SS)

  K = SS.K
  if K == 1:
    return model, LP, SS, ELBO

  for k in reversed(range(Korig, K)):
    rmodel = model.copy()
    
    rSS = SS.copy()
    rSS.removeComp(k)
    rmodel.obsModel.K = rSS.K
    rmodel.allocModel.update_global_params(rSS, mergeCompB=k)
    del rmodel.obsModel.comp[k]

    rLP = rmodel.calc_local_params(Data)
    rSS = rmodel.get_global_suff_stats(Data, rLP, doPrecompEntropy=True)
    rELBO = rmodel.calc_evidence(SS=rSS)

    # If ELBO has improved, set current model to delete component k
    if rELBO >= ELBO:
      SS = rSS
      LP = rLP
      model = rmodel
      ELBO = rELBO
    if SS.K == 1:
      break
    ### end loop over comps to delete

  if SS.K == Korig:
    msg = "BIRTH failed. Deleting all new comps improves ELBO."
    raise BirthProposalError(msg)
  
  return model, LP, SS, ELBO


def delete_empty_comps(Data, model, SS=None, 
                                  Korig=0, **kwargs):
  ''' Removes any component K, K-1, K-2, ... Korig that is too small,
        as measured by SS.N[k]    

      * does change allocmodel global params
      * does not alter any obsmodel global params for any comps.

     Returns
     ---------
      model : HModel, modified in-place to remove empty comps
      SS : SuffStatBag, modified in-place to remove empty comps
  '''

  if SS is None:
    LP = model.calc_local_params(Data)
    SS = model.get_global_suff_stats(Data, LP, doPrecompEntropy=True)

  K = SS.K
  for k in reversed(range(Korig, K)):
    if SS.N[k] < kwargs['cleanupMinSize']:
      if SS.K > 1:
        SS.removeComp(k)
        del model.obsModel.comp[k]
      else:
        msg = 'BIRTH failed. Cleanup found all new components empty.'
        raise BirthProposalError(msg)

  if SS.K < model.allocModel.K:
    model.allocModel.update_global_params(SS)
    model.obsModel.K = SS.K

  return model, SS