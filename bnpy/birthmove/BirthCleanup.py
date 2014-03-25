import numpy as np

from BirthProposalError import BirthProposalError

def delete_comps_from_expanded_model_to_improve_ELBO(Data, 
                              xbigModel, xbigSS, 
                              xfreshSS, 
                              Korig=0, **kwargs):
  ''' Attempts deleting components K, K-1, K-2, ... Korig,
       keeping (and building on) any proposals that improve the ELBO

     Returns
     ---------
      model : HModel with Knew comps
      SS : SuffStatBag with Knew comps
      ELBO : evidence lower bound for the returned model
  '''  
  K = xbigSS.K
  assert xbigSS.K == xfreshSS.K
  assert xbigModel.obsModel.K == K

  if K == 1:
    return xbigModel, xbigSS, xfreshSS

  nDoc = xbigSS.nDoc
  wc = xbigSS.WordCounts.sum()

  xfreshELBO = xbigModel.calc_evidence(SS=xfreshSS)
  for k in reversed(range(Korig, K)):
    #verify_suffstats_at_desired_scale(xbigSS, nDoc=nDoc, word_count=wc)
    assert xbigSS.nDoc == nDoc
    assert np.allclose(xbigSS.WordCounts.sum(), wc)

    rbigModel = xbigModel.copy()
    rbigSS = xbigSS.copy()
    rfreshSS = xfreshSS.copy()

    rbigSS.removeComp(k)
    rfreshSS.removeComp(k)
    
    rSS = rbigSS + rfreshSS
    rbigModel.allocModel.update_global_params(rSS, mergeCompB=k)
    rbigModel.obsModel.K -= 1
    rbigModel.obsModel.update_global_params(rSS, comps=range(Korig, rbigSS.K))

    rLP = rbigModel.calc_local_params(Data)
    rfreshSS = rbigModel.get_global_suff_stats(Data, rLP, doPrecompEntropy=True)
    rfreshELBO = rbigModel.calc_evidence(SS=rfreshSS)

    # If ELBO has improved, set current model to delete component k
    if rfreshELBO >= xfreshELBO:
      xbigSS = rbigSS
      print '****', xbigSS.WordCounts.sum(), wc
      xfreshSS = rfreshSS
      xbigModel = rbigModel
      xfreshELBO = rfreshELBO
    else:
      print '    ', xbigSS.WordCounts.sum(), wc
    if xfreshSS.K == 1:
      break
    ### end loop over comps to delete

  if xbigSS.K == Korig:
    msg = "BIRTH failed. Deleting all new comps improves ELBO."
    raise BirthProposalError(msg)  
  return xbigModel, xbigSS, xfreshSS


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
  
  return model, SS, LP, ELBO


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
