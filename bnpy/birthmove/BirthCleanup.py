import numpy as np

from ..deletemove.DeleteMoveBagOfWords import construct_LP_with_comps_removed
from BirthProposalError import BirthProposalError

def delete_comps_from_expanded_model_to_improve_ELBO(Data, 
                              xbigModel, xbigSS, 
                              xfreshSS, xfreshLP=None,
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

  origIDs = range(0, K)

  if K == 1:
    return xbigModel, xbigSS, xfreshSS, origIDs

  nDoc = xbigSS.nDoc
  wc = xbigSS.WordCounts.sum()

  xfreshELBO = xbigModel.calc_evidence(SS=xfreshSS)

  for k in reversed(range(Korig, K)):
    assert xbigSS.nDoc == nDoc
    assert np.allclose(xbigSS.WordCounts.sum(), wc)

    if kwargs['cleanupDeleteViaLP']:
      rbigModel, rbigSS, rfreshSS, rfreshELBO, rfreshLP = _make_candidate_LP(
                                                    xbigModel, Data,
                                                    xbigSS, xfreshSS, xfreshLP, 
                                                    k)
    else:
      rbigModel, rbigSS, rfreshSS, rfreshELBO = _make_candidate(
                                                    xbigModel, Data,
                                                    xbigSS, xfreshSS, 
                                                    k)

    # If ELBO has improved, set current model to delete component k
    didAccept = False
    if rfreshELBO >= xfreshELBO:
      xbigSS = rbigSS
      xfreshSS = rfreshSS
      xbigModel = rbigModel
      xfreshELBO = rfreshELBO
      if kwargs['cleanupDeleteViaLP']:
        xfreshLP = rfreshLP
      didAccept = True
      del origIDs[k]

    if xfreshSS.K == 1:
      break
    ### end loop over comps to delete

  if xbigSS.K == Korig:
    msg = "BIRTH failed. Deleting all new comps improves ELBO."
    raise BirthProposalError(msg)

  if didAccept:
    # Make sure that final model has correct scale
    xbigModel.update_global_params(xbigSS + xfreshSS)

  return xbigModel, xbigSS, xfreshSS, origIDs


def _make_candidate(xbigModel, Data, xbigSS, xfreshSS, k):
  rbigModel = xbigModel.copy()
  rbigSS = xbigSS.copy()
  rfreshSS = xfreshSS.copy()

  rbigSS.removeComp(k)
  rfreshSS.removeComp(k)
    
  rSS = rbigSS + rfreshSS
  rbigModel.update_global_params(rSS)

  rLP = rbigModel.calc_local_params(Data)
  rfreshSS = rbigModel.get_global_suff_stats(Data, rLP, doPrecompEntropy=True)
  rfreshELBO = rbigModel.calc_evidence(SS=rfreshSS)

  return rbigModel, rbigSS, rfreshSS, rfreshELBO


def _make_candidate_LP(xbigModel, Data, xbigSS, xfreshSS, xfreshLP, k):
  rfreshLP = construct_LP_with_comps_removed(Data, xbigModel,
                                             compIDs=[k], LP=xfreshLP)

  rfreshSS = xbigModel.get_global_suff_stats(Data, rfreshLP, 
                                             doPrecompEntropy=True)

  rbigModel = xbigModel.copy()
  rbigSS = xbigSS.copy()
  rbigSS.removeComp(k)
    
  rbigModel.update_global_params(rbigSS + rfreshSS)
  rfreshELBO = rbigModel.calc_evidence(SS=rfreshSS)

  return rbigModel, rbigSS, rfreshSS, rfreshELBO, rfreshLP


"""
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
"""

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
