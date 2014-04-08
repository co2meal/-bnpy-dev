import numpy as np

from BirthProposalError import BirthProposalError
import BirthCleanup

def expand_then_refine(freshModel, freshSS, freshData,
                         bigModel, bigSS,
                         **kwargs):
  ''' Create expanded model with K + K' comps, 
        then refine components K+1, K+2, ... K+K' via several VB iterations

      Guarantees that the original comps of bigModel.obsModel are not altered.

      Returns
      -------
      xbigModel : HModel with K + Kfresh comps
               * allocModel has scale bigSS + freshSS
               * obsModel has scale bigSS + freshSS
      xbigSS : SuffStatBag with K + Kfresh comps
                * has scale bigSS + freshSS
      xfreshSS : SuffStatBag with K + Kfresh comps
                * has scale freshSS
      
  '''
  xbigSS = bigSS.copy(includeELBOTerms=False, includeMergeTerms=False)
  if kwargs['expandAdjustSuffStats'] \
        and hasattr(freshModel.allocModel, 'insertCompsIntoSuffStatBag'):
    xbigSS, AInfo, RInfo = freshModel.allocModel.insertCompsIntoSuffStatBag(
                                                            xbigSS, freshSS)
  else:
    xbigSS.insertComps(freshSS)
    AInfo = None
    RInfo = None

  ### Create expanded model, K + Kfresh comps
  xbigModel = freshModel.copy()
  xbigModel.update_global_params(xbigSS)

  xbigSS.subtractSpecificComps(freshSS, range(bigSS.K, bigSS.K + freshSS.K))
  xbigModel, xfreshSS, xfreshLP, origIDs = refine_expanded_model_with_VB_iters(
                                xbigModel, freshData, 
                                xbigSS=xbigSS, Korig=bigSS.K, **kwargs)

  from IPython import embed; embed()
  assert np.allclose(xbigSS.sumLogPiUnused,
                      RInfo['sumLogPiUnused'] * bigSS.nDoc)
  assert np.allclose(xbigSS.sumLogPiActive[-freshSS.K:], 
                      AInfo['sumLogPiActive'][-freshSS.K:] * bigSS.nDoc)
  
  if hasattr(xfreshSS, 'nDoc'):
    assert xbigSS.nDoc == bigSS.nDoc
    assert xfreshSS.nDoc == freshData.nDoc

  if kwargs['cleanupDeleteToImprove']:
    if xfreshSS.hasELBOTerms():
      xfreshELBO = xbigModel.calc_evidence(SS=xfreshSS)
    else:
      xfreshELBO = None
    xbigModel, xbigSS, xfreshSS, origIDs = \
              BirthCleanup.delete_comps_from_expanded_model_to_improve_ELBO(
                                  freshData, xbigModel, 
                                  xbigSS, xfreshSS,
                                  Korig=bigSS.K)
  
  if hasattr(xfreshSS, 'nDoc'):
    assert xbigSS.nDoc == bigSS.nDoc
    assert xfreshSS.nDoc == freshData.nDoc
  xbigSS += xfreshSS

  return xbigModel, xbigSS, xfreshSS, AInfo, RInfo


def refine_expanded_model_with_VB_iters(xbigModel, freshData,
                                        xbigSS=None, Korig=0, **kwargs):
  ''' Execute multiple local/global update steps for the current model
      
      Args
      --------
      xbigSS : SuffStatBag, with K + Kfresh comps,
                                 scale equal to bigData only

      Returns
      --------
      model : HModel, with K + Kfresh comps
                      scale equal to bigData + freshData
      freshSS : SuffStatBag, with K + Kfresh comps
                      scale equal to freshData
      freshLP : dict of local parameters for freshData


      Updates (in-place)
      ----------
      xbigSS : SuffStatBag, with K + Kfresh comps
                       scale with equal to bigData only
  '''
  origIDs = np.arange(Korig, xbigSS.K)

  for riter in xrange(kwargs['refineNumIters']):
    xfreshLP = xbigModel.calc_local_params(freshData)
    xfreshSS = xbigModel.get_global_suff_stats(freshData, xfreshLP)

    # For all but last iteration, attempt removing empty topics
    if kwargs['cleanupDeleteEmpty'] and riter < kwargs['refineNumIters'] - 1:
      for posID, k in enumerate(reversed(range(Korig, xfreshSS.K))):
        if xfreshSS.N[k] < kwargs['cleanupMinSize']:
          xfreshSS.removeComp(k)
          xbigSS.removeComp(k)
          del origIDs[posID]

    if xfreshSS.K == Korig:
      msg = "BIRTH failed. No new comps above cleanupMinSize."
      raise BirthProposalError(msg)

    xbigSS += xfreshSS
    xbigModel.allocModel.update_global_params(xbigSS)
    xbigModel.obsModel.update_global_params(xbigSS)
    xbigSS -= xfreshSS

  xfreshLP = xbigModel.calc_local_params(freshData)
  xfreshSS = xbigModel.get_global_suff_stats(freshData, xfreshLP,
                                               doPrecompEntropy=True)
  return xbigModel, xfreshSS, xfreshLP, origIDs
