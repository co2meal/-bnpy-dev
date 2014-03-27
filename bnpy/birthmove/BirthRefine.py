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
  xbigSS.insertComps(freshSS)

  ### Create expanded model, K + Kfresh comps
  xbigModel = freshModel.copy()
  xbigModel.update_global_params(xbigSS)

  xbigSS.subtractSpecificComps(freshSS, range(bigSS.K, bigSS.K + freshSS.K))
  xbigModel, xfreshSS, xfreshLP = refine_expanded_model_with_VB_iters(
                                xbigModel, freshData, 
                                xbigSS=xbigSS, Korig=bigSS.K, **kwargs)

  if hasattr(xfreshSS, 'nDoc'):
    assert xbigSS.nDoc == bigSS.nDoc
    assert xfreshSS.nDoc == freshData.nDoc

  if kwargs['cleanupDeleteToImprove']:
    if xfreshSS.hasELBOTerms():
      xfreshELBO = xbigModel.calc_evidence(SS=xfreshSS)
    else:
      xfreshELBO = None
    xbigModel, xbigSS, xfreshSS = \
              BirthCleanup.delete_comps_from_expanded_model_to_improve_ELBO(
                                  freshData, xbigModel, 
                                  xbigSS, xfreshSS,
                                  Korig=bigSS.K)
  if hasattr(xfreshSS, 'nDoc'):
    assert xbigSS.nDoc == bigSS.nDoc
    assert xfreshSS.nDoc == freshData.nDoc
  xbigSS += xfreshSS

  return xbigModel, xbigSS, xfreshSS


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
  for riter in xrange(kwargs['refineNumIters']):
    xfreshLP = xbigModel.calc_local_params(freshData)
    xfreshSS = xbigModel.get_global_suff_stats(freshData, xfreshLP)

    # For all but last iteration, attempt removing empty topics
    if kwargs['cleanupDeleteEmpty'] and riter < kwargs['refineNumIters'] - 1:
      for k in reversed(range(Korig, xfreshSS.K)):
        if xfreshSS.N[k] < kwargs['cleanupMinSize']:
          xfreshSS.removeComp(k)
          xbigSS.removeComp(k)

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
  return xbigModel, xfreshSS, xfreshLP
