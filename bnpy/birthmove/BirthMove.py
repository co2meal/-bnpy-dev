'''
'''
import numpy as np

import BirthCreate
import BirthRefine
import BirthCleanup
from BirthProposalError import BirthProposalError

def run_birth_move(bigModel, bigSS, freshData, **kwargsIN):
  ''' Run birth move on provided target data, creating up to Kfresh new comps
  '''
  kwargs = dict(**kwargsIN) # make local copy!
  origids = dict( bigModel=id(bigModel), bigSS=id(bigSS) )

  try:
    if bigSS is None:
      msg = "BIRTH failed. SS must be valid SuffStatBag, not None."
      raise BirthProposalError(msg)

    if bigSS.K + kwargs['Kfresh'] > kwargs['Kmax']:
      kwargs['Kfresh'] = kwargs['Kmax'] - bigSS.K
    if kwargs['Kfresh'] < 1:
      msg = "BIRTH failed. Reached upper limit of Kmax=%d comps."
      msg = msg % (kwargs['Kmax'])
      raise BirthProposalError(msg)

    # Create freshModel, freshSS, both with Kfresh comps
    #  freshSS has scale freshData
    #  freshModel has arbitrary scale
    freshModel, freshSS = BirthCreate.create_model_with_new_comps(
                                            bigModel, bigSS, freshData,
                                            **kwargs)
    
    # Create xbigModel and xbigSS, with K + Kfresh comps
    #      freshData can be assigned to any of the K+Kfresh comps
    #      so, any of the K+Kfresh comps may be changed 
    #          but original comps won't lose influence of bigSS
    #  xbigSS has scale bigData + freshData
    #  xbigModel has scale bigData + freshData
    if kwargs['expandorder'] == 'expandThenRefine':
      xbigModel, xbigSS, xfreshSS = BirthRefine.expand_then_refine(
                                           freshModel, freshSS, freshData,    
                                               bigModel, bigSS, **kwargs)
    else:
      raise NotImplementedError('TODO')
      # BirthRefine.refine_then_expand()

    assert xbigModel.obsModel.K == xbigSS.K

    if kwargs['birthVerifyELBOIncrease']:
      assert xfreshSS.hasELBOTerms()
      propELBO = xbigModel.calc_evidence(SS=xfreshSS)
      xfreshSS.removeELBOTerms()

      curfreshLP = bigModel.calc_local_params(freshData)
      curfreshSS = bigModel.get_global_suff_stats(freshData, curfreshLP,
                                                          doPrecompEntropy=True)
      # Use already allocated model as placeholder
      #  for tracking what the current model would look like if 
      #  its parameters were updated to include freshData
      curbigModel = freshModel
      curbigModel.update_global_params(bigSS + curfreshSS)

      # Compare ELBO
      curELBO  = curbigModel.calc_evidence(SS=curfreshSS)
      ELBOstr = " propEv %.4e | curEv %.4e" % (propELBO, curELBO)
      print ELBOstr

      if propELBO > curELBO:
        pass # Accepted!
      else:
        # Reject. Abandon the move.
        msg = "BIRTH failed. No improvement over current model."
        msg += " propEv %.4e | curEv %.4e" % (propELBO, curELBO)
        raise BirthProposalError(msg)

    assert xbigModel.obsModel.K == xbigSS.K
    ### Create dict of info about this birth move
    Kcur = bigSS.K
    Ktotal = xbigSS.K
    birthCompIDs = range(Kcur, Ktotal)
    MoveInfo = dict(didAddNew=True,
                    msg='BIRTH: %d fresh comps' % (len(birthCompIDs)),
                    modifiedCompIDs=[],
                    birthCompIDs=birthCompIDs,
                    )
     
    if kwargs['birthRetainExtraMass']:
      MoveInfo['extraSS'] = xfreshSS
      MoveInfo['modifiedCompIDs'] = range(Ktotal)
    else:
      # Restore xbigSS to same scale original "big" dataset
      xbigSS -= xfreshSS
      assert np.allclose(xbigSS.N.sum(), bigSS.N.sum())

    if bigSS.hasMergeTerms():
      MergeTerms = bigSS._MergeTerms.copy()
      MergeTerms.insertEmptyComps(Ktotal-Kcur)
      xbigSS.restoreMergeTerms(MergeTerms)
    if bigSS.hasELBOTerms():
      ELBOTerms = bigSS._ELBOTerms.copy()
      ELBOTerms.insertEmptyComps(Ktotal-Kcur)
      xbigSS.restoreELBOTerms(ELBOTerms)

    if 'doVizBirth' in kwargs and kwargs['doVizBirth']:
      viz_birth_proposal(curModel, xbigModel, birthCompIDs, **kwargs)

    return xbigModel, xbigSS, MoveInfo
  except BirthProposalError, e:
    ### Clean-up code when birth proposal fails for any reason, including:
    #  * user-specified Kmax limit reached
    #  * delete phase removed all new components

    # Verify guarantees that input model and input suff stats haven't changed
    assert origids['bigModel'] == id(bigModel)
    assert origids['bigSS'] == id(bigSS)

    # Return failure info
    MoveInfo = dict(didAddNew=False,
                    msg=str(e),
                    modifiedCompIDs=[],
                    birthCompIDs=[])
    return bigModel, bigSS, MoveInfo

