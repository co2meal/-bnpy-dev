'''
'''
import numpy as np

import BirthCreate
import BirthRefine
import BirthCleanup
from BirthProposalError import BirthProposalError
import VizBirth

def run_birth_move(bigModel, bigSS, freshData, **kwargsIN):
  ''' Run birth move on provided target data, creating up to Kfresh new comps

      Returns
      -------
      bigmodel
      bigSS
      MoveInfo
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
    if kwargs['expandOrder'] == 'expandThenRefine':
      xbigModel, xbigSS, xfreshSS, AI, RI = BirthRefine.expand_then_refine(
                                           freshModel, freshSS, freshData,    
                                               bigModel, bigSS, **kwargs)
    else:
      raise NotImplementedError('TODO')
      # BirthRefine.refine_then_expand()

    assert xbigModel.obsModel.K == xbigSS.K

    if kwargs['birthVerifyELBOIncrease']:
      assert xfreshSS.hasELBOTerms()

      curbigModel = bigModel.copy()
      nStep = 3
      for step in range(nStep):
        doELBO = (step == nStep-1) # only on last step
        curfreshLP = curbigModel.calc_local_params(freshData)
        curfreshSS = curbigModel.get_global_suff_stats(freshData, curfreshLP,
                                                       doPrecompEntropy=doELBO)
        if not doELBO: # all but the last step
          curbigModel.update_global_params(bigSS + curfreshSS)
   
      curELBO  = curbigModel.calc_evidence(SS=curfreshSS)
      propELBO = xbigModel.calc_evidence(SS=xfreshSS)

      # Sanity check
      # TODO: type check to avoid this on Gauss models
      if propELBO > 0 and curELBO < 0:
        didPass = False
        ELBOmsg = " propEv %.4e is INSANE!" % (propELBO)
      else:
        percDiff = (propELBO - curELBO)/np.abs(curELBO)
        didPass = propELBO > curELBO and percDiff > 0.0001      
        ELBOmsg = " propEv %.4e | curEv %.4e" % (propELBO, curELBO)
    else:
      didPass = True
      propELBO = None
      curELBO = None
      ELBOmsg = ''

    # Visualize, if desired
    Kcur = bigSS.K
    Ktotal = xbigSS.K
    birthCompIDs = range(Kcur, Ktotal)
    if 'doVizBirth' in kwargs and kwargs['doVizBirth']:
      VizBirth.viz_birth_proposal(bigModel, xbigModel, birthCompIDs,
                                  curELBO=curELBO, propELBO=propELBO, **kwargs)

    # Reject. Abandon the move.
    if not didPass:
      msg = "BIRTH failed. No improvement over current model." + ELBOmsg
      raise BirthProposalError(msg)

    assert xbigModel.obsModel.K == xbigSS.K
    ### Create dict of info about this birth move
    msg = 'BIRTH: %d fresh comps. %s.' % (len(birthCompIDs), ELBOmsg)
    MoveInfo = dict(didAddNew=True,
                    msg=msg,
                    AdjustInfo=AI, ReplaceInfo=RI,
                    modifiedCompIDs=[],
                    birthCompIDs=birthCompIDs,
                    )

    assert not xbigSS.hasELBOTerms()
    assert not xbigSS.hasMergeTerms()
    xfreshSS.removeELBOTerms()
    if kwargs['birthRetainExtraMass']:
      MoveInfo['extraSS'] = xfreshSS
      MoveInfo['modifiedCompIDs'] = range(Ktotal)
    else:
      # Restore xbigSS to same scale as original "big" dataset
      xbigSS -= xfreshSS
      assert np.allclose(xbigSS.N.sum(), bigSS.N.sum())

    if bigSS.hasMergeTerms():
      MergeTerms = bigSS._MergeTerms.copy()
      MergeTerms.insertEmptyComps(Ktotal-Kcur)
      xbigSS.restoreMergeTerms(MergeTerms)
    if bigSS.hasELBOTerms():
      ELBOTerms = bigSS._ELBOTerms.copy()
      ELBOTerms.insertEmptyComps(Ktotal-Kcur)
      if AI is not None:
        for key in AI:
          if hasattr(ELBOTerms, key):
            arr = getattr(ELBOTerms, key) + bigSS.nDoc * AI[key]
            ELBOTerms.setField(key, arr, dims='K')
        for key in RI:
          if hasattr(ELBOTerms, key):
            ELBOTerms.setField(key, bigSS.nDoc * RI[key], dims=None)
      xbigSS.restoreELBOTerms(ELBOTerms)

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

