'''
'''

import BirthCreate
import BirthRefine
import BirthCleanup
from BirthProposalError import BirthProposalError

def run_birth_move(bigModel, bigSS, freshData, freshLP):
  '''
  '''
  try:
    if bigSS is None:
      msg = "BIRTH failed. SS must be valid SuffStatBag, not None."
      raise BirthProposalError(msg)

    # freshModel : Kfresh brand-new comps
    freshModel, freshSS = BirthCreate.create_new_comps(
                                            bigModel, bigSS, freshData,
                                                **kwargs)

    if kwargs['expandorder'] == 'expandthenrefine':
      newModel, newSS, Info = BirthRefine.expand_then_refine(
                                           freshModel, freshSS, freshData,    
                                                 bigModel, bigSS, **kwargs)
    else:
      newModel, newSS, Info = BirthRefine.refine_then_expand(
                                           freshModel, freshSS, freshData,    
                                                 bigModel, bigSS, **kwargs)

    Kfresh = Info['Kfresh']
    Kcur = bigSS.K
    assert newSS.K == Kcur + Kfresh
    birthCompIDs = range(Kcur, Kcur + Kfresh)
    
    MoveInfo = dict(didAddNew=True,
                    msg='BIRTH: %d fresh comps' % (len(birthCompIDs)),
                    modifiedCompIDs=[],
                    birthCompIDs=birthCompIDs,
                    extraSS=freshSS)

    if 'doVizBirth' in kwargs and kwargs['doVizBirth']:
      viz_birth_proposal_2D(curModel, newModel, ktarget, birthCompIDs)

    return newModel, newSS, MoveInfo
  except BirthProposalError, e:
    MoveInfo = dict(didAddNew=False, msg=str(e),
                    birthCompIDs=[], modifiedCompIDs=[])
    return curModel, SS, MoveInfo
