import numpy as np
from collections import defaultdict

def selectCandidateDeleteComps(
        hmodel, SS,
        MoveRecordsByUID=dict(),
        MovePlans=dict(),
        lapFrac=0,
        **DArgs):
    ''' Select specific comps to target with delete move.

    Returns
    -------
    MovePlans : dict, with fields
    * d_targetUIDs : list of ints
    '''
    countVec = SS.getCountVec()
    countVec = np.maximum(countVec, 1e-100)
    ScoreVec = -1.0 / countVec * \
        hmodel.obsModel.calc_evidence(None, SS, None, returnVec=1)

    ScoreByEligibleUID = dict()
    for ii, uid in enumerate(SS.uids):
        if uid in MovePlans['BirthTargetUIDs']:
            continue
        size = countVec[ii]
        if uid not in MoveRecordsByUID:
            MoveRecordsByUID[uid] = defaultdict(int)
        tooSmall = size <= DArgs['b_minAtomCountToTarget']
        hasFailureRecord_Birth = MoveRecordsByUID[uid]['b_nFailRecent'] > 0
        hasFailureRecord_Delete = MoveRecordsByUID[uid]['d_nFailRecent'] > 0
        if (tooSmall or hasFailureRecord_Birth):
            if not hasFailureRecord_Delete:
                ScoreByEligibleUID[uid] = ScoreVec[ii]

    UIDs = [x for x in ScoreByEligibleUID.keys()]
    Scores = np.asarray([x for x in ScoreByEligibleUID.values()])

    # Prioritize which comps to target
    # Keeping the first nToKeep, as ranked by the Ldata score
    keepIDs = np.argsort(-1 * Scores)[:1]
    MovePlans['d_targetUIDs'] = [UIDs[s] for s in keepIDs]
    return MovePlans
