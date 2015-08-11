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
    # Determine uids eligible for either
    # a) being deleted, or
    # b) "absorbing" deleted comps.
    eligibleUIDs = set(SS.uids)
    if 'm_UIDPairs' in MovePlans:
        for (uidA, uidB) in MovePlans['m_UIDPairs']:
            eligibleUIDs.discard(uidA)
            eligibleUIDs.discard(uidB)
    if 'BirthTargetUIDs' in MovePlans:
        for uid in MovePlans['BirthTargetUIDs']:
            eligibleUIDs.discard(uid)
    if len(eligibleUIDs) < 3:
        # Need >2 states to absorb, and one to remove
        return dict()

    # Compute score for each eligible state
    countVec = np.maximum(SS.getCountVec(), 1e-100)
    ScoreVec = -1.0 / countVec * \
        hmodel.obsModel.calc_evidence(None, SS, None, returnVec=1)
    ScoreByEligibleUID = dict()
    for uid in eligibleUIDs:
        k = SS.uid2k(uid)
        size = countVec[k]
        if uid not in MoveRecordsByUID:
            MoveRecordsByUID[uid] = defaultdict(int)
        if 'b_minAtomCountToTarget' in DArgs:
            tooSmall = size <= DArgs['b_minAtomCountToTarget']
        else:
            tooSmall = True
        hasFailureRecord_Birth = MoveRecordsByUID[uid]['b_nFailRecent'] > 0
        hasFailureRecord_Delete = MoveRecordsByUID[uid]['d_nFailRecent'] > 0
        if (tooSmall or hasFailureRecord_Birth):
            if not hasFailureRecord_Delete:
                ScoreByEligibleUID[uid] = ScoreVec[k]
    UIDs = [x for x in ScoreByEligibleUID.keys()]
    if len(UIDs) == 0:
        return dict()
    # Select the single state to target
    # by taking the one with highest score
    Scores = np.asarray([x for x in ScoreByEligibleUID.values()])
    targetUID = UIDs[np.argmax(Scores)]
    MovePlans['d_targetUIDs'] = [targetUID]
    # Also track all comps eligible to receive its transfer mass
    eligibleUIDs.discard(targetUID)
    MovePlans['d_absorbingUIDSet'] = eligibleUIDs
    return MovePlans
