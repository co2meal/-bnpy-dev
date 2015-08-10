import numpy as np
from collections import defaultdict


def selectTargetCompsForBirth(
        hmodel, SS,
        MoveRecordsByUID=dict(),
        MovePlans=dict(),
        lapFrac=0,
        **BArgs):
    ''' Select specific comps to target with birth move.

    Returns
    -------
    MovePlans : dict, with fields
    * BirthTargetUIDs : list of ints
    '''
    if 'BirthTargetUIDs' in MovePlans and len(MovePlans['BirthTargetUIDs']) > 0:
        return MovePlans

    countVec = SS.getCountVec()
    countVec = np.maximum(countVec, 1e-100)
    ScoreVec = -1.0 / countVec * \
        hmodel.obsModel.calc_evidence(None, SS, None, returnVec=1)
    ScoreVec[countVec < 1] = np.nan

    ScoreByEligibleUID = dict()
    for ii, uid in enumerate(SS.uids):
        if uid not in MoveRecordsByUID:
            MoveRecordsByUID[uid] = defaultdict(int)
        size = countVec[ii]
        oldsize = MoveRecordsByUID[uid]['b_latestCount']
        nFailRecent = MoveRecordsByUID[uid]['b_nFailRecent']

        bigEnough = size > BArgs['b_minAtomCountToTarget']
        if oldsize == 0 or nFailRecent == 0:
            hasNoFailureRecord = True
        else:
            sizePercDiff = np.abs(size - oldsize)/np.abs(oldsize)
            if sizePercDiff > BArgs['b_minChangeInAtomCountToReactivate']:
                hasNoFailureRecord = True
            else:
                hasNoFailureRecord = False

        if not (bigEnough and hasNoFailureRecord):
            continue

        ScoreByEligibleUID[uid] = ScoreVec[ii]

    UIDs = [x for x in ScoreByEligibleUID.keys()]
    Scores = np.asarray([x for x in ScoreByEligibleUID.values()])
    totalnewK = BArgs['b_Kfresh'] * Scores.size
    maxnewK = BArgs['Kmax'] - SS.K

    # Prioritize which comps to target
    # Keeping the first nToKeep, as ranked by the Ldata score
    nToKeep = maxnewK // BArgs['b_Kfresh']
    keepIDs = np.argsort(-1 * Scores)[:nToKeep]
    MovePlans['BirthTargetUIDs'] = [UIDs[s] for s in keepIDs]

    return MovePlans
    



