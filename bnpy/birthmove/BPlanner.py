import numpy as np
from collections import defaultdict

import BLogger

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
    BLogger.pprint('PLANNING BIRTH at lap %.2f ==========' % (lapFrac))
    if 'BirthTargetUIDs' in MovePlans:
        if len(MovePlans['BirthTargetUIDs']) > 0:
            uidStr = BLogger.vec2str(MovePlans['BirthTargetUIDs'])
            BLogger.pprint(
                'Keeping existing targetUIDs: ' + uidStr)
            return MovePlans

    countVec = SS.getCountVec()
    countVec = np.maximum(countVec, 1e-100)
    ScoreVec = -1.0 / countVec * \
        hmodel.obsModel.calc_evidence(None, SS, None, returnVec=1)
    ScoreVec[countVec < 1] = np.nan

    uidsTooSmall = list()
    uidsWithFailRecord = list()
    ScoreByEligibleUID = dict()
    for ii, uid in enumerate(SS.uids):
        if uid not in MoveRecordsByUID:
            MoveRecordsByUID[uid] = defaultdict(int)
        size = countVec[ii]
        oldsize = MoveRecordsByUID[uid]['b_latestCount']
        nFailRecent = MoveRecordsByUID[uid]['b_nFailRecent']

        bigEnough = size > BArgs['b_minAtomCountToTarget']
        if oldsize == 0 or nFailRecent == 0:
            hasFailureRecord = False
        else:
            sizePercDiff = np.abs(size - oldsize)/np.abs(oldsize)
            if sizePercDiff > BArgs['b_minChangeInAtomCountToReactivate']:
                hasFailureRecord = False
            else:
                hasFailureRecord = True

        if not bigEnough:
            uidsTooSmall.append((uid, size))
            continue
        if hasFailureRecord:
            uidsWithFailRecord.append((uid, nFailRecent))
            continue
        ScoreByEligibleUID[uid] = ScoreVec[ii]
    BLogger.pprint(
        'UIDs disqualified as too small. Required size >%d.' % (
            BArgs['b_minAtomCountToTarget']), 'debug')
    BLogger.pprint(
        '  ' + BLogger.vec2str([u[0] for u in uidsTooSmall]), 'debug')
    BLogger.pprint(
        '  ' + BLogger.vec2str([u[1] for u in uidsTooSmall]), 'debug')
    BLogger.pprint(
        'UIDs disqualified for past failures.', 'debug')
    BLogger.pprint(
        '  ' + BLogger.vec2str([u[0] for u in uidsWithFailRecord]), 'debug')
    BLogger.pprint(
        '  ' + BLogger.vec2str([u[1] for u in uidsWithFailRecord]), 'debug')

    # Finalize list of eligible UIDs
    UIDs = [x for x in ScoreByEligibleUID.keys()]
    BLogger.pprint(
        'Eligible UIDs: ' + BLogger.vec2str(UIDs), 'debug')
    if len(UIDs) == 0:
        return MovePlans
    # Finalize corresponding scores
    Scores = np.asarray([x for x in ScoreByEligibleUID.values()])
    BLogger.pprint(
        'Eligible Scores: ' + BLogger.vec2str(Scores), 'debug')
    # Figure out how many new states we can target this round.
    totalnewK = BArgs['b_Kfresh'] * Scores.size
    maxnewK = BArgs['Kmax'] - SS.K
    # Prioritize which comps to target
    # Keeping the first nToKeep, as ranked by the Ldata score
    nToKeep = maxnewK // BArgs['b_Kfresh']
    keepIDs = np.argsort(-1 * Scores)[:nToKeep]
    MovePlans['BirthTargetUIDs'] = [UIDs[s] for s in keepIDs]
    BLogger.pprint(
        'Target UIDs: ' + BLogger.vec2str(MovePlans['BirthTargetUIDs']))
    return MovePlans
    



