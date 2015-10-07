import numpy as np
from collections import defaultdict

import BLogger
from bnpy.viz.PrintTopics import vec2str

def selectTargetCompsForBirth(
        hmodel, SS,
        MoveRecordsByUID=dict(),
        MovePlans=dict(),
        lapFrac=0,
        **BArgs):
    ''' Select specific comps to target with birth move.

    TODO avoid comps that have not enough docs in Mult case.

    Returns
    -------
    MovePlans : dict, with fields
    * BirthTargetUIDs : list of ints
    '''
    if 'BirthTargetUIDs' in MovePlans:
        if len(MovePlans['BirthTargetUIDs']) > 0:
            uidStr = vec2str(MovePlans['BirthTargetUIDs'])
            BLogger.pprint(
                'Keeping existing targetUIDs: ' + uidStr)
            return MovePlans

    countVec = SS.getCountVec()
    K = countVec.size
    countVec = np.maximum(countVec, 1e-100)
    ScoreVec = -1.0 / countVec * \
        hmodel.obsModel.calc_evidence(None, SS, None, returnVec=1)
    ScoreVec[countVec < 1] = np.nan

    uidsTooSmall = list()
    uidsWithFailRecord = list()
    eligible_mask = np.zeros(K, dtype=np.bool8)
    for ii, uid in enumerate(SS.uids):
        if uid not in MoveRecordsByUID:
            MoveRecordsByUID[uid] = defaultdict(int)
        size = countVec[ii]
        oldsize = MoveRecordsByUID[uid]['b_latestCount']
        nFailRecent = MoveRecordsByUID[uid]['b_nFailRecent']

        bigEnough = size >= BArgs['b_minNumAtomsForTargetComp']
        if oldsize == 0 or nFailRecent == 0:
            hasFailureRecord = False
        else:
            sizePercDiff = np.abs(size - oldsize)/np.abs(oldsize)
            if sizePercDiff > BArgs['b_minPercChangeInNumAtomsToReactivate']:
                hasFailureRecord = False
            else:
                hasFailureRecord = True

        if not bigEnough:
            uidsTooSmall.append((uid, size))
            continue
        if hasFailureRecord:
            uidsWithFailRecord.append((uid, nFailRecent))
            continue
        eligible_mask[ii] = 1

    nDQ_toosmall = len(uidsTooSmall)
    nDQ_pastfail = len(uidsWithFailRecord)
    MovePlans['b_curPlan_FailUIDs'] = list() # Track uids that fail to launch
    MovePlans['b_curPlan_nDQ_toosmall'] = nDQ_toosmall
    MovePlans['b_curPlan_nDQ_pastfail'] = nDQ_pastfail

    msg = "%d/%d UIDs too small." + \
        " Required size >= %d (--b_minNumAtomsForTargetComp)"
    msg = msg % (nDQ_toosmall, K, BArgs['b_minNumAtomsForTargetComp'])
    BLogger.pprint(msg, 'debug')
    if nDQ_toosmall > 0:
        BLogger.pprint(
            ' uids  ' + vec2str([u[0] for u in uidsTooSmall]), 'debug')
        BLogger.pprint(
            ' sizes ' + vec2str([u[1] for u in uidsTooSmall]), 'debug')
    
    BLogger.pprint(
        '%d/%d UIDs disqualified for past failures.' % (
            nDQ_pastfail, K),
        'debug')
    if nDQ_pastfail > 0:
        BLogger.pprint(
            ' uids  ' + vec2str([u[0] for u in uidsWithFailRecord]),
            'debug')
        BLogger.pprint(
            ' sizes ' + vec2str([u[1] for u in uidsWithFailRecord]),
            'debug')

    # Finalize list of eligible UIDs
    UIDs = SS.uids[eligible_mask]

    BLogger.pprint('%d/%d UIDs eligible' % (len(UIDs), K), 'debug')
    BLogger.pprint(
        ' uids  ' + vec2str(UIDs), 'debug')
    if len(UIDs) == 0:
        return MovePlans
    # Finalize corresponding scores
    Scores = ScoreVec[eligible_mask]
    BLogger.pprint(
        ' sizes ' + vec2str(countVec[eligible_mask]), 'debug')
    BLogger.pprint(
        ' score ' + vec2str(Scores), 'debug')

    # Figure out how many new states we can target this round.
    totalnewK = BArgs['b_Kfresh'] * Scores.size
    maxnewK = BArgs['Kmax'] - SS.K
    # Prioritize which comps to target
    # Keeping the first nToKeep, as ranked by the Ldata score
    nToKeep = maxnewK // BArgs['b_Kfresh']
    keepIDs = np.argsort(-1 * Scores)[:nToKeep]
    MovePlans['BirthTargetUIDs'] = [UIDs[s] for s in keepIDs]

    BLogger.pprint('%d UIDs chosen for proposals (ranked by score)' % (
        len(keepIDs)))
    BLogger.pprint(
        ' uids  ' + vec2str(MovePlans['BirthTargetUIDs']))
    BLogger.pprint(
        ' sizes ' + vec2str(countVec[eligible_mask][keepIDs]), 'debug')
    BLogger.pprint(
        ' score ' + vec2str(Scores[keepIDs]), 'debug')

    return MovePlans
    

def canBirthHappenAtLap(lapFrac, b_startLap=-1, b_stopLap=-1, **kwargs):
    ''' Make binary yes/no decision if birth move can happen at provided lap.

    Returns
    -------
    answer : boolean
        True only if lapFrac >= b_startLap and lapFrac < stopLap

    Examples
    --------
    >>> canBirthHappenAtLap(0.1, b_startLap=1, b_stopLap=2)
    True
    >>> canBirthHappenAtLap(1.0, b_startLap=1, b_stopLap=2)
    True
    >>> canBirthHappenAtLap(1.1, b_startLap=1, b_stopLap=2)
    False
    >>> canBirthHappenAtLap(2.0, b_startLap=1, b_stopLap=2)
    False
    >>> canBirthHappenAtLap(10.5, b_startLap=1, b_stopLap=2)
    False
    >>> canBirthHappenAtLap(10.5, b_startLap=1, b_stopLap=11)
    False
    >>> canBirthHappenAtLap(10.5, b_startLap=1, b_stopLap=12)
    True
    '''
    if b_startLap < 0:
        return False
    elif b_startLap >= 0 and np.ceil(lapFrac) < b_startLap:
        return False 
    elif b_stopLap >= 0 and np.ceil(lapFrac) >= b_stopLap:
        return False
    else:
        return True
