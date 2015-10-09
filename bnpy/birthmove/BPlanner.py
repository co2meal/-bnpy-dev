import numpy as np
from collections import defaultdict

import BLogger
from bnpy.viz.PrintTopics import vec2str

def selectTargetCompsForBirth(
        hmodel, SS,
        curSSbatch=None,
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

    if BArgs['Kmax'] - SS.K <= 0:
        BLogger.pprint(
            "Cannot plan any more births." + \
            " Reached upper limit of %d existing comps (--Kmax)" % (
                BArgs['Kmax'])
            )
        return MovePlans

    countVec = SS.getCountVec()
    K = countVec.size
    countVec = np.maximum(countVec, 1e-100)
    ScoreVec = -1.0 / countVec * \
        hmodel.obsModel.calc_evidence(None, SS, None, returnVec=1)
    ScoreVec[countVec < 1] = np.nan

    SizeVec_b = curSSbatch.getCountVec()
    atomstr = 'atoms'
    if hasattr(curSSbatch, 'WordCounts'):
        if curSSbatch.hasSelectionTerm('DocUsageCount'):
            SizeVec_b = curSSbatch.getSelectionTerm('DocUsageCount')
            atomstr = 'docs'

    uidsTooSmall = list()
    uidsWithFailRecord = list()
    eligible_mask = np.zeros(K, dtype=np.bool8)
    for ii, uid in enumerate(SS.uids):
        if uid not in MoveRecordsByUID:
            MoveRecordsByUID[uid] = defaultdict(int)

        bigEnough = SizeVec_b[ii] >= BArgs['b_minNumAtomsForTargetComp']
        if not bigEnough:
            uidsTooSmall.append((uid, SizeVec_b[ii]))
            continue

        size = countVec[ii]
        oldsize = MoveRecordsByUID[uid]['b_latestCount']
        nFailRecent = MoveRecordsByUID[uid]['b_nFailRecent']
        if oldsize == 0 or nFailRecent == 0:
            hasFailureRecord = False
        else:
            sizePercDiff = np.abs(size - oldsize)/np.abs(oldsize)
            if sizePercDiff > BArgs['b_minPercChangeInNumAtomsToReactivate']:
                hasFailureRecord = False
            else:
                hasFailureRecord = True

        if hasFailureRecord:
            uidsWithFailRecord.append((uid, size, nFailRecent))
            continue
        eligible_mask[ii] = 1

    nDQ_toosmall = len(uidsTooSmall)
    nDQ_pastfail = len(uidsWithFailRecord)
    MovePlans['b_curPlan_FailUIDs'] = list() # Track uids that fail to launch
    MovePlans['b_curPlan_nDQ_toosmall'] = nDQ_toosmall
    MovePlans['b_curPlan_nDQ_pastfail'] = nDQ_pastfail

    msg = "%d/%d UIDs too small (too few %s)." + \
        " Required size >= %d (--b_minNumAtomsForTargetComp)"
    msg = msg % (nDQ_toosmall, K, atomstr,
        BArgs['b_minNumAtomsForTargetComp'])
    BLogger.pprint(msg, 'debug')
    if nDQ_toosmall > 0:
        lineUID = vec2str([u[0] for u in uidsTooSmall])
        lineSize = vec2str([u[1] for u in uidsTooSmall])
        BLogger.pprint([lineUID, lineSize], 
            prefix=['%6s' % 'uids',
                '%6s' % atomstr],
            )
    
    BLogger.pprint(
        '%d/%d UIDs disqualified for past failures.' % (
            nDQ_pastfail, K),
        'debug')
    if nDQ_pastfail > 0:
        lineUID = vec2str([u[0] for u in uidsWithFailRecord])
        lineSize = vec2str([u[1] for u in uidsWithFailRecord])
        BLogger.pprint([lineUID, lineSize],
            prefix=['%6s' % 'uids',
                '%6s' % 'size'],
            )
        
    # Finalize list of eligible UIDs
    UIDs = SS.uids[eligible_mask]
    BLogger.pprint('%d/%d UIDs eligible' % (len(UIDs), K), 'debug')
    # EXIT if nothing eligible.
    if len(UIDs) == 0:
        return MovePlans

    lineUID = vec2str(UIDs)
    lineSize = vec2str(countVec[eligible_mask])
    lineBatchSize = vec2str(SizeVec_b[eligible_mask])
    lineScore = vec2str(ScoreVec[eligible_mask])
    BLogger.pprint([lineUID, lineSize, lineBatchSize, lineScore],
            prefix=['%6s' % 'uids',
                '%6s' % 'size',
                '%6s' % atomstr,
                '%6s' % 'score',
                ],
            )

    # Figure out how many new states we can target this round.
    # Prioritize the top comps as ranked by the Ldata score
    # until we max out the budget of Kmax total comps.
    maxnewK = BArgs['Kmax'] - SS.K
    totalnewK_perEligibleComp = np.minimum(
        np.ceil(SizeVec_b[eligible_mask]), BArgs['b_Kfresh'])
    sortorder = np.argsort(-1 * ScoreVec[eligible_mask])
    sortedCumulNewK = np.cumsum(totalnewK_perEligibleComp[sortorder])
    nToKeep = np.searchsorted(sortedCumulNewK, maxnewK + 0.0042)
    if nToKeep > 0:
        keepIDs = sortorder[:nToKeep]
        newK = sortedCumulNewK[nToKeep-1]
    else:
        keepIDs = sortorder[:1]    
        newK = maxnewK
    MovePlans['BirthTargetUIDs'] = [UIDs[s] for s in keepIDs]

    if nToKeep < len(UIDs):
        BLogger.pprint(
            'Selected %d/%d eligible UIDs to track.' % (nToKeep, len(UIDs)) + \
            '\n Could create up to %d new clusters, %d total clusters.' % (
                newK, newK + SS.K) + \
            '\n Total budget allows at most %d clusters (--Kmax).' % (
                BArgs['Kmax']),
            )
    BLogger.pprint('%d/%d UIDs chosen for proposals (ranked by score)' % (
        len(keepIDs), len(UIDs)))
    BLogger.pprint(
        ' uids  ' + vec2str(MovePlans['BirthTargetUIDs']))
    BLogger.pprint(
        ' sizes ' + vec2str(countVec[eligible_mask][keepIDs]), 'debug')
    BLogger.pprint(
        ' score ' + vec2str(ScoreVec[eligible_mask][keepIDs]), 'debug')
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
