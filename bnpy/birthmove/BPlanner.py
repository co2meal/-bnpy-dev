import numpy as np
from collections import defaultdict

import BLogger
from bnpy.viz.PrintTopics import vec2str

def selectShortListForBirthAtLapStart(
        hmodel, SS,
        MoveRecordsByUID=dict(),
        MovePlans=dict(),
        lapFrac=0,
        b_minNumAtomsForTargetComp=2,
        **BArgs):
    ''' Select list of comps to possibly target with birth during next lap.

    Shortlist uids are guaranteed to never be involved in a merge/delete.
    They are kept aside especially for a birth move, at least in this lap.
    
    Returns
    -------
    MovePlans : dict with updated fields
    * b_shortlistUIDs : list of ints,
        Each uid in b_shortlistUIDs could be a promising birth target.
        None of these should be touched by deletes or merges in this lap.
    '''
    MovePlans['b_shortlistUIDs'] = list()
    if not canBirthHappenAtLap(lapFrac, **BArgs):
        return MovePlans

    K = hmodel.obsModel.K
    if BArgs['Kmax'] - K <= 0:
        BLogger.pprint(
            "Cannot plan any more births." + \
            " Reached upper limit of %d existing comps (--Kmax)." % (
                BArgs['Kmax'])
            )
        return MovePlans

    if SS is None:
        MovePlans['b_shortlistUIDs'] = np.arange(K).tolist()
        BLogger.pprint(
            "No SS provided. Shortlist contains all %d possible comps." % (K))
        return MovePlans
    assert SS.K == K

    CountVec = SS.getCountVec()
    eligible_mask = np.zeros(K, dtype=np.bool8)
    nTooSmall = 0
    nPastFail = 0
    for k, uid in enumerate(SS.uids):
        if uid not in MoveRecordsByUID:
            MoveRecordsByUID[uid] = defaultdict(int)

        tooSmall = CountVec[k] <= b_minNumAtomsForTargetComp
        hasFailRecord = MoveRecordsByUID[uid]['b_nFailRecent'] > 0
        if (not tooSmall) and (not hasFailRecord):
            eligible_mask[k] = 1
            MovePlans['b_shortlistUIDs'].append(uid)
        elif tooSmall:
            nTooSmall += 1
        elif hasFailRecord:
            nPastFail += 1

    MovePlans['b_nDQ_toosmall'] = nTooSmall
    MovePlans['b_nDQ_pastfail'] = nPastFail
    nShortList = len(MovePlans['b_shortlistUIDs'])
    BLogger.pprint(
        "%d/%d uids selected for short list." % (nShortList, K))
    if nShortList > 0:
        lineUID = vec2str(MovePlans['b_shortlistUIDs'])
        lineSize = vec2str(CountVec[eligible_mask])
        BLogger.pprint([lineUID, lineSize], 
            prefix=['%7s' % 'uids',
                    '%7s' % 'size'],
            )

    return MovePlans


def selectCompsForBirthAtCurrentBatch(
        hmodel, SS,
        SSbatch=None,
        MoveRecordsByUID=dict(),
        MovePlans=dict(),
        lapFrac=0,
        **BArgs):
    ''' Select specific comps to target with birth move at current batch.

    Returns
    -------
    MovePlans : dict with updated fields
    * b_targetUIDs : list of ints,
        Each uid in b_targetUIDs will be tried immediately, at current batch.
    '''
    if BArgs['Kmax'] - SS.K <= 0:
        BLogger.pprint(
            "Cannot plan any more births." + \
            " Reached upper limit of %d existing comps (--Kmax)." % (
                BArgs['Kmax'])
            )
        if 'b_targetUIDs' in MovePlans:
            del MovePlans['b_targetUIDs']
        return MovePlans

    if 'b_targetUIDs' in MovePlans:
        if len(MovePlans['b_targetUIDs']) > 0:
            uidStr = vec2str(MovePlans['b_targetUIDs'])
            BLogger.pprint(
                'AGAIN! Target b_targetUIDs from previous batch\n ' + uidStr)
            return MovePlans

    K = SS.K
    # Get per-comp sizes for aggregate dataset
    SizeVec_all = np.maximum(SS.getCountVec(), 1e-100)
    CountVec_b = np.maximum(SSbatch.getCountVec(), 1e-100)
    SizeVec_b = CountVec_b

    atomstr = 'atoms'
    labelstr = 'nAtom_b'
    # Adjust the counts and units appropriately, if we modeling documents
    if hasattr(SSbatch, 'WordCounts'):
        if SSbatch.hasSelectionTerm('DocUsageCount'):
            SizeVec_b = SSbatch.getSelectionTerm('DocUsageCount')
            atomstr = 'docs'
            labelstr = 'nDoc_b'

    # Compute per-comp score metric
    # ScoreVec[k] is large if k does a bad job modeling its assigned data
    ScoreVec = -1.0 / SizeVec_all * \
        hmodel.obsModel.calc_evidence(None, SS, None, returnVec=1)
    ScoreVec[SizeVec_all < 1] = np.nan

    uidsBusyWithOtherMoves = list()
    uidsTooSmall = list()
    uidsWithFailRecord = list()
    eligible_mask = np.zeros(K, dtype=np.bool8)
    for ii, uid in enumerate(SS.uids):
        if uid not in MoveRecordsByUID:
            MoveRecordsByUID[uid] = defaultdict(int)

        if 'd_targetUIDs' in MovePlans:
            if uid in MovePlans['d_targetUIDs']:
                uidsBusyWithOtherMoves.append(uid)
                continue
        if 'd_absorbingUIDSet' in MovePlans:
            if uid in MovePlans['d_absorbingUIDSet']:
                uidsBusyWithOtherMoves.append(uid)
                continue

        if 'm_targetUIDSet' in MovePlans:
            if uid in MovePlans['m_targetUIDSet']:
                uidsBusyWithOtherMoves.append(uid)
                continue

        # Filter out uids without large presence in current batch
        bigEnough = SizeVec_b[ii] >= BArgs['b_minNumAtomsForTargetComp']
        if not bigEnough:
            uidsTooSmall.append((uid, SizeVec_b[ii]))
            continue

        # Filter out uids we've failed with birth moves before
        size = SizeVec_all[ii]
        oldsize = MoveRecordsByUID[uid]['b_latestCount']
        oldbatchsize = MoveRecordsByUID[uid]['b_latestBatchCount']
        nFailRecent = MoveRecordsByUID[uid]['b_nFailRecent']
        if nFailRecent == 0:
            hasFailureRecord = False
        else:
            sizePercDiff = np.abs(size - oldsize) / \
                (1e-100 + np.abs(oldsize))
            sizeChangedEnoughToReactivate = sizePercDiff > \
                BArgs['b_minPercChangeInNumAtomsToReactivate']
            minBatchSizeToReactivate = oldbatchsize * \
                (1.0 + BArgs['b_minPercChangeInNumAtomsToReactivate'])
            if size > oldsize and sizeChangedEnoughToReactivate:
                hasFailureRecord = False
                msg = "uid %d reactivated by total size!" % (uid) + \
                    " new_size %.1f  old_size %.1f" % (
                        size, oldsize)
                BLogger.pprint(msg, 'debug')
            elif CountVec_b[ii] >= minBatchSizeToReactivate:
                hasFailureRecord = False
                msg = "uid %d reactivated by batch size!" % (uid) + \
                    " new_batchsize %.1f  old_batchsize %.1f" % (
                        CountVec_b[ii], oldbatchsize)
                BLogger.pprint(msg, 'debug')
            else:
                hasFailureRecord = True
        if hasFailureRecord:
            uidsWithFailRecord.append((uid, nFailRecent))
            continue

        eligible_mask[ii] = 1

    nDQ_toobusy = len(uidsBusyWithOtherMoves)
    nDQ_toosmall = len(uidsTooSmall)
    nDQ_pastfail = len(uidsWithFailRecord)

    # TODO: do we need these?    
    # MovePlans['b_curPlan_nDQ_toosmall'] = nDQ_toosmall
    # MovePlans['b_curPlan_nDQ_pastfail'] = nDQ_pastfail
    # MovePlans['b_curPlan_nDQ_toobusy'] = nDQ_toobusy

    msg = "%d/%d UIDs too busy with other moves (merge/delete)." % (
        nDQ_toobusy, K)
    BLogger.pprint(msg, 'debug')

    msg = "%d/%d UIDs too small (too few %s in current batch)." + \
        " Required size >= %d (--b_minNumAtomsForTargetComp)"
    msg = msg % (nDQ_toosmall, K, atomstr,
        BArgs['b_minNumAtomsForTargetComp'])
    BLogger.pprint(msg, 'debug')
    if nDQ_toosmall > 0:
        lineUID = vec2str([u[0] for u in uidsTooSmall])
        lineSize = vec2str([u[1] for u in uidsTooSmall])
        BLogger.pprint([lineUID, lineSize], 
            prefix=['%7s' % 'uids',
                    '%7s' % labelstr],
            )

    # Notify about past failure disqualifications to the log
    BLogger.pprint(
        '%d/%d UIDs disqualified for past failures.' % (
            nDQ_pastfail, K),
        'debug')
    if nDQ_pastfail > 0:
        lineUID = vec2str([u[0] for u in uidsWithFailRecord])
        lineFail = vec2str([u[1] for u in uidsWithFailRecord])
        BLogger.pprint([lineUID, lineFail],
            prefix=['%7s' % 'uids',
                    '%7s' % 'nFail'],
            )
    # Finalize list of eligible UIDs
    UIDs = SS.uids[eligible_mask]
    BLogger.pprint('%d/%d UIDs eligible' % (len(UIDs), K), 'debug')
    # EXIT if nothing eligible.
    if len(UIDs) == 0:
        return MovePlans

    # Mark all uids that are eligible!
    for uid in UIDs:
        MoveRecordsByUID[uid]['b_latestEligibleLap'] = lapFrac

    lineUID = vec2str(UIDs)
    lineSize = vec2str(SizeVec_all[eligible_mask])
    lineBatchSize = vec2str(SizeVec_b[eligible_mask])
    lineScore = vec2str(ScoreVec[eligible_mask])
    BLogger.pprint([lineUID, lineSize, lineBatchSize, lineScore],
            prefix=[
                '%7s' % 'uids',
                '%7s' % 'size',
                '%7s' % labelstr,
                '%7s' % 'score',
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
    MovePlans['b_targetUIDs'] = [UIDs[s] for s in keepIDs]

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
    lineUID = vec2str(MovePlans['b_targetUIDs'])
    lineSize = vec2str(SizeVec_all[eligible_mask][keepIDs])
    lineBatchSize = vec2str(SizeVec_b[eligible_mask][keepIDs])
    lineScore = vec2str(ScoreVec[eligible_mask][keepIDs])
    BLogger.pprint([lineUID, lineSize, lineBatchSize, lineScore],
            prefix=[
                '%7s' % 'uids',
                '%7s' % 'size',
                '%7s' % labelstr,
                '%7s' % 'score',
                ],
            )

    return MovePlans





"""
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
"""

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