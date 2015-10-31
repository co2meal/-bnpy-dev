import numpy as np
from collections import defaultdict

import DLogger
import bnpy.birthmove.BPlanner as BPlanner
from bnpy.viz.PrintTopics import count2str, vec2str

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
    * d_absorbingUIDSet : set of ints, all uids that can absorb target mass
    OR
    * failMsg : string explaining why building list of eligible UIDs failed
    '''
    DLogger.pprint("PLANNING delete at lap %.2f" % (lapFrac))
    K = SS.K
    
    # Determine uids eligible for either
    # a) being deleted, or
    # b) "absorbing" deleted comps.
    eligibleUIDs = set(SS.uids)

    if len(eligibleUIDs) < 3:
        DLogger.pprint(
            "Delete proposal requires at least 3 eligible UIDs.\n" + \
            "   Need 1 uid to target, and at least 2 to absorb." + \
            "   Only have %d total uids in the model." % (len(eligibleUIDs)))
        failMsg = "Ineligible. Did not find >= 3 eligible UIDs to absorb."
        return dict(failMsg=failMsg)

    uidsBusyWithOtherMoves = set()
    if 'm_UIDPairs' in MovePlans:
        for (uidA, uidB) in MovePlans['m_UIDPairs']:
            eligibleUIDs.discard(uidA)
            eligibleUIDs.discard(uidB)
            uidsBusyWithOtherMoves.add(uidA)
            uidsBusyWithOtherMoves.add(uidB)
    if 'b_shortlistUIDs' in MovePlans:
        for uid in MovePlans['b_shortlistUIDs']:
            eligibleUIDs.discard(uid)
            uidsBusyWithOtherMoves.add(uid)

    if len(eligibleUIDs) < 3:
        DLogger.pprint("Delete requires at least 3 UIDs" + \
            " not occupied by merge or birth.\n" + \
            "   Need 1 uid to target, and at least 2 to absorb.\n" + \
            "   Only have %d total uids eligible." % (len(eligibleUIDs)))
        failMsg = "Ineligible. Too many uids occupied by merge or shortlisted for birth."
        return dict(failMsg=failMsg)

    # Compute score for each eligible state
    countVec = np.maximum(SS.getCountVec(), 1e-100)
    ScoreVec = -1.0 / countVec * \
        hmodel.obsModel.calc_evidence(None, SS, None, returnVec=1)
    CountMap = dict()

    ScoreByEligibleUID = dict()
    hasFailRecordList = list()
    nFailRecord = 0
    nReactivated = 0
    for uid in eligibleUIDs:
        k = SS.uid2k(uid)
        size = countVec[k]
        if uid not in MoveRecordsByUID:
            MoveRecordsByUID[uid] = defaultdict(int)
        # Criteria: avoid comps we've failed deleting in the past
        # unless they have changed by a reasonable amount.
        nFailRecent_Delete = MoveRecordsByUID[uid]['d_nFailRecent'] > 0
        oldsize = MoveRecordsByUID[uid]['d_latestCount']
        if oldsize > 0 and nFailRecent_Delete > 0:
            nFailRecord += 1
            sizePercDiff = np.abs(size - oldsize)/(1e-100 + np.abs(oldsize))
            if sizePercDiff <= DArgs['b_minPercChangeInNumAtomsToReactivate']:
                hasFailRecordList.append(uid)
                continue
            else:
                nReactivated += 1

        # If we make it here, the uid is eligible
        ScoreByEligibleUID[uid] = ScoreVec[k]

    # Log which uids were marked has high potential births
    msg = "%d/%d UIDs busy with other moves (birth/merge)" % (
       len(uidsBusyWithOtherMoves), K)
    DLogger.pprint(msg)
    if len(uidsBusyWithOtherMoves) > 0:
        DLogger.pprint(
            '  ' + vec2str(uidsBusyWithOtherMoves), 'debug')

    # Log which uids were marked has having a record.
    msg = '%d/%d UIDs un-deleteable for past failures. %d reactivated.' % (
        len(hasFailRecordList), K, nReactivated)
    DLogger.pprint(msg)
    if len(hasFailRecordList) > 0:
        DLogger.pprint(
            '  ' + vec2str(hasFailRecordList), 'debug')
    # Log all remaining eligible uids
    UIDs = [x for x in ScoreByEligibleUID.keys()]
    msg = '%d/%d UIDs eligible for targeted delete proposal' % (
        len(UIDs), K)
    DLogger.pprint(msg)
    if len(UIDs) == 0:
        failMsg = "Nope. 0 UIDs eligible as delete target." + \
            " %d too busy with other moves. %d have past failures." % (
                len(uidsBusyWithOtherMoves), len(hasFailRecordList))
        return dict(failMsg=failMsg)

    DLogger.pprint(
        ' uid   ' + vec2str([u for u in UIDs]), 'debug')
    # Log eligible counts and scores
    DLogger.pprint(
        ' count ' + vec2str([countVec[SS.uid2k(u)] for u in UIDs]), 'debug')
    DLogger.pprint(
        ' score ' + vec2str([ScoreVec[SS.uid2k(u)] for u in UIDs]), 'debug')
    # Select the single state to target
    # by taking the one with highest score
    Scores = np.asarray([x for x in ScoreByEligibleUID.values()])
    targetUID = UIDs[np.argmax(Scores)]
    MovePlans['d_targetUIDs'] = [targetUID]
    # Also determine all comps eligible to receive its transfer mass
    eligibleUIDs.discard(targetUID)
    MovePlans['d_absorbingUIDSet'] = eligibleUIDs

    DLogger.pprint('Selecting one single state to target.')
    DLogger.pprint('targetUID: ' + count2str(targetUID))
    DLogger.pprint('absorbingUIDs: ' + vec2str(eligibleUIDs))

    return MovePlans
