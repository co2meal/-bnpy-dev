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
    '''
    DLogger.pprint("DPLANNER: selecting target at lap %.2f" % (lapFrac))

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
        DLogger.pprint("No eligible UIDs. All occupied by merge or birth.")
        return dict()

    # Compute score for each eligible state
    countVec = np.maximum(SS.getCountVec(), 1e-100)
    ScoreVec = -1.0 / countVec * \
        hmodel.obsModel.calc_evidence(None, SS, None, returnVec=1)
    CountMap = dict()

    ScoreByEligibleUID = dict()
    canDoBirthList = list()
    hasFailRecordList = list()
    for uid in eligibleUIDs:
        k = SS.uid2k(uid)
        size = countVec[k]
        if uid not in MoveRecordsByUID:
            MoveRecordsByUID[uid] = defaultdict(int)
        # First criteria: avoid comps we've failed deleting in the past
        hasFailureRecord_Delete = MoveRecordsByUID[uid]['d_nFailRecent'] > 0
        if hasFailureRecord_Delete:
            hasFailRecordList.append(uid)
            continue
        # Second: do not delete comps we have not yet tried with births
        if not BPlanner.canBirthHappenAtLap(lapFrac, **DArgs):
            eligibleForBirth = False
        else:
            if 'b_minNumAtomsForTargetComp' in DArgs:
                tooSmall = size <= DArgs['b_minNumAtomsForTargetComp']
            else:
                tooSmall = True
            hasFailureRecord_Birth = MoveRecordsByUID[uid]['b_nFailRecent'] > 0
            eligibleForBirth = (not tooSmall) and (not hasFailureRecord_Birth)
        if eligibleForBirth:
            canDoBirthList.append(uid)
            continue
        # If we make it here, the uid is eligible
        ScoreByEligibleUID[uid] = ScoreVec[k]

    # Log which uids were marked has high potential births
    DLogger.pprint('UIDs disqualified for potential future birth:', 'debug')
    DLogger.pprint(
        '  ' + vec2str([u for u in canDoBirthList]), 'debug')
    # Log which uids were marked has having a record.
    DLogger.pprint('UIDs disqualified for past failures:', 'debug')
    DLogger.pprint(
        '  ' + vec2str([u for u in hasFailRecordList]), 'debug')
    # Log all remaining eligible uids
    UIDs = [x for x in ScoreByEligibleUID.keys()]
    DLogger.pprint('All UIDs eligible for deletion:', 'debug')
    DLogger.pprint(
        ' uid   ' + vec2str([u for u in UIDs]), 'debug')
    if len(UIDs) == 0:
        return dict()
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
    # Also track all comps eligible to receive its transfer mass
    eligibleUIDs.discard(targetUID)
    MovePlans['d_absorbingUIDSet'] = eligibleUIDs

    DLogger.pprint('targetUID: ' + count2str(targetUID))
    DLogger.pprint('absorbingUIDs: ' + vec2str(eligibleUIDs))

    return MovePlans
