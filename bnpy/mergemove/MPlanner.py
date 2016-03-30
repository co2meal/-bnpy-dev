'''
MPlanner.py

Functions for deciding which merge pairs to track.
'''
import numpy as np
import sys
import os
import itertools
import MLogger
from collections import defaultdict
from bnpy.viz.PrintTopics import vec2str, count2str

ELBO_GAP_ACCEPT_TOL = 1e-6

def selectCandidateMergePairs(hmodel, SS,
        m_maxNumPairsContainingComp=3,
        MovePlans=dict(),
        MoveRecordsByUID=dict(),
        lapFrac=None,
        m_minPercChangeInNumAtomsToReactivate=0.01,
        m_nLapToReactivate=10,
        **kwargs):
    ''' Select candidate pairs to consider for merge move.
    
    Returns
    -------
    Info : dict, with fields
        * m_UIDPairs : list of tuples, each defining a pair of uids
        * m_targetUIDSet : set of all uids involved in a proposed merge pair
    '''
    MLogger.pprint(
        "PLANNING merges at lap %.2f. K=%d" % (lapFrac, SS.K),
        'debug')

    # Mark any targetUIDs used in births as off-limits for merges
    uidUsageCount = defaultdict(int)
    if 'b_shortlistUIDs' in MovePlans:
        for uid in MovePlans['b_shortlistUIDs']:
            uidUsageCount[uid] = 10 * m_maxNumPairsContainingComp
    nDisqualified = len(uidUsageCount.keys())
    MLogger.pprint(
        "   %d/%d UIDs ineligible because on shortlist for births. " % (
            nDisqualified, SS.K),
        'debug')
    if nDisqualified > 0:
        MLogger.pprint(
            "   Ineligible UIDs:" + \
                vec2str(uidUsageCount.keys()),
            'debug')

    uid2k = dict()
    uid2count = dict()
    for uid in SS.uids:
        uid2k[uid] = SS.uid2k(uid)
        uid2count[uid] = SS.getCountForUID(uid)

    EligibleUIDPairs = list()
    EligibleAIDPairs = list()
    nPairTotal = 0
    nPairDQ = 0
    nPairBusy = 0
    for kA, uidA in enumerate(SS.uids):
        for b, uidB in enumerate(SS.uids[kA+1:]):
            kB = kA + b + 1
            assert kA < kB
            nPairTotal += 1
            if uidUsageCount[uidA] > 0 or uidUsageCount[uidB] > 0:
                nPairBusy += 1
                continue
            if uidA < uidB:
                uidTuple = (uidA, uidB)
            else:
                uidTuple = (uidB, uidA)
            aidTuple = (kA, kB)

            if uidTuple not in MoveRecordsByUID:
                EligibleUIDPairs.append(uidTuple)
                EligibleAIDPairs.append(aidTuple)
            else:
                pairRecord = MoveRecordsByUID[uidTuple]
                assert pairRecord['m_nFailRecent'] >= 1
                latestMinCount = pairRecord['m_latestMinCount']
                newMinCount = np.minimum(uid2count[uidA], uid2count[uidB])
                percDiff = np.abs(latestMinCount - newMinCount) / \
                    latestMinCount
                if (lapFrac - pairRecord['m_latestLap']) >= m_nLapToReactivate:
                    EligibleUIDPairs.append(uidTuple)
                    EligibleAIDPairs.append(aidTuple)
                    del MoveRecordsByUID[uidTuple]
                elif percDiff >= m_minPercChangeInNumAtomsToReactivate:
                    EligibleUIDPairs.append(uidTuple)
                    EligibleAIDPairs.append(aidTuple)
                    del MoveRecordsByUID[uidTuple]
                else:
                    nPairDQ += 1
    MLogger.pprint(
        "   %d/%d pairs eligible. %d disqualified by past failures." % (
            len(EligibleAIDPairs), nPairTotal, nPairDQ),
        'debug')
    # Compute Ldata gain for each possible pair of comps
    oGainMat = hmodel.obsModel.calcHardMergeGap_SpecificPairs(
        SS, EligibleAIDPairs)
    if hmodel.getAllocModelName().count('Mixture'):
        GainMat = oGainMat + hmodel.allocModel.calcHardMergeGap_SpecificPairs(
            SS, EligibleAIDPairs)
    elif hmodel.getAllocModelName().count('HMM'):
        GainMat = oGainMat + hmodel.allocModel.calcHardMergeGap_SpecificPairs(
            SS, EligibleAIDPairs)
    else:
        GainMat = oGainMat

    # Find pairs with positive gains
    posLocs = np.flatnonzero(GainMat > - ELBO_GAP_ACCEPT_TOL)
    sortIDs = np.argsort(-1 * GainMat[posLocs])
    posLocs = posLocs[sortIDs]
    nKeep = 0
    mUIDPairs = list()
    mAIDPairs = list()
    mGainVals = list()
    for loc in posLocs:
        uidA, uidB = EligibleUIDPairs[loc]
        kA, kB = EligibleAIDPairs[loc]
        if uidUsageCount[uidA] >= m_maxNumPairsContainingComp or \
                uidUsageCount[uidB] >= m_maxNumPairsContainingComp:
            continue
        uidUsageCount[uidA] += 1
        uidUsageCount[uidB] += 1

        mAIDPairs.append((kA, kB))
        mUIDPairs.append((uidA, uidB))
        mGainVals.append(GainMat[loc])
        if nKeep == 0:
            MLogger.pprint("Chosen pairs:", 'debug')
        MLogger.pprint(
            "%4d, %4d : gain %.3e, size %s %s" % (
                uidA, uidB, 
                GainMat[loc],
                count2str(uid2count[uidA]),
                count2str(uid2count[uidB]),
                ),
            'debug')
    Info = dict()
    Info['m_UIDPairs'] = mUIDPairs
    Info['m_GainVals'] = mGainVals 
    Info['mPairIDs'] = mAIDPairs
    targetUIDs = set()
    for uidA, uidB in mUIDPairs:
        targetUIDs.add(uidA)
        targetUIDs.add(uidB)
        if 'b_shortlistUIDs' in MovePlans:
            for uid in MovePlans['b_shortlistUIDs']:
                assert uid != uidA
                assert uid != uidB
    Info['m_targetUIDSet'] = targetUIDs
    return Info




    '''
    from IPython import embed; embed()
    # Mask out the upper triangle of entries here (other entries are zeros)
    triuIDs = np.triu_indices(SS.K, 1)
    # Identify only the positive gains
    posLocs = np.flatnonzero(GainMat[triuIDs] > - ELBO_GAP_ACCEPT_TOL)
    # Rank the positive pairs from largest to smallest gain in ELBO
    sortIDs = np.argsort(-1 * GainMat[triuIDs][posLocs])
    posLocs = posLocs[sortIDs]
    MLogger.pprint(
        "   %d/%d total pairs have positive gains." % (
            posLocs.size, triuIDs[0].size),
        'debug')

    if len(posLocs) > 0:
        MLogger.pprint(
            "   Filtering pairs so no UID is ineligible\n" + 
            "       or appears in more than %d pairs" % (
                m_maxNumPairsContainingComp) +
            ", set by --m_maxNumPairsContainingComp",
            'debug')

    # Make final list of pairs to track
    mUIDPairs = list()
    mIDPairs = list()
    mGainVals = list()
    mSizeVals = list()
    SizeVec = SS.getCountVec()
    for loc in posLocs:
        kA = triuIDs[0][loc]
        kB = triuIDs[1][loc]
        uidA = SS.uids[triuIDs[0][loc]]
        uidB = SS.uids[triuIDs[1][loc]]
        ctA = uidUsageCount[uidA]
        ctB = uidUsageCount[uidB]
        if ctA >= m_maxNumPairsContainingComp or \
                ctB >= m_maxNumPairsContainingComp:
            continue
        uidUsageCount[uidA] += 1
        uidUsageCount[uidB] += 1
        mUIDPairs.append((uidA, uidB))
        mIDPairs.append((kA, kB))
        mGainVals.append(GainMat[kA, kB])
        mSizeVals.append((SizeVec[kA], SizeVec[kB]))
    # Final status update
    MLogger.pprint(
        "   %d pairs selected for merge tracking." % (
            len(mUIDPairs)),
        'debug')
    if len(mUIDPairs) > 0:
        MLogger.pprint("Chosen pairs:", 'debug')
        for ii, (uidA, uidB) in enumerate(mUIDPairs):
            MLogger.pprint(
                "%4d, %4d : gain %.3e, size %s %s" % (
                    uidA, uidB, 
                    mGainVals[ii],
                    count2str(mSizeVals[ii][0]),
                    count2str(mSizeVals[ii][1])
                    ),
                'debug')
    Info = dict()
    Info['m_uidUsageCount'] = uidUsageCount
    Info['m_GainMat'] = GainMat
    Info['m_GainVals'] = mGainVals
    Info['m_UIDPairs'] = mUIDPairs
    Info['mPairIDs'] = mIDPairs
    targetUIDs = set()
    for uidA, uidB in mUIDPairs:
        targetUIDs.add(uidA)
        targetUIDs.add(uidB)
        if 'b_shortlistUIDs' in MovePlans:
            for uid in MovePlans['b_shortlistUIDs']:
                assert uid != uidA
                assert uid != uidB
    Info['m_targetUIDSet'] = targetUIDs
    return Info
    '''