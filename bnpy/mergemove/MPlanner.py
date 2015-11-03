'''
MPlanner.py

Functions for deciding which merge pairs to track.
'''
import numpy as np
import sys
import os

import MLogger
from collections import defaultdict
from bnpy.viz.PrintTopics import vec2str, count2str

ELBO_GAP_ACCEPT_TOL = 1e-6

def selectCandidateMergePairs(hmodel, SS,
        m_maxNumPairsContainingComp=3,
        MovePlans=dict(),
        MoveRecordsByUID=dict(),
        lapFrac=None,
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

    # Compute Ldata gain for each possible pair of comps
    oGainMat = hmodel.obsModel.calcHardMergeGap_AllPairs(SS)
    if hmodel.getAllocModelName().count('Mixture'):
        GainMat = oGainMat + hmodel.allocModel.calcHardMergeGap_AllPairs(SS)
    elif hmodel.getAllocModelName().count('HMM'):
        GainMat = oGainMat + hmodel.allocModel.calcHardMergeGap_AllPairs(SS)
    else:
        GainMat = oGainMat
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
