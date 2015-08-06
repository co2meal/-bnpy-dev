'''
MPlanner.py

Functions for deciding which merge pairs to track.
'''
import numpy as np
import sys
import os

from collections import defaultdict

ELBO_GAP_ACCEPT_TOL = 1e-6

def selectCandidateMergePairs(hmodel, SS,
        m_maxNumPairsContainingComp=3,
        **kwargs):
    ''' Select candidate pairs to consider for merge move.
    
    Returns
    -------
    Info : dict, with fields
        * m_UIDPairs : list of tuples, each defining a pair of uids
    '''
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
    # Make final list of pairs to track
    uidUsageCount = defaultdict(int)
    mUIDPairs = list()
    mIDPairs = list()
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
    Info = dict()
    Info['m_uidUsageCount'] = uidUsageCount
    Info['m_GainMat'] = GainMat
    Info['m_UIDPairs'] = mUIDPairs
    Info['mPairIDs'] = mIDPairs
    return Info
