import numpy as np
import os

import BLogger
from collections import defaultdict
from BCreateOneProposal import makeSummaryForBirthProposal_HTMLWrapper

def makeSummariesForManyBirthProposals(
        Dslice=None,
        curModel=None,
        curLPslice=None,
        curSSwhole=None,
        curSSslice=None,
        LPkwargs=None,
        newUIDs=list(),
        b_targetUIDs=None,
        xSSProposalsByUID=None,
        MovePlans=dict(),
        MoveRecordsByUID=dict(),
        taskoutpath='/tmp/',
        lapFrac=0.0,
        batchPos=0,
        nBatch=0,
        **BArgs):
    '''

    Args
    ----
    BArgs : dict of all kwarg options for birth moves

    Returns
    -------
    xSSProposalsByUID : dict
    MovePlans : dict
        Tracks aggregate performance across all birth proposals.
    MoveRecordsByUID : dict
        each key is a uid. Tracks performance for each uid.
    '''
    if b_targetUIDs is None:
        b_targetUIDs = MovePlans['b_targetUIDs']
    if len(b_targetUIDs) > 0:
        BLogger.pprint(
            'CREATING birth proposals at lap %.2f' % (lapFrac))
    if xSSProposalsByUID is None:
        xSSProposalsByUID = dict()

    failedUIDs = list()
    # Loop thru copy of the target comp UID list
    # So that we can remove elements from it within the loop
    for ii, targetUID in enumerate(b_targetUIDs):

        if targetUID in xSSProposalsByUID:
            raise ValueError("Already have a proposal for this UID")

        Kfresh = BArgs['b_Kfresh']
        newUIDs_ii = newUIDs[(ii * Kfresh):((ii+1) * Kfresh)]
        if len(newUIDs_ii) < 2:
            raise ValueError("Cannot make proposal with less than 2 new UIDs")
        xSSslice, Info = makeSummaryForBirthProposal_HTMLWrapper(
            Dslice, curModel, curLPslice,
            curSSwhole=curSSwhole,
            targetUID=targetUID,
            newUIDs=newUIDs_ii,
            LPkwargs=LPkwargs,
            lapFrac=lapFrac,
            **BArgs)

        if xSSslice is not None:
            # Proposal successful, with at least 2 non-empty clusters.
            # Move on to the evaluation stage!
            xSSProposalsByUID[targetUID] = xSSslice
        else:
            # Failure. Expansion did not create good proposal.
            failedUIDs.append(targetUID)
            MovePlans['b_nTrial'] += 1
            MovePlans['b_nFailedProp'] += 1

            if targetUID not in MoveRecordsByUID:
                MoveRecordsByUID[targetUID] = defaultdict(int)
            Rec = MoveRecordsByUID[targetUID]
            ktarget = curSSwhole.uid2k(targetUID)
            Rec['b_nTrial'] += 1
            Rec['b_nFail'] += 1
            Rec['b_nFailRecent'] += 1
            Rec['b_nSuccessRecent'] = 0
            Rec['b_latestLap'] = lapFrac
            Rec['b_latestCount'] = curSSwhole.getCountVec()[ktarget]
            if curSSslice:
                Rec['b_latestBatchCount'] = curSSslice.getCountVec()[ktarget]
                if curSSslice.hasSelectionTerm('DocUsageCount'):
                    Rec['b_latestBatchNDoc'] = \
                        curSSslice.getSelectionTerm('DocUsageCount')[ktarget]

    for failUID in failedUIDs:
        b_targetUIDs.remove(failUID)
    MovePlans['b_targetUIDs'] = b_targetUIDs
    return xSSProposalsByUID, MovePlans, MoveRecordsByUID
