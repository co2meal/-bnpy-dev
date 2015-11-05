import numpy as np
import os
from collections import defaultdict

from BCreateOneProposal import makeSummaryForBirthProposal_HTMLWrapper

def makeSummariesForManyBirthProposals(
        Dslice, curModel, curLPslice,
        curSSwhole=None,
        b_targetUIDs=list(),
        newUIDs=list(),
        xSSProposalsByUID = dict(),
        MovePlans=dict()
        MoveRecordsByUID=dict()
        b_debugWriteHTML=0,
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
    if len(b_targetUIDs) > 0:
        BLogger.pprint(
            'CREATING %d birth proposals at lap %.2f' % (lapFrac))

    # Loop thru copy of the target comp UID list
    # So that we can remove elements from it within the loop
    for ii, targetUID in enumerate(MovePlans['b_targetUIDs']):

        if targetUID in xSSProposalsByUID:
            raise ValueError("Already have a proposal for this UID")

        xSSslice, Info = makeSummaryForBirthProposal_HTMLWrapper(
            Dslice, curModel, curLPslice,
            curSSwhole=curSSwhole,
            targetUID=targetUID,
            newUIDs=newUIDs,
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
            MovePlans['b_nFailedProp'] += 1
            MovePlans['b_nTrial'] += 1

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
            Rec['b_latestBatchCount'] = SSbatch.getCountVec()[ktarget]
            if SSbatch.hasSelectionTerm('DocUsageCount'):
                Rec['b_latestBatchNDoc'] = \
                    SSbatch.getSelectionTerm('DocUsageCount')[ktarget]

    return xSSProposalsByUID, MovePlans, MoveRecordsByUID
