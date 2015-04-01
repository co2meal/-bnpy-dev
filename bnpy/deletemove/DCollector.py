"""
Functions for collecting a target dataset for a delete move.

- addDataFromBatchToPlan
- getDataSubsetRelevantToPlan
"""

import numpy as np
import DeleteLogger


def addDataFromBatchToPlan(Plan, hmodel, Dchunk, LPchunk,
                           uIDs=None,
                           batchID=0,
                           lapFrac=None,
                           isFirstBatch=0,
                           dtargetMaxSize=1000,
                           dtargetMinCount=0.01,
                           **kwargs):
    """ Add relevant data from provided chunk to the planned target set.

        Returns
        -------
        Plan : dict, same reference as provided, updated in-place.

        Updates
        -------
        Plan will have new fields if successfully augmented the target set
        * DTargetData
        * batchIDs

        If the target set goes over the budget space of dtargetMaxSize,
        then Plan will be wiped out to an empty dict.
    """
    assert uIDs is not None
    assert len(uIDs) == hmodel.allocModel.K
    assert len(uIDs) == hmodel.obsModel.K

    if isFirstBatch:
        msg = '<<<<<<<<<<<<<<<<<<<< addDataFromBatchToPlan @ lap %6.2f' \
              % (lapFrac)
        DeleteLogger.log(msg)

    relData, relIDs = getDataSubsetRelevantToPlan(
        Dchunk, LPchunk, Plan,
        dtargetMinCount=dtargetMinCount)
    relSize = getSize(relData)
    if relSize < 1:
        msg = ' %6.3f | batch %3d | batch trgtSize 0 | agg trgtSize 0' \
              % (lapFrac, batchID)
        DeleteLogger.log(msg)
        return Plan

    # ----    Add all these docs to the Plan
    batchIDs = [batchID for n in xrange(relSize)]
    if hasValidKey(Plan, 'DTargetData'):
        Plan['DTargetData'].add_data(relData)
        Plan['batchIDs'].extend(batchIDs)
    else:
        Plan['DTargetData'] = relData
        Plan['batchIDs'] = batchIDs
        Plan['dataUnitIDs'] = relIDs

    curTargetSize = getSize(Plan['DTargetData'])
    if curTargetSize > dtargetMaxSize:
        for key in Plan.keys():
            del Plan[key]
        msg = ' %6.3f | batch %3d | targetSize %d EXCEEDED BUDGET of %d' \
            % (lapFrac, batchID, curTargetSize, dtargetMaxSize)
        DeleteLogger.log(msg)
        DeleteLogger.log("ABANDONED.")
        return Plan

    if lapFrac is not None:
        msg = ' %6.3f | batch %3d | batch trgtSize %5d | agg trgtSize %5d' \
            % (lapFrac, batchID, relSize, curTargetSize)
        DeleteLogger.log(msg)

    # ----    Track stats specific to chosen subset
    targetLPchunk = hmodel.allocModel.selectSubsetLP(Dchunk, LPchunk, relIDs)
    targetSSchunk = hmodel.get_global_suff_stats(relData, targetLPchunk,
                                                 doPrecompEntropy=1)
    targetSSchunk.uIDs = uIDs.copy()

    # ----   targetSS tracks aggregate stats across batches
    if not hasValidKey(Plan, 'targetSS'):
        Plan['targetSS'] = targetSSchunk.copy()
    else:
        Plan['targetSS'] += targetSSchunk

    # ----    targetSSByBatch tracks batch-specific stats
    if not hasValidKey(Plan, 'targetSSByBatch'):
        Plan['targetSSByBatch'] = dict()
    Plan['targetSSByBatch'][batchID] = targetSSchunk

    return Plan


def getDataSubsetRelevantToPlan(Dchunk, LPchunk, Plan,
                                dtargetMinCount=0.01):
    """ Get subset of provided DataObj containing units relevant to the Plan.

        Returns
        --------
        relData : None or bnpy.data.DataObj
        relIDs : list of integer ids of relevant units of provided Dchunk
    """
    if not hasValidKey(Plan, 'candidateIDs'):
        return None, []

    for dd, delCompID in enumerate(Plan['candidateIDs']):
        if 'DocTopicCount' in LPchunk:
            DocTopicCount = LPchunk['DocTopicCount']
            curkeepmask = DocTopicCount[:, delCompID] >= dtargetMinCount
        elif 'respPair' in LPchunk or 'TransStateCount' in LPchunk:
            curkeepmask = np.zeros(Dchunk.nDoc, dtype=np.int32)
            for n in xrange(Dchunk.nDoc):
                start = Dchunk.doc_range[n]
                stop = Dchunk.doc_range[n + 1]
                Usage_n = np.sum(LPchunk['resp'][start:stop, delCompID])
                curkeepmask[n] = Usage_n >= dtargetMinCount
        else:
            curkeepmask = LPchunk['resp'][:, delCompID] >= dtargetMinCount

        # Aggregate current mask with masks for all previous delCompID values
        if dd > 0:
            keepmask = np.logical_or(keepmask, curkeepmask)
        else:
            keepmask = curkeepmask

    relUnitIDs = np.flatnonzero(keepmask)
    if len(relUnitIDs) < 1:
        return None, relUnitIDs
    else:
        relData = Dchunk.select_subset_by_mask(relUnitIDs,
                                               doTrackFullSize=False)
        return relData, relUnitIDs


def hasValidKey(dict, key):
    """ Return True if key is in dict and not None, False otherwise.
    """
    return key in dict and dict[key] is not None


def getSize(Data):
    """ Return the integer size of the provided dataset.
    """
    if Data is None:
        return 0
    elif hasattr(Data, 'nDoc'):
        return Data.nDoc
    else:
        return Data.nObs
