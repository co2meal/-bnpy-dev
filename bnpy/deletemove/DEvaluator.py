"""
Functions that evaluate delete proposals and decide to accept/reject.

- runDeleteMove
"""
import numpy as np
import DeleteLogger
from DCollector import hasValidKey


def runDeleteMoveAndUpdateMemory(curModel, curSS, Plan,
                                 nRefineIters=2,
                                 LPkwargs=None,
                                 SSmemory=None,
                                 Kmax=np.inf,
                                 lapFrac=None,
                                 **kwargs):
    """ Propose model with fewer comps and accept if ELBO improves.

    Will update the memoized suff stats for each batch (SSmemory)
    in place to reflect any accepted deletions.

    Returns
    --------
    bestModel : HModel, with K' states
    bestSS : SuffStatBag with K' states
    SSmemory : dict of suff stats, one per batch with K' states
    Plan : dict, with updated fields
    * didAccept
    * acceptedUIDs
    * acceptedIDs

    Post Condition
    --------
    SSmemory has valid stats for each batch under proposed model
    with K' states. Summing over all entries of SSmemory
    will be exactly equal to the whole-dataset stats bestSS.

    bestSS and each entry of SSmemory have NO ELBO or Merge terms.
    """
    if SSmemory is None:
        SSmemory = dict()
    if LPkwargs is None:
        LPkwargs = dict()
    if curSS.K == 1:
        Plan['didAccept'] = 0
        Plan['acceptedUIDs'] = list()
        return curModel, curSS, SSmemory, Plan

    # bestModel, bestSS represent best so far
    bestModel = curModel
    bestSS = curSS
    besttargetSS = Plan['targetSS']
    assert np.allclose(besttargetSS.uIDs, bestSS.uIDs)

    # Calculate the current ELBO score
    targetData = Plan['DTargetData']
    totalScale = curModel.obsModel.getDatasetScale(curSS)
    bestELBOobs = curModel.obsModel.calcELBO_Memoized(curSS)
    bestELBOalloc = curModel.allocModel.calcELBOFromSS_NoCacheableTerms(curSS)
    bestELBOobs /= totalScale
    bestELBOalloc /= totalScale

    totalELBOImprovement = 0
    didAccept = 0
    acceptedUIDs = list()
    acceptedPairs = list()
    for delCompUID in Plan['candidateUIDs']:
        if bestSS.K == 1:
            continue  # Don't try to remove the final comp!

        delCompID = np.flatnonzero(bestSS.uIDs == delCompUID)[0]

        # Construct candidate with delCompUID removed
        propModel = bestModel.copy()
        ptargetSS = besttargetSS.copy()
        propSS = bestSS.copy()
        propSS.setMergeFieldsToZero()

        propSS -= ptargetSS
        propModel.update_global_params(propSS)

        # TODO pick component to merge into by ELBO criteria,
        # not just hard choice
        mPairIDs = makeMPairIDsWith(delCompID, propSS.K,  
                                    Plan['candidateIDs'])
        kA, kB = propModel.getBestMergePair(propSS, mPairIDs)
        ELBOgap_cached_rest = propModel.allocModel.\
            calcCachedELBOGap_SinglePair(
                propSS, kA, kB, delCompID=delCompID) / totalScale
        ELBOTerms = propModel.allocModel.\
            calcCachedELBOTerms_SinglePair(
                propSS, kA, kB, delCompID=delCompID)
        
        propSS.mergeComps(kA, kB)
        for key, arr in ELBOTerms.items():
            propSS.setELBOTerm(key, arr, propSS._ELBOTerms._FieldDims[key])
        

        # Here, propSS represents entire dataset
        ptargetSS.removeComp(delCompID)
        propSS += ptargetSS
        propModel.update_global_params(propSS)

        # Verify construction via the merge of all non-target entities
        propCountPlusTarget = propSS.getCountVec().sum() \
            + besttargetSS.getCountVec()[delCompID]
        bestCount = bestSS.getCountVec().sum()
        assert np.allclose(propCountPlusTarget, bestCount)

        # Refine candidate with local/global steps
        didAcceptCur = 0

        for riter in xrange(nRefineIters):
            ptargetLP = propModel.calc_local_params(targetData, **LPkwargs)
            propSS -= ptargetSS
            ptargetSS = propModel.get_global_suff_stats(targetData, ptargetLP,
                                                        doPrecompEntropy=1)
            propSS += ptargetSS
            propModel.update_global_params(propSS)

            propELBOobs = propModel.obsModel.\
                calcELBO_Memoized(propSS) / totalScale
            propELBOalloc = propModel.allocModel.\
                calcELBOFromSS_NoCacheableTerms(propSS) / totalScale
            ELBOgap_cached_target = propModel.allocModel.\
                calcCachedELBOGap_FromSS(
                    besttargetSS, ptargetSS) / totalScale
            
            
            ELBOgap = propELBOobs - bestELBOobs \
                      + propELBOalloc - bestELBOalloc \
                      + ELBOgap_cached_rest + ELBOgap_cached_target
            if not np.isfinite(ELBOgap):
                break
            if ELBOgap > 0 or bestSS.K > Kmax:
                didAcceptCur = 1
                didAccept = 1
                break

        # Log result of this proposal
        curMsg = makeLogMessage(bestSS, besttargetSS, totalELBOImprovement,
            label='cur', compUID=delCompUID)
        propMsg = makeLogMessage(propSS, ptargetSS, totalELBOImprovement,
            label='prop', compUID=delCompUID, didAccept=didAcceptCur)
        DeleteLogger.log(curMsg)
        DeleteLogger.log(propMsg)

        # Update best model/stats to accepted values
        if didAcceptCur:
            totalELBOImprovement += ELBOgap
            acceptedUIDs.append(delCompUID)
            acceptedPairs.append((kA, kB))
            bestELBOobs = propELBOobs
            bestELBOalloc = propELBOalloc
            bestModel = propModel
            besttargetLP = ptargetLP
            besttargetSS = ptargetSS
            bestSS = propSS
            bestSS.setMergeFieldsToZero()
        # << end for loop over each candidate comp

    Plan['didAccept'] = didAccept
    Plan['acceptedUIDs'] = acceptedUIDs

    # Update SSmemory to reflect accepted deletes
    if didAccept:
        bestSS.setELBOFieldsToZero()
        bestSS.setMergeFieldsToZero()
        for batchID in SSmemory:
            SSmemory[batchID].setELBOFieldsToZero()
            SSmemory[batchID].setMergeFieldsToZero()

            if hasValidKey(Plan, 'targetSSByBatch'):
                doEditBatch = batchID in Plan['targetSSByBatch']

            # Decrement : subtract old value of targets in this batch
            # Here, SSmemory has K states
            if doEditBatch:
                SSmemory[batchID] -= Plan['targetSSByBatch'][batchID]

            # Update batch-specific stats with accepted deletes
            #for uID in acceptedUIDs:
            #    kk = np.flatnonzero(SSmemory[batchID].uIDs == uID)[0]
            #    SSmemory[batchID].removeComp(kk)
            for (kA, kB) in acceptedPairs:
                SSmemory[batchID].mergeComps(kA, kB)
            
            assert np.allclose(SSmemory[batchID].uIDs, bestSS.uIDs)
            assert SSmemory[batchID].K == besttargetLP['resp'].shape[1]
            assert SSmemory[batchID].K == bestModel.allocModel.K
            assert SSmemory[batchID].K == bestSS.K

            # Increment : add in new value of targets in this batch
            # Here, SSmemory has K-1 states
            if doEditBatch:
                relUnitIDs = np.flatnonzero(Plan['batchIDs'] == batchID)
                Data_b = targetData.select_subset_by_mask(
                    relUnitIDs, doTrackFullSize=False)
                targetLP_b = bestModel.allocModel.selectSubsetLP(
                    targetData,
                    besttargetLP,
                    relUnitIDs)
                targetSS_b = bestModel.get_global_suff_stats(
                    Data_b, targetLP_b)
                SSmemory[batchID] += targetSS_b

            SSmemory[batchID].setELBOFieldsToZero()
            SSmemory[batchID].setMergeFieldsToZero()

    return bestModel, bestSS, SSmemory, Plan


def makeMPairIDsWith(k, K, excludeIDs):
    """ Create list of possible merge pairs including k, excluding excludeIDs.

    Returns
    -------
    mPairIDs : list of tuples
        each entry is a pair (kA, kB) satisfying kA < kB < K
    """
    mPairIDs = list()
    for j in xrange(K):
        if j in excludeIDs:
            continue
        if j == k:
            continue
        if j < k:
            mPairIDs.append((j,k))
        else:
            mPairIDs.append((k,j))
    return mPairIDs

def makeLogMessage(aggSS, targetSS, ELBO,
                   label='cur', didAccept=-1,
                   compUID=0):

    if label.count('cur'):
        label = " compUID %3d  " % (compUID) + label
    else:
        if didAccept:
            label = '    ACCEPTED ' + label
        else:
            label = '             ' + label

    msg = '%s K=%3d | ELBO % .6f | aggSize %12.4f' \
          % (label,
             targetSS.K,
             ELBO,
             aggSS.getCountVec().sum())

    if label.count('cur'):
        k = np.flatnonzero(aggSS.uIDs == compUID)[0]
        msg += " | aggN[k] %12.4f | targetN[k] %12.4f" \
               % (aggSS.getCountVec()[k], targetSS.getCountVec()[k])
    else:
        msg = msg.replace('ELBO', '    ')
        msg = msg.replace('aggSize', '       ')
    return msg
