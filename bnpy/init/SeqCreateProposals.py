import numpy as np
import copy
import warnings

from bnpy.util.StateSeqUtil import calcContigBlocksFromZ
from bnpy.data.XData import XData

def proposeNewResp_randBlocks(Z_n, propResp,
        origK=0,
        PRNG=np.random.RandomState,
        Kfresh=3,
        minBlockSize=1,
        maxBlockSize=10,
        **kwargs):
    ''' Create new value of resp matrix with randomly-placed new blocks.

    We create Kfresh new blocks in total.
    Each one can potentially wipe out some (or all) of previous blocks.

    Returns
    -------
    propResp : 2D array of size N x Kmax
    propK : int
        total number of states used in propResp array
    '''
    # Unpack and make sure size limits work out
    T = Z_n.size
    if minBlockSize >= T:
        return propResp, origK
    maxBlockSize = np.minimum(maxBlockSize, T)

    for kfresh in range(Kfresh):
        blockSize = PRNG.randint(minBlockSize, maxBlockSize)
        a = PRNG.randint(0, T - blockSize + 1)
        b = a + blockSize
        propResp[a:b, :origK] = 0
        propResp[a:b, origK+kfresh] = 1
    return propResp, origK + Kfresh



def proposeNewResp_bisectExistingBlocks(Z_n, propResp,
        Data_n=None,
        tempModel=None,
        origK=0,
        PRNG=np.random.RandomState,
        Kfresh=3,
        PastAttemptLog=dict(),
        **kwargs):
    ''' Create new value of resp matrix with randomly-placed new blocks.

    We create Kfresh new blocks in total.
    Each one can potentially wipe out some (or all) of previous blocks.

    Returns
    -------
    propResp : 2D array of size N x Kmax
    propK : int
        total number of states used in propResp array
    '''
    # Iterate over current contig blocks 
    blockSizes, blockStarts = calcContigBlocksFromZ(Z_n)
    nBlocks = len(blockSizes)

    sortOrder = np.argsort(-1 * blockSizes)
    blockSizes = blockSizes[sortOrder]
    blockStarts = blockStarts[sortOrder]

    kfresh = 0 # number of new states added
    for blockID in range(nBlocks):
        if kfresh >= Kfresh:
            break
        a = blockStarts[blockID]
        b = blockStarts[blockID] + blockSizes[blockID]

        # Avoid overlapping with previous attempts that failed
        maxOverlapWithPreviousFailure = 0.0
        for (preva,prevb), prevm in PastAttemptLog.items():
            # skip previous attempts that succeed
            if prevm > preva:
                continue
            Tunion = np.maximum(b, prevb) - np.minimum(a, preva) 
            minb = np.minimum(b, prevb)
            maxa = np.maximum(a, preva)
            if maxa < minb:
                Tintersect = minb - maxa
            else:
                Tintersect = 0
                continue
            IoU = Tintersect / float(Tunion)
            maxOverlapWithPreviousFailure = np.maximum(
                maxOverlapWithPreviousFailure, IoU) 
        if maxOverlapWithPreviousFailure > 0.95:
            print 'SKIPPING BLOCK %d,%d with overlap %.2f' % (
                a, b, maxOverlapWithPreviousFailure)
            continue

        stride = int(np.ceil((b - a) / 25.0))
        stride = np.maximum(1, stride)
        offset = PRNG.choice(np.arange(stride))
        a += offset
        bestm = findBestCutForBlock(Data_n, tempModel,
            a=a,
            b=b,
            stride=stride)

        PastAttemptLog[(a,b)] = bestm
        # print 'BEST BISECTION CUT: [%4d, %4d, %4d] w/ stride %d' % (
        #     a, bestm, b, stride)
        if bestm == a:
            propResp[a:b, :origK] = 0
            propResp[a:b, origK + kfresh] = 1
            kfresh += 1
        else:
            propResp[a:bestm, :origK] = 0
            propResp[a:bestm, origK + kfresh] = 1
            kfresh += 1

            if kfresh >= Kfresh:
                break

            propResp[bestm:b, :origK] = 0           
            propResp[bestm:b, origK + kfresh] = 1
            kfresh += 1
    return propResp, origK + kfresh


def proposeNewResp_subdivideExistingBlocks(Z_n, propResp,
        origK=0,
        PRNG=np.random.RandomState,
        nStatesToEdit=3,
        Kfresh=5,
        minBlockSize=1,
        maxBlockSize=10,
        **kwargs):
    ''' Create new value of resp matrix with new blocks.

    We select nStatesToEdit states to change.
    For each one, we take each contiguous block,
        defined by interval [a,b]
        and subdivide that interval into arbitrary number of states
            [a, l1, l2, l3, ... lK, b]
        where the length of each new block is drawn from
            l_i ~ uniform(minBlockSize, maxBlockSize)

    Returns
    -------
    propResp : 2D array of size N x Kmax
    propK : int
        total number of states used in propResp array
    '''
    # Unpack and make sure size limits work out
    T = Z_n.size
    if minBlockSize >= T:
        return propResp, origK
    maxBlockSize = np.minimum(maxBlockSize, T)

    blockSizes, blockStarts = calcContigBlocksFromZ(Z_n)
    nBlocks = len(blockSizes)

    candidateStateIDs = list()
    candidateBlockIDsByState = dict()
    for blockID in xrange(nBlocks):
        stateID = Z_n[blockStarts[blockID]]
        if blockSizes[blockID] >= minBlockSize:
            candidateStateIDs.append(stateID)
            if stateID not in candidateBlockIDsByState:
                candidateBlockIDsByState[stateID] = list()
            candidateBlockIDsByState[stateID].append(blockID)

    if len(candidateStateIDs) == 0:
        return propResp, origK
    selectedStateIDs = PRNG.choice(candidateStateIDs, 
        size=np.minimum(len(candidateStateIDs), nStatesToEdit),
        replace=False)

    kfresh = origK
    for stateID in selectedStateIDs:
        if kfresh >= Kfresh:
            break

        # Find contig blocks assigned to this state
        for blockID in candidateBlockIDsByState[stateID]:
            if kfresh >= Kfresh:
                break
            a = blockStarts[blockID]
            b = a + blockSizes[blockID]
            maxSize = np.minimum(b-a, maxBlockSize)
            avgSize = (maxSize + minBlockSize) / 2
            expectedLen = avgSize * Kfresh
            if expectedLen < (b - a):
                intervalLocs = [PRNG.randint(a, b-expectedLen)]
            else:
                intervalLocs = [a]
            for ii in range(Kfresh):
                nextBlockSize = PRNG.randint(minBlockSize, maxSize)
                intervalLocs.append(nextBlockSize + intervalLocs[ii])
                if intervalLocs[ii+1] >= b:
                    break
            intervalLocs = np.asarray(intervalLocs, dtype=np.int32)
            intervalLocs = np.minimum(intervalLocs, b)
            # print 'Current interval   : [ %d, %d]' % (a, b)
            # print 'Subdivided interval: ', intervalLocs
            for iID in range(intervalLocs.size-1):
                if kfresh >= Kfresh:
                    break
                prevLoc = intervalLocs[iID]
                curLoc = intervalLocs[iID+1]
                propResp[prevLoc:curLoc, :] = 0
                propResp[prevLoc:curLoc, kfresh] = 1
                kfresh += 1
    assert kfresh >= origK
    return propResp, kfresh



def proposeNewResp_uniquifyExistingBlocks(Z_n, propResp,
        tempSS=None,
        origK=0,
        PRNG=np.random.RandomState,
        nStatesToEdit=None,
        Kfresh=5,
        minBlockSize=1,
        maxBlockSize=10,
        **kwargs):
    ''' Create new resp matrix with new unique blocks from existing blocks.

    We select at most nStatesToEdit states to change,
    where each one has multiple contiguous blocks.

    For each one, we take all its contiguous blocks,
    defined by intervals [a1,b1], [a2,b2], ... [aN, bN], ...
    and make a unique state for each interval.
     
    Returns
    -------
    propResp : 2D array of size N x Kmax
    propK : int
        total number of states used in propResp array
    '''
    if nStatesToEdit is None:
        nStatesToEdit = Kfresh

    # Unpack and make sure size limits work out
    T = Z_n.size
    if minBlockSize >= T:
        return propResp, origK
    maxBlockSize = np.minimum(maxBlockSize, T)

    blockSizes, blockStarts = calcContigBlocksFromZ(Z_n)
    nBlocks = len(blockSizes)

    candidateBlockIDsByState = dict()
    for blockID in xrange(nBlocks):
        stateID = Z_n[blockStarts[blockID]]
        if stateID not in candidateBlockIDsByState:
            candidateBlockIDsByState[stateID] = list()
        candidateBlockIDsByState[stateID].append(blockID)

    candidateStateIDs = list()
    for stateID in candidateBlockIDsByState.keys():
        hasJustOneBlock = len(candidateBlockIDsByState[stateID]) < 2 
        if tempSS is None:
            appearsOnlyInThisSeq = True
        else:
            appearsOnlyInThisSeq = tempSS.N[stateID] < 1.0

        if hasJustOneBlock and appearsOnlyInThisSeq:
            del candidateBlockIDsByState[stateID]
        else:
            candidateStateIDs.append(stateID)
    
    if len(candidateStateIDs) == 0:
        return propResp, origK
    selectedStateIDs = PRNG.choice(candidateStateIDs, 
        size=np.minimum(len(candidateStateIDs), nStatesToEdit),
        replace=False)

    kfresh = 0
    for stateID in selectedStateIDs:
        if kfresh >= Kfresh:
            break
        # Make each block assigned to this state its own unique proposed state
        for blockID in candidateBlockIDsByState[stateID]:
            if kfresh >= Kfresh:
                break
            a = blockStarts[blockID]
            b = a + blockSizes[blockID]
            propResp[a:b, :] = 0
            propResp[a:b, origK + kfresh] = 1
            kfresh += 1
    return propResp, origK + kfresh

def proposeNewResp_dpmixture(Z_n, propResp,
        ktarget=0,
        tempModel=None,
        tempSS=None,
        Data_n=None,
        origK=0,
        Kfresh=3,
        nVBIters=3,
        PRNG=None,
        **kwargs):
    ''' Create new resp matrix by DP mixture clustering of subsampled data.

    Returns
    -------
    propResp : 2D array, N x K'
    '''
    # Avoid circular imports
    from bnpy.allocmodel import DPMixtureModel
    from bnpy import HModel
    from bnpy.mergemove import MergePlanner, MergeMove

    relDataIDs = np.flatnonzero(Z_n == ktarget)
    if hasattr(Data_n, 'Xprev'):
        Xprev = Data_n.Xprev[relDataIDs]
    else:
        Xprev = None
    targetData = XData(X=Data_n.X[relDataIDs],
        Xprev=Xprev)

    myDPModel = DPMixtureModel('VB', gamma0=10)
    myObsModel = copy.deepcopy(tempModel.obsModel)
    delattr(myObsModel, 'Post')
    myObsModel.ClearCache()

    myHModel = HModel(myDPModel, myObsModel)
    initname = PRNG.choice(['randexamplesbydist', 'randcontigblocks'])
    myHModel.init_global_params(targetData, K=Kfresh, 
        initname=initname,
        **kwargs)

    Kfresh = myHModel.obsModel.K
    mergeIsPromising = True
    while Kfresh > 1 and mergeIsPromising:
        for vbiter in xrange(nVBIters):
            targetLP = myHModel.calc_local_params(targetData)
            targetSS = myHModel.get_global_suff_stats(targetData, targetLP)
            # Delete unnecessarily small comps
            if vbiter == nVBIters - 1:
                smallIDs = np.flatnonzero(targetSS.getCountVec() <= 1)
                for kdel in reversed(smallIDs):
                    if targetSS.K > 1:
                        targetSS.removeComp(kdel)
            # Global step
            myHModel.update_global_params(targetSS)

        # Do merges
        mPairIDs, MM = MergePlanner.preselect_candidate_pairs(myHModel, targetSS,
             preselect_routine='wholeELBO',
             doLimitNumPairs=0,
             returnScoreMatrix=1,
             **kwargs)
        targetLP = myHModel.calc_local_params(targetData,
            mPairIDs=mPairIDs, limitMemoryLP=1)
        targetSS = myHModel.get_global_suff_stats(targetData, targetLP,
            mPairIDs=mPairIDs,
            doPrecompEntropy=1,
            doPrecompMergeEntropy=1)
        myHModel.update_global_params(targetSS)
        curELBO = myHModel.calc_evidence(SS=targetSS)
        myHModel, targetSS, curELBO, Info = MergeMove.run_many_merge_moves(
            myHModel, targetSS, curELBO,
            mPairIDs, M=MM,
            isBirthCleanup=1)
        mergeIsPromising = len(Info['AcceptedPairs']) > 0
        Kfresh = targetSS.K

    if mergeIsPromising:
        targetLP = myHModel.calc_local_params(targetData)
    propResp[relDataIDs, :] = 0
    propResp[relDataIDs, origK:origK+Kfresh] = targetLP['resp']
    return propResp, origK+Kfresh


def findBestCutForBlock(Data_n, tempModel,
        a=0, b=400,
        stride=3):
    ''' Search for best cut point over interval [a,b] in provided sequence n.
    '''
    tempModel = tempModel.copy()
    def calcObsModelELBOForInterval(SSab):
        tempModel.obsModel.update_global_params(SSab)
        ELBOab = tempModel.obsModel.calc_evidence(None, SSab, None)
        return ELBOab

    SSab = tempModel.obsModel.calcSummaryStatsForContigBlock(
        Data_n, a=a, b=b)
    ELBOab = calcObsModelELBOForInterval(SSab)

    # Initialize specific suff stat bags for intervals [a,m] and [m,b]
    SSmb = SSab
    SSam = SSab.copy()
    SSam.setAllFieldsToZero()
    assert np.allclose(SSam.N.sum() + SSmb.N.sum(), b-a)

    score = -1 * np.inf * np.ones(b-a)
    score[0] = ELBOab  
    for m in np.arange(a+stride, b, stride):
        assert m > a
        assert m < b
        # Grab segment recently converted to [a,m] interval
        SSstride = tempModel.obsModel.calcSummaryStatsForContigBlock(
            Data_n, a=(m - stride), b=m)
        SSam += SSstride
        SSmb -= SSstride
        assert np.allclose(SSam.N.sum() + SSmb.N.sum(), b-a)

        ELBOam = calcObsModelELBOForInterval(SSam)
        ELBOmb = calcObsModelELBOForInterval(SSmb)
        score[m - a] = ELBOam + ELBOmb
        #print a, m, b, 'score %.3e  Nam %.3f' % (score[m - a], SSam.N[0])
  
    bestm = a + np.argmax(score)
    return bestm
