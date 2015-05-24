import numpy as np

from bnpy.deletemove.DEvaluator import runDeleteMoveAndUpdateMemory
from bnpy.util.StateSeqUtil import calcContigBlocksFromZ

def createSingleSeqLPWithNewStates(Data_n, LP_n, hmodel, 
        SS=None,
        verbose=0,
        Kfresh=3,
        minBlockSize=20,
        nRefineIters=3,
        PRNG=None,
        seed=0,
        **kwargs):
    ''' Propose a new LP that has additional states unique to current sequence.

    Returns
    -------
    LP_n : dict of local params, with K + Knew states (Knew >= 0)
    '''
    if PRNG is None:
        PRNG = np.random.RandomState(seed)
    tempModel = hmodel.copy()
    tempSS = SS
    if tempSS is not None:
        tempSS = tempSS.copy()
        if hasattr(tempSS, 'uIDs'):
            delattr(tempSS, 'uIDs')

    resp_n = LP_n['resp']
    K = resp_n.shape[1]
    origK = K
    propK = K

    propResp = np.zeros((resp_n.shape[0], resp_n.shape[1] + Kfresh))
    propResp[:, :propK] = resp_n

    Z_n = np.argmax(resp_n, axis=1)
    blockSizes, blockStarts = calcContigBlocksFromZ(Z_n)
    nBlocks = len(blockSizes)
    blockOrder = np.arange(nBlocks)
    PRNG.shuffle(blockOrder)
    for blockID in blockOrder:
        if blockSizes[blockID] <= minBlockSize:
            continue
        # Choose random subwindow of this block to create new state
        wSize = PRNG.randint(minBlockSize, high=blockSizes[blockID])
        wStart = PRNG.randint(blockSizes[blockID] - wSize)
        wStart += blockStarts[blockID]
        wStop = wStart + wSize

        # Assign proposed window to new state
        propK = propK + 1
        propResp[wStart:wStop, :propK-1] = 0
        propResp[wStart:wStop, propK-1] = 1

        if propK == origK + Kfresh:
            break

    # Resegment with new states
    propLP_n = tempModel.allocModel.initLPFromResp(
        Data_n, dict(resp=propResp[:, :propK]))
    
    # Refine this segmentation via repeated local/global steps
    for step in xrange(nRefineIters):
        if step == 0:
            if tempSS is not None and tempSS.K < propK:
                tempSS.insertEmptyComps(propK - origK)
        elif step > 0:
            tempSS -= propSS_n
        
        propSS_n = tempModel.get_global_suff_stats(Data_n, propLP_n)
        if tempSS is None:
            tempSS = propSS_n.copy()
        else:
            tempSS += propSS_n

        if step == nRefineIters - 1:
            newIDs = np.arange(origK, propK)
            for newID in reversed(newIDs):
                if tempSS.N[newID] < 1:
                    tempSS.removeComp(newID)

        tempModel.update_global_params(tempSS)
        propLP_n = tempModel.calc_local_params(Data_n, limitMemoryLP=1)

    if propLP_n['evidence'] > LP_n['evidence']:
        return propLP_n, tempModel, tempSS
    else:
        return LP_n, hmodel, SS

def initSingleSeq_SeqAllocContigBlocks(n, Data, hmodel, 
        SS=None,
        Kmax=50,
        initBlockLen=20,
        verbose=0,
        **kwargs):
    ''' Initialize single sequence using new states and existing ones.

    Returns
    -------
    hmodel : HModel
        represents whole dataset
    SS : SuffStatBag
        represents whole dataset
    '''
    if verbose:
        print '<<<<<<<<<<<<<<<< Creating states for seq. %d' % (n)

    assert hasattr(Data, 'nDoc')
    Data_n = Data.select_subset_by_mask([n], doTrackTruth=1)
    obsModel = hmodel.obsModel

    resp_n = np.zeros((Data_n.nObs, Kmax))
    if SS is None:
        K = 1
        SSobsonly = None
    else:
        K = SS.K + 1
        SSobsonly = SS.copy()

    SSprevComp = None

    blockTuples = getListOfContigBlocks(Data_n)
    for blockID, (a, b) in enumerate(blockTuples):
        SSab = obsModel.calcSummaryStatsForContigBlock(
            Data_n, a=a, b=b)

        if hasattr(Data_n, 'TrueParams'):
            Zab = Data_n.TrueParams['Z'][a:b]
            trueStateIDstr = ', '.join(['%d:%d' % (kk, np.sum(Zab==kk)) 
                for kk in np.unique(Zab)])
            trueStateIDstr = ' truth: ' + trueStateIDstr
        else:
            trueStateIDstr = ''

        if blockID == 0:
            SSprevComp = SSab
            resp_n[a:b, K-1] = 1
            if verbose >= 2:
                print "block %d/%d: %d-%d" % (blockID, len(blockTuples), a, b),
                print 'assigned to first state %d' % (K-1), trueStateIDstr
            continue

        # Should we merge current interval [a,b] with previous state?
        ELBOimprovement = obsModel.calcHardMergeGap_SpecificPairSS(
            SSprevComp, SSab)
        if (ELBOimprovement >= -0.000001):
            # Positive means we merge block [a,b] with previous state
            SSprevComp += SSab
            resp_n[a:b, K-1] = 1
            if verbose >= 2:
                print "block %d/%d: %d-%d" % (blockID, len(blockTuples), a, b),
                print 'building on existing state %d' % (K-1), trueStateIDstr

        else:
            # Insert finished block as a new component
            if SSobsonly is None:
                SSobsonly = SSprevComp
            else:
                # Remove any keys associated with alloc model
                for key in SSobsonly._FieldDims.keys():
                    if key not in SSprevComp._FieldDims:
                        del SSobsonly._FieldDims[key]
                SSobsonly.insertComps(SSprevComp)

            # Try to merge this new state into an already existing state
            resp_n, K, SSobsonly = mergeDownLastStateIfPossible(
                resp_n, K, SSobsonly, SSprevComp, obsModel, verbose=verbose)

            # Assign block [a,b] to a new state!
            K += 1
            resp_n[a:b, K-1] = 1
            SSprevComp = SSab
            if verbose >= 2:
                print "block %d/%d: %d-%d" % (blockID, len(blockTuples), a, b),
                print 'building on existing state %d' % (K-1), trueStateIDstr

    # Deal with final block
    if SSobsonly is not None:
        # Remove any keys associated with alloc model
        for key in SSobsonly._FieldDims.keys():
            if key not in SSprevComp._FieldDims:
                del SSobsonly._FieldDims[key]
        SSobsonly.insertComps(SSprevComp)
        resp_n, K, SSobsonly = mergeDownLastStateIfPossible(
            resp_n, K, SSobsonly, SSprevComp, obsModel, verbose=verbose)
    del SSobsonly

    hmodel, SS, SS_n = refineSegmentationViaLocalGlobalSteps(
        SS, hmodel, Data_n, resp_n, K,
        verbose=verbose)

    hmodel, SS = removeSmallUniqueCompsViaDelete(
        SS, hmodel, Data_n, SS_n, 
        verbose=verbose,
        initBlockLen=initBlockLen)

    return hmodel, SS


def refineSegmentationViaLocalGlobalSteps(
        SS, hmodel, Data_n, resp_n, K,
        verbose=0,
        nSteps=3,
        ):
    '''

    Returns
    -------
    hmodel : HModel
        consistent with entire dataset
    SS : SuffStatBag
        representing whole dataset
    SS_n : SuffStatBag
        represents only current sequence n
    '''
    for rep in range(nSteps):
        if rep == 0:
            LP_n = dict(resp=resp_n[:, :K])
            LP_n = hmodel.allocModel.initLPFromResp(Data_n, LP_n)
        else:
            LP_n = hmodel.calc_local_params(Data_n)

        SS_n = hmodel.get_global_suff_stats(Data_n, LP_n)
        assert np.allclose(SS_n.N.sum(), Data_n.nObs)
        
        # Update whole-dataset stats
        if rep == 0:
            prevSS_n = SS_n
            if SS is None:
                SS = SS_n.copy()
            else:
                SS.insertEmptyComps(SS_n.K - SS.K)
                SS += SS_n
        else:
            SS -= prevSS_n
            SS += SS_n
            prevSS_n = SS_n

        # Reorder the states from big to small
        # using aggregate stats from entire dataset
        if rep == nSteps - 1:
            order = np.argsort(-1 * SS.N)
            SS_n.reorderComps(order)
            SS.reorderComps(order)
        hmodel.update_global_params(SS)
        for i in range(3):
            hmodel.allocModel.update_global_params(SS)
    return hmodel, SS, SS_n

def removeSmallUniqueCompsViaDelete(
        SS, hmodel, Data_n, SS_n,
        initBlockLen=20,
        verbose=0):
    ''' Remove small comps unique to sequence n if accepted by delete move.

    Returns
    -------
    hmodel : HModel
        consistent with entire dataset
    SS : SuffStatBag
        representing whole dataset
    '''
    # Try deleting any UIDs unique to this sequence with small size
    K_n = np.sum(SS_n.N > 0.01)
    IDs_n = np.flatnonzero(SS_n.N <= initBlockLen)
    candidateUIDs = list()
    for uID in IDs_n:
        if np.allclose(SS.N[uID], SS_n.N[uID]):
            candidateUIDs.append(uID)
    if len(candidateUIDs) > 0:
        SS.uIDs = np.arange(SS.K)
        SS_n.uIDs = np.arange(SS.K)
        Plan = dict(
            candidateUIDs=candidateUIDs,
            DTargetData=Data_n,
            targetSS=SS_n,
            )
        hmodel, SS, _, DResult = runDeleteMoveAndUpdateMemory(
            hmodel, SS, Plan,
            nRefineIters=2,
            LPkwargs=dict(limitMemoryLP=1),
            SSmemory=None,
            )
        if verbose >= 2:
            print 'Cleanup deletes: %d/%d accepted' % (
                DResult['nAccept'], DResult['nTotal'])
        delattr(SS, 'uIDs')
        delattr(SS_n, 'uIDs')
        K_n -= DResult['nAccept']
    if verbose:
        K_true = len(np.unique(Data_n.TrueParams['Z']))
        print 'Total states: %d' % (SS.K)
        print 'Total states in cur seq: %d' % (K_n)
        print 'Total true states in cur seq: %d' % (K_true)
    return hmodel, SS

def mergeDownLastStateIfPossible(resp_n, K_n, SS, SSprevComp, obsModel,
        verbose=False):
    ''' Try to merge the last state into the existing set.

    Returns
    -------
    resp_n : 2D array, size T x Kmax
    K_n : int
        total number of states used by sequence n
    SS_n : SuffStatBag
    '''
    if K_n == 1:
        return resp_n, K_n, SS

    kB = K_n - 1
    gaps = np.zeros(K_n - 1)
    for kA in xrange(K_n - 1):
        SS_kA = SS.getComp(kA, doCollapseK1=0)
        gaps[kA] = obsModel.calcHardMergeGap_SpecificPairSS(SS_kA, SSprevComp)
    kA = np.argmax(gaps)
    if gaps[kA] > -0.00001:
        if verbose >= 2:
            print "merging this block into state %d" % (kA)
        SS.mergeComps(kA, kB)
        mask = resp_n[:, kB] > 0
        resp_n[mask, kA] = 1
        resp_n[mask, kB] = 0
        return resp_n, K_n - 1, SS
    else:
        # No merge would be accepted
        return resp_n, K_n, SS

def getListOfContigBlocks(Data_n=None, initBlockLen=20, T=None):
    ''' Generate tuples identifying contiguous blocks within given seq.

    Examples
    --------
    >>> getListOfContigBlocks(T=31, initBlockLen=15)
    [(0, 15), (15, 30), (30, 31)]
    >>> getListOfContigBlocks(T=50, initBlockLen=20)
    [(0, 20), (20, 40), (40, 50)]
    >>> getListOfContigBlocks(T=20, initBlockLen=10)
    [(0, 10), (10, 20)]
    '''
    if T is None:
        start = Data_n.doc_range[0]
        stop = Data_n.doc_range[1]
        T = stop - start
    nBlocks = np.maximum(1, int(np.ceil(T / float(initBlockLen))))
    if nBlocks == 1:
        return [(0, T)]
    else:
        bList = list()
        for blockID in range(nBlocks):
            a = blockID * initBlockLen
            b = (blockID+1) * initBlockLen
            if blockID == nBlocks - 1:
                b = T
            bList.append((a, b))
        return bList
