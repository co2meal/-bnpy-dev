import copy
import numpy as np
import warnings

from bnpy.deletemove.DEvaluator import runDeleteMoveAndUpdateMemory
from bnpy.util.StateSeqUtil import calcContigBlocksFromZ
from bnpy.data.XData import XData
from bnpy.allocmodel.hmm.HDPHMMUtil import calcELBOForSingleSeq_FromLP

def createSingleSeqLPWithNewStates(Data_n, LP_n, hmodel, 
        SS=None,
        verbose=0,
        Kfresh=3,
        minBlockSize=20,
        maxBlockSize=np.inf,
        nRefineIters=3,
        creationProposalName='mixture',
        PRNG=None,
        seed=0,
        **kwargs):
    ''' Propose a new LP that has additional states unique to current sequence.

    Returns
    -------
    LP_n : dict of local params, with K + Knew states (Knew >= 0)
    '''
    kwargs['minBlockSize'] = minBlockSize
    kwargs['maxBlockSize'] = maxBlockSize
    kwargs['PRNG'] = PRNG

    if PRNG is None:
        PRNG = np.random.RandomState(seed)
    tempModel = hmodel.copy()
    tempSS = SS
    if tempSS is not None:
        tempSS = tempSS.copy()
        if hasattr(tempSS, 'uIDs'):
            delattr(tempSS, 'uIDs')

    resp_n = LP_n['resp']
    N, origK = resp_n.shape

    propResp = np.zeros((N, origK + Kfresh))
    propResp[:, :origK] = resp_n

    Z_n = np.argmax(resp_n, axis=1)

    # Target comps with sufficient size in this sequence
    uIDs = np.unique(Z_n)
    sizes = np.asarray([np.sum(Z_n == uID) for uID in uIDs])
    elig_mask = sizes >= minBlockSize
    if np.sum(elig_mask) == 0:
        return LP_n, hmodel, SS
    sizes = sizes[elig_mask]
    uIDs = uIDs[elig_mask]
    ktarget = PRNG.choice(uIDs)

    if creationProposalName == 'mixture':
        propResp, propK = proposeNewResp_DPMixtureOnTargetData(Z_n, propResp,
            ktarget=ktarget,
            origK=origK,
            Kfresh=np.minimum(Kfresh,2),
            tempModel=tempModel,
            tempSS=tempSS,
            Data_n=Data_n,
            **kwargs)
    elif creationProposalName == 'subdivideExistingBlocks':
        propResp, propK = proposeNewResp_subdivideExistingBlocks(
            Z_n, propResp, PRNG=PRNG,
            origK=origK, Kfresh=Kfresh, 
            minBlockSize=minBlockSize, 
            maxBlockSize=maxBlockSize)
    elif creationProposalName == 'randBlocks':
        propResp, propK = proposeNewResp_randBlocks(
            Z_n, propResp, PRNG=PRNG,
            origK=origK, Kfresh=Kfresh, 
            minBlockSize=minBlockSize, 
            maxBlockSize=maxBlockSize)
    else:
        msg = "Unrecognized creationProposalName: %s" % (creationProposalName)
        raise NotImplementedError(msg)
    if propK == origK:
        return LP_n, hmodel, SS

    # Create complete LP fields from the proposed resp segmentation
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

        tempModel.update_global_params(tempSS)
        propLP_n = tempModel.calc_local_params(Data_n, limitMemoryLP=1)

    # Do a few steps to remove empties     
    extraIDs_remaining = np.arange(origK, propK).tolist()
    nEmpty = np.sum(tempSS.N[origK:] <= 1)
    if verbose:
        print extraIDs_remaining, '<< original extra ids'
    while nEmpty > 0:
        if verbose:
            print ['%.2f' % (x) for x in tempSS.N[origK:]]
            print extraIDs_remaining, '<< remaining extra ids before'

        L = len(extraIDs_remaining)
        for kLoc, kk in enumerate(reversed(np.arange(origK, tempSS.K))):
            if tempSS.N[kk] <= 1:
                if verbose:
                    print 'removing comp %d' % (kk)
                tempSS.removeComp(kk)
                propSS_n.removeComp(kk)
                extraIDs_remaining.pop(L - kLoc - 1)
        if verbose:
            print extraIDs_remaining, '<< remaining extra ids AFTER'

        # Make model have consistent num of global params
        tempModel.update_global_params(tempSS)

        propLP_n = tempModel.calc_local_params(Data_n, limitMemoryLP=1)
        tempSS -= propSS_n
        propSS_n = tempModel.get_global_suff_stats(Data_n, propLP_n)
        tempSS += propSS_n
        nEmpty = np.sum(tempSS.N[origK:] <= 1)

    propScore = calcELBOForSingleSeq_FromLP(Data_n, propLP_n, hmodel)
    curScore = calcELBOForSingleSeq_FromLP(Data_n, LP_n, hmodel)
    doAccept = propScore > curScore

    if kwargs['doVizSeqCreate']:
        print 'propLP K %3d evidence %.3f  score %.6f' % (
            propLP_n['resp'].shape[1], propLP_n['evidence'], propScore)
        print ' curLP K %3d evidence %.3f  score %.6f' % (
            LP_n['resp'].shape[1], LP_n['evidence'], curScore)

        if doAccept:
            Nnew = propLP_n['resp'][:, origK:].sum(axis=0)
            massNew_str = ' '.join(['%.2f' % (x) for x in Nnew])
            print 'ACCEPTED! Mass of new states: [%s]' % (massNew_str)
        else:
            print 'rejected'
        showProposal(Data_n, Z_n, propResp, propLP_n, extraIDs_remaining,
            doAccept=doAccept)
        print ''
        print ''
        print ''

    if doAccept:
        return propLP_n, tempModel, tempSS
    else:
        return LP_n, hmodel, SS

def showProposal(Data_n, Z_n, propResp, propLP_n, origIDs,
        doAccept=None):
    from matplotlib import pylab
    from bnpy.util.StateSeqUtil import alignEstimatedStateSeqToTruth
    from bnpy.util.StateSeqUtil import makeStateColorMap
    origIDs = np.asarray(origIDs)
    Ztrue = np.asarray(Data_n.TrueParams['Z'], np.int32)

    # Map Ztrue to compact set of uniqueIDs
    # that densely covers 0, 1, 2, ... Kunique
    # instead of (possibly) covering 0, 1, 2, ... Ktrue, Ktrue >> Kunique
    uLabels = np.unique(Ztrue)
    ZtrueA = -1 * np.ones_like(Ztrue)
    for uLoc, uID in enumerate(uLabels):
        mask = Ztrue == uID
        ZtrueA[mask] = uLoc
    assert np.all(ZtrueA >= 0)
    assert np.all(ZtrueA < len(uLabels))
    Ztrue = ZtrueA

    curZA, AlignInfo = alignEstimatedStateSeqToTruth(
        Z_n, Ztrue, returnInfo=1)

    nTrue = Ztrue.max() + 1
    Kcur = curZA.max() + 1
    nExtra =  curZA.max() + 1 - nTrue
    Kmax = np.maximum(nTrue, nTrue + nExtra)

    propZstart = propResp.argmax(axis=1)
    propZAstart = alignEstimatedStateSeqToTruth(
        propZstart, Ztrue, useInfo=AlignInfo)

    # Relabel each unique state represented in propZrefined
    # so that state ids correspond to those in propZstart

    # Step 1: propZrefined states all have negative ids
    propZrefined = -1 * propLP_n['resp'].argmax(axis=1) - 1
    for origLoc, kk in enumerate(range(Kcur, propZrefined.max() + 1)):
        propZrefined[propZrefined == -1*kk-1] = origIDs[origLoc]
    # Transform any original states back to original ids
    propZrefined[propZrefined < 0] = -1 * propZrefined[propZrefined < 0] - 1

    propZArefined = alignEstimatedStateSeqToTruth(
        propZrefined, Ztrue, useInfo=AlignInfo)

    nHighlight = propZAstart.max() + 1 - Kmax

    cmap = makeStateColorMap(nTrue=nTrue,
        nExtra=np.maximum(0, nExtra),
        nHighlight=np.maximum(0, nHighlight))

    Kmaxxx = np.maximum(Kmax, propZAstart.max()+1)
    Kmaxxx = np.maximum(Kmaxxx, propZArefined.max()+1)
    imshowArgs = dict(interpolation='nearest', 
        aspect=Z_n.size/1.0,
        cmap=cmap,
        vmin=0, vmax=Kmaxxx-1)
    pylab.subplots(nrows=4, ncols=1, figsize=(12,5))
    # show ground truth
    #print 'UNIQUE IDS in each aligned Z'
    #print np.unique(Ztrue)
    #print np.unique(curZA)
    #print np.unique(propZstart), '>', np.unique(propZAstart)
    #print np.unique(propZrefined), '>', np.unique(propZArefined)

    ax = pylab.subplot(4,1,1)
    pylab.imshow(Ztrue[np.newaxis,:], **imshowArgs)
    pylab.title('Ground truth')
    pylab.xticks([]); pylab.yticks([]);
    # show current
    pylab.subplot(4,1,2, sharex=ax)
    pylab.imshow(curZA[np.newaxis,:], **imshowArgs)
    pylab.title('Current estimate')
    pylab.xticks([]); pylab.yticks([]);
    # show init
    pylab.subplot(4,1,3, sharex=ax)
    pylab.imshow(propZAstart[np.newaxis,:], **imshowArgs)
    pylab.title('Initial proposal')
    pylab.xticks([]); pylab.yticks([]);

    # show final
    pylab.subplot(4,1,4, sharex=ax)
    pylab.imshow(propZArefined[np.newaxis,:], **imshowArgs)
    if doAccept is None:
        acceptMsg = ''
    elif doAccept:
        acceptMsg = '  ACCEPTED!'
    else:
        acceptMsg = '  REJECTED!'

    pylab.title('Refined proposal' + acceptMsg)
    pylab.yticks([]);
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        pylab.tight_layout()
    pylab.show(block=False)
    keypress = raw_input("Press any key to continue>>>")
    pylab.close()

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
        print a, b
        propResp[a:b, :origK] = 0
        propResp[a:b, origK+kfresh] = 1
    return propResp, origK + Kfresh

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
    return propResp, kfresh


def proposeNewResp_DPMixtureOnTargetData(Z_n, propResp,
        ktarget=0,
        tempModel=None,
        tempSS=None,
        Data_n=None,
        origK=0,
        Kfresh=3,
        minBlockSize=0,
        maxBlockSize=10,
        nVBIters=3,
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
    myHModel.init_global_params(targetData, initname='randexamplesbydist',
        K=Kfresh)
    Kfresh = myHModel.obsModel.K
    mergeIsPromising = True
    while Kfresh > 1 and mergeIsPromising:
        for vbiter in xrange(nVBIters):
            targetLP = myHModel.calc_local_params(targetData)
            targetSS = myHModel.get_global_suff_stats(targetData, targetLP)
            # Delete unnecessarily small comps
            if vbiter == nVBIters - 1:
                smallIDs = np.flatnonzero(targetSS.getCountVec() < 5)
                for kdel in reversed(smallIDs):
                    targetSS.removeComp(kdel)
            # Global step
            myHModel.update_global_params(targetSS)

        # Do merges
        mPairIDs, MM = MergePlanner.preselect_candidate_pairs(myHModel, targetSS,
             preselect_routine='wholeELBO',
             doLimitNumPairs=0,
             returnScoreMatrix=1,
             **kwargs)
        targetLP = myHModel.calc_local_params(targetData)
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


"""

def proposeNewResp_randomSubwindowsOfContigBlocks(Z_n, propResp,
        origK=0,
        Kfresh=3,
        PRNG=np.random.RandomState,
        minBlockSize=0,
        maxBlockSize=10,
        **kwargs):
    ''' Create new value of resp matrix by randomly breaking up contig. blocks.

    Returns
    -------
    propResp : 2D array, N x K'
    '''
    propK = origK

    blockSizes, blockStarts = calcContigBlocksFromZ(Z_n)
    nBlocks = len(blockSizes)
    blockOrder = np.arange(nBlocks)
    PRNG.shuffle(blockOrder)
    for blockID in blockOrder:
        if blockSizes[blockID] <= minBlockSize:
            continue
        # Choose random subwindow of this block to create new state
        # min achievable size: minBlockSize + 1
        # max size: maxBlockSize
        maxBlockSize = np.minimum(maxBlockSize, blockSizes[blockID])
        wSize = PRNG.randint(minBlockSize, high=maxBlockSize)
        wStart = PRNG.randint(blockSizes[blockID] - wSize)
        wStart += blockStarts[blockID]
        wStop = wStart + wSize

        # Assign proposed window to new state
        propK = propK + 1
        propResp[wStart:wStop, :propK-1] = 0
        propResp[wStart:wStop, propK-1] = 1

        if propK == origK + Kfresh:
            break
    return propResp, propK
"""
