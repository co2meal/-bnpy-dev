import numpy as np

from bnpy.deletemove.DEvaluator import runDeleteMoveAndUpdateMemory

def initSingleSeq_SeqAllocContigBlocks(n, Data, hmodel, 
        SS=None,
        Kmax=50,
        initBlockLen=20,
        verbose=0,
        **kwargs):
    ''' Initialize LP and SS for one single sequence.

    Returns
    -------
    SS : aggregate sufficient stats
    SS_n : sufficient stats for this sequence
    LP_n : local parameters for this sequence
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

            print verbose
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

    nRep = 3
    for rep in range(nRep):
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

        if rep == nRep - 1:
            order = np.argsort(-1 * SS.N)
            SS_n.reorderComps(order)
            SS.reorderComps(order)
        hmodel.update_global_params(SS)
        for i in range(3):
            hmodel.allocModel.update_global_params(SS)

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
            Kmax=Kmax,
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
    from IPython import embed; embed()
    return SS, SS_n, LP_n



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
        if verbose:
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
    if allocFieldNames is None:
        allocFieldNames = hmodel.allocModel.getSummaryFieldNames()
        allocFieldDims = hmodel.allocModel.getSummaryFieldDims()

    obsModel = hmodel.obsModel

    if hasattr(Data, 'doc_range'):
        start = Data.doc_range[n]
        stop = Data.doc_range[n + 1]
        T = stop - start
    else:
        start = 0
        T = Data.nObs
    nBlocks = np.maximum(1, int(T // initBlockLen))

    # Loop over each contig block of data, and assign it en masse to one
    # cluster
    Z = -1 * np.ones(T, dtype=np.int32)
    SSagg = SS
    tmpAllocFields = dict()
    if SSagg is None:
        kUID = 0
        Norig = 0
    else:
        kUID = SSagg.K
        Norig = SSagg.N.sum()
        for key in allocFieldNames:
            tmpAllocFields[key] = SSagg.removeField(key)

    # We traverse the current sequence block by block,
    # Indices a,b denote the start and end of the current block
    # *in this sequence*
    # SSactive denotes the most recent current stretch assigned to one comp
    # SSab denotes the current block
    for blockID in xrange(nBlocks):
        if nBlocks == 1:
            a = 0
            b = T
        elif blockID == 0:
            # First block
            a = 0
            b = a + initBlockLen
        elif blockID == nBlocks - 1:
            # Final block
            a = b
            b = T
        else:
            # All interior blocks
            a = b
            b = a + initBlockLen

        SSab = obsModel.calcSummaryStatsForContigBlock(
            Data, a=start + a, b=start + b)
        if blockID == 0:
            Z[a:b] = kUID
            SSactive = SSab
            continue

        ELBOgap = obsModel.calcHardMergeGap_SpecificPairSS(SSactive, SSab)
        if (ELBOgap >= -0.000001):
            # Positive means we prefer to assign block [a,b] to current state
            # So combine the current block into the active block
            # and move on to the next block
            Z[a:b] = kUID
            SSactive += SSab
        else:
            # Negative value means we assign block [a,b] to a new state!
            Z[a:b] = kUID + 1
            SSagg, Z = updateAggSSWithFinishedCurrentBlock(
                SSagg, SSactive, Z, obsModel,
                Kmax=Kmax + Kextra,
                mergeToSimplify=mergeToSimplify)

            # Create a new active block, starting at [a,b]
            SSactive = SSab  # make a soft copy / alias
            kUID = 1 * SSagg.K

    # Final block needs to be recorded.
    SSagg, Z = updateAggSSWithFinishedCurrentBlock(
        SSagg, SSactive, Z, obsModel,
        Kmax=Kmax + Kextra, mergeToSimplify=mergeToSimplify)

    # Compute sequence-specific suff stats
    # This includes allocmodel stats
    if hasattr(Data, 'nDoc'):
        Data_n = Data.select_subset_by_mask([n])
    else:
        Data_n = Data
    LP_n = convertLPFromHardToSoft(dict(Z=Z), Data_n, startIDsAt0=True,
                                   Kmax=SSagg.K)
    LP_n = hmodel.allocModel.initLPFromResp(Data_n, LP_n)
    SS_n = hmodel.get_global_suff_stats(Data_n, LP_n)

    # Verify that our aggregate suff stats
    # represent every single timestep in this sequence
    assert np.allclose(SSagg.N.sum() - Norig, Z.size)
    assert np.allclose(SS_n.N.sum(), Z.size)
    for ii, key in enumerate(allocFieldNames):
        dims = allocFieldDims[ii]
        if key in tmpAllocFields:
            arr, dims2 = tmpAllocFields[key]
            assert dims == dims2
            # Inflate with empties
            if len(dims) == 2 and dims[0] == 'K' and dims[1] == 'K':
                Kcur = arr.shape[0]
                Kextra = SSagg.K - Kcur
                if Kextra > 0:
                    arrBig = np.zeros((SSagg.K, SSagg.K))
                    arrBig[:Kcur, :Kcur] = arr
                    arr = arrBig
            elif len(dims) == 1 and dims[0] == 'K':
                Kcur = arr.size
                Kextra = SSagg.K - Kcur
                if Kextra > 0:
                    arr = np.append(arr, np.zeros(Kextra))

            arr += getattr(SS_n, key)

        else:
            arr = getattr(SS_n, key).copy()
        SSagg.setField(key, arr, dims)

    return Z, SS_n, SSagg
"""
