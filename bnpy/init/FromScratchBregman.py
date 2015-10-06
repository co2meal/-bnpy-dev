'''
FromScratchBregman.py

Initialize suff stats for observation models via Bregman clustering.
'''

import numpy as np
import bnpy.data

from FromTruth import \
    convertLPFromHardToSoft, \
    convertLPFromTokensToDocs, \
    convertLPFromDocsToTokens

def init_global_params(hmodel, Data, **kwargs):
    ''' Initialize parameters of observation model.
    
    Post Condition
    --------------
    hmodel internal parameters updated to reflect sufficient statistics.
    '''
    SS, Info = initSS_BregmanDiv(
        Data, hmodel, includeAllocSummary=True, **kwargs)
    hmodel.allocModel.update_global_params(SS)
    hmodel.obsModel.update_global_params(SS)

    Info['targetSS'] = SS
    return Info

def initSS_BregmanDiv(
        Dslice=None, 
        curModel=None, 
        curLPslice=None,
        K=5, 
        ktarget=None, 
        seed=0,
        includeAllocSummary=False,
        NiterForBregmanKMeans=1,
        **kwargs):
    ''' Create observation model statistics via Breg. distance sampling.

    Args
    ------
    Data

    Keyword args
    ------------
    TODO

    Returns
    -------
    xSS : SuffStatBag
    DebugInfo : dict
        contains info about provenance of this initialization.
    '''
    # Reformat any keyword argument to drop 
    # prefix of 'b_' or 'init_'
    for key, val in kwargs.items():
        if key.startswith('b_'):
            newkey = key[2:]
            kwargs[newkey] = val
            del kwargs[key]
        elif key.startswith('init_'):
            newkey = key[5:]
            kwargs[newkey] = val
            del kwargs[key]
    if 'NiterForBregmanKMeans' in kwargs:
        NiterForBregmanKMeans = kwargs['NiterForBregmanKMeans']

    Niter = np.maximum(NiterForBregmanKMeans, 0)
    PRNG = np.random.RandomState(int(seed))
    DebugInfo, targetData, targetX, targetW, chosenRespIDs = \
        makeDataSubsetByThresholdResp(
            Dslice,
            curModel,
            curLPslice,
            ktarget,
            K=K,
            **kwargs)
    if targetData is None:
        return None, DebugInfo

    K = np.minimum(K, targetX.shape[0])
    # Perform plusplus initialization + Kmeans clustering
    targetZ, Mu, Lscores = runKMeans_BregmanDiv(
        targetX, K, curModel.obsModel,
        W=targetW,
        Niter=Niter,
        seed=seed) 
    # Convert segmentation Z into proper local parameters dict LP
    xtargetLP = convertLPFromHardToSoft(
        dict(Z=targetZ), targetData, initGarbageState=0)
    if isinstance(Dslice, bnpy.data.WordsData):
        if curModel.obsModel.DataAtomType.count('word'):
            xtargetLP = convertLPFromDocsToTokens(xtargetLP, targetData)
    if curLPslice is not None:
        xtargetLP['resp'] *= \
            curLPslice['resp'][chosenRespIDs, ktarget][:,np.newaxis]    
        # Verify that initial xLP resp is a subset of curLP's resp,
        # leaving out only the docs that didnt have enough tokens.
        assert np.all(xtargetLP['resp'].sum(axis=1) <= \
                      curLPslice['resp'][chosenRespIDs, ktarget] + 1e-5)
    # Summarize the local parameters
    if includeAllocSummary:
        if hasattr(curModel.allocModel, 'initLPFromResp'):
            xtargetLP = curModel.allocModel.initLPFromResp(
                targetData, xtargetLP)
        xSS = curModel.get_global_suff_stats(
            targetData, xtargetLP)
    else:
        xSS = curModel.obsModel.get_global_suff_stats(
            targetData, None, xtargetLP)
    # Reorder the components from big to small
    bigtosmall = np.argsort(-1 * xSS.getCountVec())
    xSS.reorderComps(bigtosmall)
    DebugInfo.update(dict(
        targetZ=targetZ,
        targetData=targetData,
        Mu=Mu,
        Lscores=Lscores))
    return xSS, DebugInfo

def runKMeans_BregmanDiv(X, K, obsModel, W=None,
                         Niter=100, seed=0, init='plusplus',
                         smoothFracInit=1.0, smoothFrac=0):
    ''' Run hard clustering algorithm to find K clusters.

    Returns
    -------
    Z : 1D array, size N
    Mu : 2D array, size K x D
    Lscores : 1D array, size Niter
    '''
    chosenZ, Mu, _ = initKMeans_BregmanDiv(
        X, K, obsModel, W=W, seed=seed, smoothFrac=smoothFracInit)
    if Niter == 0:
        Z = -1 * np.ones(X.shape[0])
        Z[chosenZ] = np.arange(K)

        Div = obsModel.calcSmoothedBregDiv(
            X=X, Mu=Mu, W=W, smoothFrac=smoothFrac)
        Ldata = Div.min(axis=1).sum()
        Lprior = obsModel.calcBregDivFromPrior(
            Mu=Mu, smoothFrac=smoothFrac).sum()
        return Z, Mu, [Ldata+Lprior]
    Lscores = list()
    prevN = np.zeros(K)
    for riter in xrange(Niter):
        Div = obsModel.calcSmoothedBregDiv(
            X=X, Mu=Mu, W=W, smoothFrac=smoothFrac)
        Z = np.argmin(Div, axis=1)
        Ldata = Div.min(axis=1).sum()
        Lprior = obsModel.calcBregDivFromPrior(
            Mu=Mu, smoothFrac=smoothFrac).sum()
        Lscore = Ldata + Lprior
        Lscores.append(Lscore)
        N = np.zeros(K)
        for k in xrange(K):
            if W is None:
                W_k = None
                N[k] = np.sum(Z==k)
            else:
                W_k = W[Z==k]
                N[k] = np.sum(W_k)
            if N[k] > 0:
                Mu[k] = obsModel.calcSmoothedMu(X[Z==k], W_k)
            else:
                Mu[k] = obsModel.calcSmoothedMu(X=None)

        # print riter, Lscore
        # if W is None:
        #     print '   ', ' '.join(['%.0f' % (x) for x in N])
        # else:
        #     assert np.allclose(N.sum(), W.sum())
        #     print '   ', ' '.join(['%.2f' % (x) for x in N])
        if np.max(np.abs(N - prevN)) == 0:
            break
        prevN[:] = N
    return Z, Mu, Lscores

def initKMeans_BregmanDiv(
        X, K, obsModel, W=None, seed=0, smoothFrac=1.0):
    ''' Initialize cluster means Mu for K clusters.

    Returns
    -------
    chosenZ : 1D array, size K
        int ids of atoms selected
    Mu : 2D array, size K x D
    minDiv : 1D array, size N
    '''
    PRNG = np.random.RandomState(int(seed))
    N = X.shape[0]
    if W is None:
        W = np.ones(N)
    chosenZ = np.zeros(K, dtype=np.int32)
    chosenZ[0] = PRNG.choice(N, p=W/np.sum(W))

    # Initialize Mu array : K x Mushape
    Mu0 = obsModel.calcSmoothedMu(X[chosenZ[0]], W=W[chosenZ[0]])
    Mu = np.zeros((K,)+Mu0.shape)
    Mu[0] = Mu0
    minDiv = obsModel.calcSmoothedBregDiv(
        X=X, Mu=Mu0, W=W, smoothFrac=smoothFrac)[:,0]
    minDiv[chosenZ[0]] = 0
    for k in range(1, K):
        chosenZ[k] = PRNG.choice(N, p=minDiv/np.sum(minDiv))
        Mu[k] = obsModel.calcSmoothedMu(X[chosenZ[k]], W=W[chosenZ[k]])
        curDiv = obsModel.calcSmoothedBregDiv(
            X=X, Mu=Mu[k], W=W, smoothFrac=smoothFrac)[:,0]
        curDiv[chosenZ[k]] = 0
        minDiv = np.minimum(minDiv, curDiv)
    return chosenZ, Mu, minDiv

def makeDataSubsetByThresholdResp(
        Data, curModel, 
        curLP=None,
        ktarget=None,
        minNumAtomsInEachTargetDoc=100,
        minRespForEachTargetAtom=0.1,
        K=0,
        **kwargs):
    ''' Make subset of provided dataset by thresholding assignments.

    Args
    ----
    Data : bnpy dataset
    curLP : dict of local parameters
    ktarget : integer id of cluster to target, in {0, 1, ... K-1}

    Returns
    -------
    DebugInfo : dict
    targetData : bnpy data object, representing data subset
    targetX : 2D array, size N x K, whose rows will be clustered
    targetW : 1D array, size N
        None indicates uniform weight on all data items
    chosenRespIDs : 1D array, size curLP['resp'].shape[0]
        None indicates no curLP provided.
    '''
    if isinstance(Data, bnpy.data.WordsData):
        Natoms_total = Data.nDoc
        atomType = 'doc'
        if curLP is None:
            weights = None
            Natoms_target = Natoms_total
        else:
            weights = curLP['resp'][:,ktarget]
            if 'DocTopicCount' in curLP:
                DocUsage = curLP['DocTopicCount'][:,ktarget]
                Natoms_target = np.float32((DocUsage > 0.0001).sum())
            else:
                Natoms_target = curLP['resp'][:,ktarget].sum()

        # Make nDoc x vocab_size array 
        X = Data.getSparseDocTypeCountMatrix(weights=weights)
        # Keep only rows with minimum count
        if minNumAtomsInEachTargetDoc is None:
            rowsWithEnoughData = np.arange(X.shape[0])
        else:
            rowsWithEnoughData = np.flatnonzero(
                np.asarray(X.sum(axis=1)) > minNumAtomsInEachTargetDoc)
        Natoms_targetAboveThr = rowsWithEnoughData.size

        targetAssemblyMsg = \
            "  Targeted comp has %.2f docs with mass >eps" % (
                Natoms_target) \
            + " out of %d docs in current dataset." % (
                Natoms_total) \
            + "\n  Filtering to find docs with > %d words assigned." % (
                minNumAtomsInEachTargetDoc) \
            + "\n  Found %d docs meeting this requirement." % (
                Natoms_targetAboveThr)

        # Raise error if target dataset not big enough.
        Keff = np.minimum(K, rowsWithEnoughData.size)
        if Keff <= 1:
            DebugInfo = dict(
                targetAssemblyMsg=targetAssemblyMsg,
                atomType=atomType,
                Natoms_total=Natoms_total,
                Natoms_target=Natoms_target,
                Natoms_targetAboveThr=Natoms_targetAboveThr,
                errorMsg="Dataset too small to cluster." + \
                    " Wanted 2 or more items, found %d." % (Keff))
            return DebugInfo, None, None, None, None
        # Assemble the target dataset
        targetData = Data.select_subset_by_mask(rowsWithEnoughData)
        targetX = targetData.getDocTypeCountMatrix()
        chosenDataIDs = rowsWithEnoughData
        targetW = None
        if curModel.obsModel.DataAtomType.count('doc'):
            chosenRespIDs = np.asarray(
                rowsWithEnoughData, dtype=np.int32)
            if curLP is not None:
                targetW = weights[rowsWithEnoughData]
        elif curModel.obsModel.DataAtomType.count('word'):
            chosenRespIDs = list()
            for d in rowsWithEnoughData:
                start_d = Data.doc_range[d]
                stop_d = Data.doc_range[d+1]
                chosenRespIDs.extend(np.arange(start_d, stop_d))
            chosenRespIDs = np.asarray(chosenRespIDs, dtype=np.int32)

    elif isinstance(Data, bnpy.data.XData) or \
            isinstance(Data, bnpy.data.GroupXData):
        Natoms_total = Data.X.shape[0]
        atomType = 'atoms'
        if curLP is None:
            targetData = Data
            targetX = Data.X
            targetW = None
            chosenRespIDs = None
            Natoms_target = targetX.shape[0]
            Natoms_targetAboveThr = targetX.shape[0]
            targetAssemblyMsg = \
                "  Using all %d/%d atoms for initialization." % (
                    Natoms_target, Natoms_total)

        else:
            chosenRespIDs = np.flatnonzero(
                curLP['resp'][:,ktarget] > 
                minRespForEachTargetAtom)
            Natoms_target = curLP['resp'][:,ktarget].sum()
            Natoms_targetAboveThr = chosenRespIDs.size
            targetAssemblyMsg = \
                "  Targeted comp has %.2f %s assigned out of %d." % (
                    Natoms_target, atomType, Natoms_total) \
                + "\n  Filtering to find atoms with resp > %.2f" % (
                    minRespForEachTargetAtom) \
                + "\n  Found %d atoms meeting this requirement." % (
                    Natoms_targetAboveThr)

            # Raise error if target dataset not big enough.
            Keff = np.minimum(K, chosenRespIDs.size)
            if Keff <= 1:
                DebugInfo = dict(
                    targetAssemblyMsg=targetAssemblyMsg,
                    atomType=atomType,
                    Natoms_total=Natoms_total,
                    Natoms_target=Natoms_target,
                    Natoms_targetAboveThr=Natoms_targetAboveThr,
                    errorMsg="Filtered dataset too small." + \
                        "Wanted %d items, found %d." % (K, Keff))
                return DebugInfo, None, None, None, None
            targetData = Data.select_subset_by_mask(chosenRespIDs)
            targetX = targetData.X
            targetW = curLPslice['resp'][chosenRespIDs,ktarget]

    DebugInfo = dict(
        targetAssemblyMsg=targetAssemblyMsg,
        atomType=atomType,
        Natoms_total=Natoms_total,
        Natoms_target=Natoms_target,
        Natoms_targetAboveThr=Natoms_targetAboveThr,
        )
    return DebugInfo, targetData, targetX, targetW, chosenRespIDs
