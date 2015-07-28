import numpy as np

from SAssignToExisting import assignSplitStats, _calc_expansion_alphaEbeta
from scipy.special import digamma, gammaln
from BirthProposalError import BirthProposalError

try:
    import KMeansRex
    def RunKMeans(X, K, seed):
        return KMeansRex.RunKMeans(
            X, K, initname='plusplus', seed=seed, Niter=500)
except ImportError:
    from scipy.cluster.vq import kmeans2
    def RunKMeans(X, K, seed):
        np.random.seed(seed)
        Mu, Z = kmeans2(X, K, minit='points')
        return Mu, Z

def createSplitStats(
        Dslice, hmodel, curLPslice, curSSwhole=None,
        creationProposalName='truelabels',
        **kwargs):
    ''' Reassign target component to new states.

    Returns
    -------
    xSSslice : stats for reassigned mass
        total count is equal to SS.N[ktarget]
        number of components is Kx
    '''
    createSplitStatsMap = dict([
        (k,v) for (k,v) in globals().items()
        if str(k).count('createSplitStats')])
    aName = hmodel.getAllocModelName()
    funcName = 'createSplitStats_' + aName + '_' + creationProposalName
    if funcName not in createSplitStatsMap:
        raise NotImplementedError('Unrecognized function: ' + funcName)
    
    createSplitStatsFunc = createSplitStatsMap[funcName]
    xSSslice = createSplitStatsFunc(
        Dslice, hmodel, curLPslice, curSSwhole=curSSwhole,
        **kwargs)
    return xSSslice



def createSplitStats_DPMixtureModel_kmeans(
        Dslice, hmodel, curLPslice, curSSwhole=None,
        targetUID=0, LPkwargs=dict(),
        newUIDs=None,
        lapFrac=0,
        **kwargs):
    ''' Reassign target component to new states, via kmeans.

    Returns
    -------
    xSSslice : stats for reassigned mass
        total count is equal to SS.N[ktarget]
        number of components is Kx
    '''
    xK = newUIDs.size
    ktarget = curSSwhole.uid2k(targetUID)
    Dtarget = Dslice.select_subset_by_mask(
        np.flatnonzero(curLPslice['resp'][:, ktarget] > 0.05))
    Xtarget = Dtarget.X
    Mu, Z = RunKMeans(Xtarget, xK, Niter=25, seed=lapFrac)
    Z = Z.flatten()
    resp = np.zeros((Xtarget.shape[0], xK))
    Kused = 0
    for k in range(xK):
        mask_k = Z == k
        Nk = np.sum(mask_k)
        if Nk > 5:
            resp[mask_k, Kused] = 1.0
            Kused += 1
    resp = resp[:, :Kused]
    
    LPtarget = dict(resp=resp)
    SSfake = hmodel.get_global_suff_stats(Dtarget, LPtarget)
    SSfake.setUIDs(newUIDs[:Kused])

    xSSslice = assignSplitStats(
        Dslice, hmodel, curLPslice, curSSwhole, SSfake, targetUID=targetUID)
    return xSSslice


def createSplitStats_DPMixtureModel_truelabels(
        Dslice, hmodel, curLPslice, curSSwhole=None,
        targetUID=0, LPkwargs=dict(),
        newUIDs=None,
        lapFrac=0,
        **kwargs):
    ''' Reassign target component to new states, based on true labels.

    Returns
    -------
    xSSslice : stats for reassigned mass
        total count is equal to SS.N[ktarget]
        number of components is Kx
    '''
    ktarget = curSSwhole.uid2k(targetUID)

    uLabels = np.unique(Dslice.TrueParams['Z'])
    Ktrue = uLabels.size
    trueResp = np.zeros((Dslice.nObs, Ktrue))
    for k in range(Ktrue):
        trueResp[Dslice.TrueParams['Z'] == k, k] = 1.0
    scaledResp = trueResp
    scaledResp /= curLPslice['resp'][:, ktarget][:, np.newaxis]
    np.maximum(scaledResp, 1e-100, out=scaledResp)

    xLPslice = dict(resp=scaledResp)
    xSSslice = hmodel.get_global_suff_stats(
        Dslice, xLPslice, doPrecompEntropy=1)
    xSSslice.setUIDs(newUIDs[:Ktrue])
    return xSSslice



DefaultLPkwargs = dict(
    restartLP=1,
    convThrLP=0.001,
    nCoordAscentItersLP=50,
    )

def createSplitStats_HDPTopicModel_truelabels(
        Dslice, curModel, curLPslice, curSSwhole=None,
        targetUID=0, LPkwargs=DefaultLPkwargs,
        newUIDs=None,
        lapFrac=0,
        returnPropSS=0,
        **kwargs):
    ''' Reassign target component to new states, based on true labels.

    Returns
    -------
    xSSslice : stats for reassigned mass
        total count is equal to SS.N[ktarget]
        number of components is Kx
    '''
    ktarget = curSSwhole.uid2k(targetUID)

    trueResp = Dslice.TrueParams['resp'].copy()
    Ktrue = trueResp.shape[1]
    Korig = curSSwhole.K

    scaledResp = trueResp
    scaledResp *= curLPslice['resp'][:, ktarget][:, np.newaxis]
    np.maximum(scaledResp, 1e-100, out=scaledResp)
    assert np.allclose(scaledResp.sum(axis=1),
                       curLPslice['resp'][:, ktarget])
    xLPslice = dict(resp=scaledResp)

    xalphaEbeta = _calc_expansion_alphaEbeta(curModel, ktarget, Ktrue)
    thetaEmptyComp = xalphaEbeta[0] * 1.0
    thetaRem = curLPslice['thetaRem'] * 1.0

    xDocTopicCount = np.zeros((Dslice.nDoc, Ktrue))
    for d in xrange(Dslice.nDoc):
        start = Dslice.doc_range[d]
        stop = Dslice.doc_range[d+1]
        xDocTopicCount[d,:] = np.sum(
            scaledResp[start:stop] * \
            Dslice.word_count[start:stop][:,np.newaxis], axis=0)
    xtheta = xDocTopicCount + xalphaEbeta

    digammaSumTheta = curLPslice['digammaSumTheta'].copy()
    xElogPi = digamma(xtheta) - digammaSumTheta[:, np.newaxis]
    ElogPiRem = digamma(thetaRem) - digammaSumTheta
    ElogPiEmptyComp = digamma(thetaEmptyComp) - digammaSumTheta

    xLPslice['digammaSumTheta'] = digammaSumTheta
    xLPslice['DocTopicCount'] = xDocTopicCount
    xLPslice['theta'] = xtheta
    xLPslice['thetaRem'] = thetaRem
    xLPslice['ElogPi'] = xElogPi
    xLPslice['ElogPiRem'] = ElogPiRem
    xLPslice['thetaEmptyComp'] = thetaEmptyComp
    xLPslice['ElogPiEmptyComp'] = ElogPiEmptyComp
    xLPslice['ElogPiOrigComp'] = curLPslice['ElogPi'][:, ktarget]
    xLPslice['gammalnThetaOrigComp'] = np.sum(
        gammaln(curLPslice['theta'][:, ktarget]))
    slack = curLPslice['DocTopicCount'][:, ktarget] - \
            curLPslice['theta'][:, ktarget]
    xLPslice['slackThetaOrigComp'] = np.sum(
        slack * curLPslice['ElogPi'][:, ktarget])

    xSSslice = curModel.get_global_suff_stats(
        Dslice, xLPslice, doPrecompEntropy=1, doTrackTruncationGrowth=1)
    xSSslice.setUIDs(newUIDs[:Ktrue])

    if returnPropSS:
        from SAssignToExisting import \
             _verify_HDPTopicModel_and_return_xSSslice_and_propSSslice
        return _verify_HDPTopicModel_and_return_xSSslice_and_propSSslice(
            Dslice, curModel, curLPslice, xLPslice, xSSslice,
            ktarget=ktarget, **kwargs)
    return xSSslice



def createSplitStats_HDPTopicModel_kmeans(
        Dslice, curModel, curLPslice, curSSwhole=None,
        targetUID=0, LPkwargs=DefaultLPkwargs,
        newUIDs=None,
        lapFrac=0,
        returnPropSS=0,
        **kwargs):
    ''' Reassign target component to new states, based on true labels.

    Returns
    -------
    xSSslice : stats for reassigned mass
        total count is equal to SS.N[ktarget]
        number of components is Kx
    '''
    ktarget = curSSwhole.uid2k(targetUID)
    xK = newUIDs.size

    # First, cluster the documents that prominently use the target topic
    doc_mask = np.flatnonzero(curLPslice['DocTopicCount'][:, ktarget] > 1)
    Dtarget = Dslice.select_subset_by_mask(doc_mask)
    Xtarget = Dtarget.getDocTypeCountMatrix()
    Mu, Z = RunKMeans(Xtarget, xK, seed=lapFrac*1000)
    Z = Z.flatten()

    # Make a token-specific resp that assigns all words from each doc
    # to the given topic
    xresp = np.zeros((Dslice.word_id.size, xK))
    Kused = 0
    for k in range(xK):
        docs_k = np.flatnonzero(Z == k)
        Nk = docs_k.size
        for d in docs_k:
            start = Dslice.doc_range[d]
            stop = Dslice.doc_range[d+1]
            xresp[start:stop, Kused] = 1.0
        Kused += 1
    xresp = xresp[:, :Kused]

    xSSfake = curModel.obsModel.get_global_suff_stats(
        Dslice, None, dict(resp=xresp))
    bigtosmall = np.argsort(-1 * xSSfake.getCountVec())
    xSSfake.reorderComps(bigtosmall)
    xSSfake.setUIDs(newUIDs[:Kused])

    return assignSplitStats(
        Dslice, curModel, curLPslice, curSSwhole, xSSfake,
        targetUID=targetUID,
        LPkwargs=LPkwargs,
        returnPropSS=returnPropSS,
        lapFrac=lapFrac,
        **kwargs)

def createSplitStats_HDPTopicModel_MultBregDiv(
        Dslice, curModel, curLPslice, curSSwhole=None,
        targetUID=0, LPkwargs=DefaultLPkwargs,
        newUIDs=None,
        lapFrac=0,
        includeRemainderTopic=1,
        returnPropSS=0,
        **kwargs):
    ''' Reassign target component to new states using bregman divergence.

    Returns
    -------
    xSSslice : stats for reassigned mass
        total count is equal to SS.N[ktarget]
        number of components is Kx
    DebugInfo : dict with info for visualization, etc
    '''
    ktarget = curSSwhole.uid2k(targetUID)
    xK = newUIDs.size
    # Select (xK-1) documents by Bregman divergence
    #  assigning remainder to "big topic"
    chosenDocIDs, Info = sampleDocIDsByBregDiv_Mult(
        Dslice, curModel, curLPslice,
        K=xK,
        ktarget=ktarget,
        lapFrac=lapFrac,
        **kwargs)
    if includeRemainderTopic:
        chosenDocIDs = chosenDocIDs[:-1]
        Kused = 1
    else:
        Kused = 0
    # Make fake suff stats
    xresp = np.zeros((Dslice.word_id.size, xK))
    for d in chosenDocIDs:
        start = Dslice.doc_range[d]
        stop = Dslice.doc_range[d+1]
        xresp[start:stop, Kused] = curLPslice['resp'][start:stop, ktarget]
        Kused += 1
    xresp = xresp[:, :Kused]
    if includeRemainderTopic:
        emptyIDs = xresp.sum(axis=1) < 1e-50
        xresp[emptyIDs, 0] = curLPslice['resp'][emptyIDs, ktarget]
    xSSfake = curModel.obsModel.get_global_suff_stats(
        Dslice, None, dict(resp=xresp))
    # Reorder the components from big to small
    bigtosmall = np.argsort(-1 * xSSfake.getCountVec())
    xSSfake.reorderComps(bigtosmall)
    xSSfake.setUIDs(newUIDs[:Kused])

    DebugInfo = dict(
        chosenDocIDs=chosenDocIDs,
        xSSfake=xSSfake,
        )
    xSSslice = xSSfake
    for i in range(5):
        xSSslice = assignSplitStats(
            Dslice, curModel, curLPslice, curSSwhole, xSSslice,
            targetUID=targetUID,
            LPkwargs=LPkwargs,
            returnPropSS=returnPropSS,
            lapFrac=lapFrac,
            **kwargs)
        print xSSslice.getCountVec()
    nnzCount = np.sum(xSSslice.getCountVec() >= 1 - 1e-4)
    if nnzCount < 2:
        raise BirthProposalError(
            "Could not create at least two comps with mass >= 1.")
    if returnPropSS:
        return xSSslice[0], xSSslice[1]
    return xSSslice, DebugInfo

def sampleDocIDsByBregDiv_Mult(
        Dslice, curModel=None, curLPslice=None,
        K=5, ktarget=None, minCountThr=10, 
        lapFrac=0, doSample=True,
        **kwargs):
    ''' Sample document ids by Bregman divergence.

    Returns
    -------
    chosenDocIDs : list of xK document IDs
    '''
    PRNG = np.random.RandomState(1000 * lapFrac)
    DocWordMat = Dslice.getDocTypeCountMatrix(
        weights=curLPslice['resp'][:,ktarget])
    # Keep only rows with minimum count
    rowsWithEnoughData = np.flatnonzero(
        DocWordMat.sum(axis=1) > minCountThr)
    DocWordMat = DocWordMat[rowsWithEnoughData]
    Keff = np.minimum(K, DocWordMat.shape[0])
    if Keff < 1:
        raise BirthProposalError(
            "Not enough data. Looked for %d documents, found only %d" % (
                K, Keff))
    K = Keff
    np.maximum(DocWordMat, 1e-100, out=DocWordMat)

    WholeDataMean = calcClusterMean_Mult(
        DocWordMat, hmodel=curModel)[np.newaxis, :]
    minDiv = calcBregDiv_Mult(DocWordMat, WholeDataMean)
    assert minDiv.min() > -1e-10
    WCMeans = np.zeros((K, WholeDataMean.shape[1]))
    lamVals = np.zeros(K+1)
    chosenDocIDs = np.zeros(K)
    for k in range(K):
        # Find data point with largest minDiv value
        if doSample:
            p = minDiv[:,0] / np.sum(minDiv)
            n = PRNG.choice(minDiv.size, p=p)
        else:
            n = minDiv.argmax()
        chosenDocIDs[k] = rowsWithEnoughData[n]
        lamVals[k] = minDiv[n]
        # Add this point to the clusters
        WCMeans[k,:] = calcClusterMean_Mult(
            DocWordMat[n], hmodel=curModel)
        # Recalculate minimum distance to existing means
        curDiv = calcBregDiv_Mult(DocWordMat, WCMeans[k])
        np.minimum(curDiv, minDiv, out=minDiv)
        minDiv[n] = 0
        assert minDiv.min() > -1e-10
    lamVals[-1] = minDiv.max()
    return chosenDocIDs, dict(Means=WCMeans, lamVals=lamVals)


def calcClusterMean_Mult(WordCountData, lam=0.05, hmodel=None):
    if hmodel is not None:
        lam = hmodel.obsModel.Prior.lam
    if WordCountData.ndim == 1:
        WordCountData = WordCountData[np.newaxis,:]
    WordCountSumVec = np.sum(WordCountData, axis=0)
    ClusterMean = WordCountSumVec + lam
    return ClusterMean

def calcBregDiv_Mult(WordCountData, WordCountMeans):
    ''' Calculate Bregman divergence between rows of two matrices.

    Args
    ----
    WordCountData : 2D array, N x vocab_size
    WordCountMeans : 2D array, K x vocab_size

    Returns
    -------
    Div : 2D array, N x K
    '''
    if WordCountData.ndim == 1:
        WordCountData = WordCountData[np.newaxis,:]
    if WordCountMeans.ndim == 1:
        WordCountMeans = WordCountMeans[np.newaxis,:]
    assert WordCountData.shape[1] == WordCountMeans.shape[1]
    N = WordCountData.shape[0]
    K = WordCountMeans.shape[0]
    Nx = WordCountData.sum(axis=1)
    assert np.all(Nx >= 1.0 - 1e-10)
    Nmean = WordCountMeans.sum(axis=1)
    assert np.all(Nmean >= 1.0 - 1e-10)
    Div = np.zeros((N, K))
    for k in xrange(K):
        Div[:, k] = np.sum(WordCountData * np.log(WordCountData / WordCountMeans[k,:][np.newaxis,:]), axis=1)
        Div[:, k] += Nx * np.log(Nmean[k]/Nx)
    return Div