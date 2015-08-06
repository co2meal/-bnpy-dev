import numpy as np
import os
import sys
import bnpy.init.FromTruth

from SCleanup import cleanupMergeClusters, cleanupDeleteSmallClusters
from SAssignToExisting import assignSplitStats, _calc_expansion_alphaEbeta
from scipy.special import digamma, gammaln
from BirthProposalError import BirthProposalError
from bnpy.viz.PlotComps import plotAndSaveCompsFromSS
from bnpy.viz.ProposalViz import plotELBOtermsForProposal
from bnpy.viz.ProposalViz import plotDocUsageForProposal
from bnpy.viz.ProposalViz import makeSingleProposalHTMLStr

try:
    import KMeansRex
    def RunKMeans(X, K, seed=0):
        return KMeansRex.RunKMeans(
            X, K, initname='plusplus', seed=seed, Niter=500)
except ImportError:
    from scipy.cluster.vq import kmeans2
    def RunKMeans(X, K, seed=0):
        np.random.seed(seed)
        Mu, Z = kmeans2(X, K, minit='points')
        return Mu, Z

def createSplitStats(
        Dslice, hmodel, curLPslice, curSSwhole=None,
        b_creationProposalName='truelabels',
        **kwargs):
    ''' Reassign target component to new states.

    Returns
    -------
    xSSslice : SuffStatBag
        Contains exact summaries for reassignment of target mass.
        * Total mass is equal to mass assigned to ktarget in curLPslice
        * Number of components is Kx
    Info : dict
        Contains info for detailed debugging of construction process.
    '''
    if 'b_debugOutputDir' in kwargs and kwargs['b_debugOutputDir'] == 'None':
        kwargs['b_debugOutputDir'] = None

    createSplitStatsMap = dict([
        (k,v) for (k,v) in globals().items()
        if str(k).count('createSplitStats')])
    aName = hmodel.getAllocModelName()
    funcName = 'createSplitStats_' + aName + '_' + b_creationProposalName
    if funcName not in createSplitStatsMap:
        raise NotImplementedError('Unrecognized function: ' + funcName)
    
    # Execute model-specific function to make expansion stats
    # This call may return early if expansion failed,
    # due to creating too few states that are big-enough.
    # Need to finalize debug html before raising error.
    createSplitStatsFunc = createSplitStatsMap[funcName]
    xSSslice, DebugInfo = createSplitStatsFunc(
        Dslice, hmodel, curLPslice, curSSwhole=curSSwhole,
        **kwargs)
    # Write final debug HTML page, if specified.
    if 'b_debugOutputDir' in kwargs and kwargs['b_debugOutputDir']:
        htmlstr = makeSingleProposalHTMLStr(DebugInfo, **kwargs)
        htmlfilepath = os.path.join(kwargs['b_debugOutputDir'], 'index.html')
        with open(htmlfilepath, 'w') as f:
            f.write(htmlstr)
        print "DEBUGGER WROTE: ", htmlfilepath
    # Raise error if we didn't create enough "big-enough" states.
    nnzCount = np.sum(xSSslice.getCountVec() >= 1)
    if nnzCount < 2:
        raise BirthProposalError(
            "Could not create at least two comps with mass >= 1.")
    # If here, we have a valid proposal. 
    # Need to verify mass conservation
    ktarget = curSSwhole.uid2k(kwargs['targetUID'])
    if hasattr(Dslice, 'word_count'):
        origMass = np.inner(Dslice.word_count, curLPslice['resp'][:,ktarget])
    else:
        origMass = curLPslice['resp'][:,ktarget].sum()
    newMass = xSSslice.getCountVec().sum()
    assert np.allclose(newMass, origMass, atol=1e-6, rtol=0)
    return xSSslice, DebugInfo



def createSplitStats_DPMixtureModel_kmeans(
        Dslice, hmodel, curLPslice, 
        curSSwhole=None,
        targetUID=0, 
        ktarget=None,
        LPkwargs=dict(),
        newUIDs=None,
        lapFrac=0,
        b_minSize=2,
        **kwargs):
    ''' Reassign target component to new states, via kmeans.

    Returns
    -------
    xSSslice : stats for reassigned mass
        total count is equal to SS.N[ktarget]
        number of components is Kx
    '''
    if ktarget is None:
        ktarget = curSSwhole.uid2k(targetUID)

    xK = newUIDs.size
    keepIDs = np.flatnonzero(curLPslice['resp'][:, ktarget] > 0.05)
    if len(keepIDs) < xK:
        raise BirthProposalError(
            "Not enough data. Looked for %d atoms, found only %d" % (
                xK, len(keepIDs)))

    Dtarget = Dslice.select_subset_by_mask(keepIDs)
    # Run Kmeans on subset of data.
    Xtarget = Dtarget.X
    Mu, Z = RunKMeans(Xtarget, xK, seed=lapFrac)
    Z = Z.flatten()
    # Create soft assignment matrix, keeping only big-enough clusters
    resp = np.zeros((Xtarget.shape[0], xK))
    Kused = 0
    for k in range(xK):
        mask_k = Z == k
        Nk = np.sum(mask_k)
        if Nk >= b_minSize:
            resp[mask_k, Kused] = 1.0
            Kused += 1
    if Kused < 2:
        raise BirthProposalError(
            "Init clusters not big enough. Only <=1 with size >%d." % (
                b_minSize))

    resp = resp[:, :Kused] * curLPslice['resp'][keepIDs, ktarget][:,np.newaxis]
    xLPfake = dict(resp=resp)
    xSSfake = hmodel.get_global_suff_stats(Dtarget, xLPfake)
    xSSfake.setUIDs(newUIDs[:Kused])

    xSSslice = assignSplitStats(
        Dslice, hmodel, curLPslice, xSSfake,
        curSSwhole=curSSwhole,
        targetUID=targetUID)

    DebugInfo = dict(
        Z=Z,
        Mu=Mu)
    return xSSslice, DebugInfo


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

def createSplitStats_HDPTopicModel_BregDiv(
        Dslice, curModel, curLPslice, 
        curSSwhole=None,
        targetUID=0,
        ktarget=None,
        LPkwargs=DefaultLPkwargs,
        newUIDs=None,
        lapFrac=0,
        b_includeRemainderTopic=1,
        b_nRefineSteps=3,
        b_minNumAtomsInDoc=100,
        b_debugOutputDir=None,
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
    if curSSwhole is None:
        curSSwhole = curModel.get_global_suff_stats(Dslice, curLPslice)
    if ktarget is None:
        ktarget = curSSwhole.uid2k(targetUID)

    xK = newUIDs.size
    xSSfake, DebugInfo = initSSByBregDiv_Mult(
        Dslice, curModel, curLPslice,
        K=xK,
        ktarget=ktarget,
        lapFrac=lapFrac,
        b_minNumAtomsInDoc=b_minNumAtomsInDoc,
        b_includeRemainderTopic=b_includeRemainderTopic,
        **kwargs)
    xSSfake.setUIDs(newUIDs[:xSSfake.K])

    if b_debugOutputDir:
        plotAndSaveCompsFromSS(
            curModel, curSSwhole, b_debugOutputDir, 'OrigComps.png',
            vocabList=Dslice.vocabList,
            compsToHighlight=[ktarget])
        plotAndSaveCompsFromSS(
            curModel, xSSfake, b_debugOutputDir, 'NewComps_Init.png',
            vocabList=Dslice.vocabList)
        curLdict = curModel.calc_evidence(SS=curSSwhole, todict=1)
        propLdictList = list()
        docUsageByUID = dict()
        for k, uid in enumerate(xSSfake.uids):
            if k == 0 and b_includeRemainderTopic:
                docUsage_ktarget = np.sum( 
                    curLPslice['DocTopicCount'][:, ktarget] > 0.01)
                docUsage_rest = docUsage_ktarget - xSSfake.K + \
                    b_includeRemainderTopic
                docUsageByUID[uid] = [docUsage_rest]
            else:
                docUsageByUID[uid] = [1]

    xSSslice = xSSfake
    nnzCount = np.sum(xSSslice.getCountVec() >= 1)
    for i in range(b_nRefineSteps):
        # Obtain valid suff stats that represent Dslice for given model
        # using xSSslice as initial "seed" clusters.
        # Note: xSSslice need only have observation-model stats here.
        xSSslice = assignSplitStats(
            Dslice, curModel, curLPslice, xSSslice,
            curSSwhole=curSSwhole,
            targetUID=targetUID,
            ktarget=ktarget,
            LPkwargs=LPkwargs,
            returnPropSS=returnPropSS,
            lapFrac=lapFrac,
            **kwargs)

        # Show diagnostics for new states
        CountVec = xSSslice.getCountVec()
        print ' '.join(['%.0f' % (x) for x in CountVec])
        if b_debugOutputDir:
            plotAndSaveCompsFromSS(
                curModel, xSSslice, b_debugOutputDir,
                filename='NewComps_Step%d.png' % (i+1),
                vocabList=Dslice.vocabList)
            propSS = curSSwhole.copy()
            propSS.transferMassFromExistingToExpansion(
                uid=targetUID, xSS=xSSslice)
            propModel = curModel.copy()
            propModel.update_global_params(propSS)
            propLdict = propModel.calc_evidence(SS=propSS, todict=1)
            propLdictList.append(propLdict)

            docUsageVec = xSSslice.getSelectionTerm('DocUsageCount')
            for k, uid in enumerate(xSSslice.uids):
                docUsageByUID[uid].append(docUsageVec[k])

        # Cleanup by deleting small clusters 
        if i < b_nRefineSteps - 1:
            if i == b_nRefineSteps - 2:
                # After all but last step, delete small (but not empty) comps
                minNumAtomsToStay = b_minNumAtomsInDoc
            else:
                # Always remove empty clusters. They waste our time.
                minNumAtomsToStay = 1
            xSSslice = cleanupDeleteSmallClusters(xSSslice, minNumAtomsToStay)
        # Cleanup by merging clusters
        if i == b_nRefineSteps - 2:
            DebugInfo['mergestep'] = i + 1
            xSSslice = cleanupMergeClusters(
                xSSslice, curModel,
                obsSS=xSSfake,
                vocabList=Dslice.vocabList,
                b_debugOutputDir=b_debugOutputDir, **kwargs)
        # Exit early if no promising new clusters are created
        nnzCount = np.sum(xSSslice.getCountVec() >= 1)
        if nnzCount < 2:
            break

    if b_debugOutputDir:
        savefilename = os.path.join(
            b_debugOutputDir, 'ProposalTrace_ELBO.png')
        plotELBOtermsForProposal(curLdict, propLdictList,
                                 savefilename=savefilename)
        savefilename = os.path.join(
            b_debugOutputDir, 'ProposalTrace_DocUsage.png')
        plotDocUsageForProposal(docUsageByUID,
                                savefilename=savefilename)
        GainELBO = propLdictList[-1]['Ltotal'] - curLdict['Ltotal']
        if np.sum(xSSslice.getCountVec() > 1) < 2:
            DebugInfo['status'] = \
                'Rejected. Did not create >1 new comps with significant mass'
        elif GainELBO > 0:
            DebugInfo['status'] = \
                'Accepted. ELBO improved by %.3f' % (GainELBO)
        else:
            DebugInfo['status'] = 'Rejected. ELBO did not improve.'

    if returnPropSS:
        return xSSslice[0], xSSslice[1]
    return xSSslice, DebugInfo

def initSSByBregDiv_Mult(
        Dslice, curModel=None, curLPslice=None,
        K=5, ktarget=None, 
        b_minNumAtomsInDoc=10, 
        b_includeRemainderTopic=0,
        b_initHardCluster=0,
        lapFrac=0, doSample=True,
        **kwargs):
    ''' Create observation model statistics via Breg. distance sampling.

    Returns
    -------
    xSS : SuffStatBag
    Info : dict
        contains info about which docs were used to inspire this init. 
    '''
    PRNG = np.random.RandomState(1000 * lapFrac)
    DocWordMat = Dslice.getDocTypeCountMatrix(
        weights=curLPslice['resp'][:,ktarget])
    # Keep only rows with minimum count
    rowsWithEnoughData = np.flatnonzero(
        DocWordMat.sum(axis=1) > b_minNumAtomsInDoc)
    enoughDocWordMat = DocWordMat[rowsWithEnoughData]
    Keff = np.minimum(K, enoughDocWordMat.shape[0])
    if Keff < 1:
        raise BirthProposalError(
            "Not enough data. Looked for %d documents, found only %d" % (
                K, Keff))
    K = Keff
    np.maximum(enoughDocWordMat, 1e-100, out=enoughDocWordMat)

    WholeDataMean = calcClusterMean_Mult(
        enoughDocWordMat, hmodel=curModel)[np.newaxis, :]
    minDiv = calcBregDiv_Mult(enoughDocWordMat, WholeDataMean)
    assert minDiv.min() > -1e-10
    WCMeans = np.zeros((K, WholeDataMean.shape[1]))
    lamVals = np.zeros(K+1)
    chosenDocIDs = np.zeros(K, dtype=np.int32)
    for k in range(K):
        # Find data point with largest minDiv value
        if doSample:
            pvec = minDiv[:,0] / np.sum(minDiv)
            n = PRNG.choice(minDiv.size, p=pvec)
        else:
            n = minDiv.argmax()
        chosenDocIDs[k] = rowsWithEnoughData[n]
        lamVals[k] = minDiv[n]
        # Add this point to the clusters
        WCMeans[k,:] = calcClusterMean_Mult(
            enoughDocWordMat[n], hmodel=curModel)
        # Recalculate minimum distance to existing means
        curDiv = calcBregDiv_Mult(enoughDocWordMat, WCMeans[k])
        np.minimum(curDiv, minDiv, out=minDiv)
        minDiv[n] = 0
        assert minDiv.min() > -1e-10
    lamVals[-1] = minDiv.max()

    if b_includeRemainderTopic == 1:
        chosenDocIDs = chosenDocIDs[:-1]
        WCMeans = np.vstack([WholeDataMean, WCMeans[:-1]])
        WCMeans[0] -= DocWordMat[chosenDocIDs].sum(axis=0)

    Z = -1 * np.ones(Dslice.nDoc, dtype=np.int32)
    if b_initHardCluster:
        DivMat = calcBregDiv_Mult(enoughDocWordMat, WCMeans)
        Z[rowsWithEnoughData] = DivMat.argmin(axis=1)
    else:
        if b_includeRemainderTopic:
            Z[chosenDocIDs] = 1 + np.arange(len(chosenDocIDs))
            Z[Z<1] = 0 # all other docs to rem cluster
        else:
            Z[chosenDocIDs] = np.arange(len(chosenDocIDs))

    docLP = bnpy.init.FromTruth.convertLPFromHardToSoft(
        dict(Z=Z), Dslice, initGarbageState=0)
    xtokenLP = bnpy.init.FromTruth.convertLPFromDocsToTokens(docLP, Dslice)
    xtokenLP['resp'] *= curLPslice['resp'][:, ktarget][:,np.newaxis]    

    # Verify that initial xLP resp is a subset of curLP's resp,
    # leaving out only the docs that didnt have enough tokens.
    assert np.all(xtokenLP['resp'].sum(axis=1) <= \
                  curLPslice['resp'][:, ktarget] + 1e-5)
    xSS = curModel.obsModel.get_global_suff_stats(Dslice, None, xtokenLP)

    # Reorder the components from big to small
    bigtosmall = np.argsort(-1 * xSS.getCountVec())
    xSS.reorderComps(bigtosmall)

    Info = dict(
        Z=Z,
        Means=WCMeans, 
        lamVals=lamVals, 
        chosenDocIDs=chosenDocIDs)
    return xSS, Info

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
        Div[:, k] = np.sum(WordCountData * np.log(
            WordCountData / WordCountMeans[k,:][np.newaxis,:]), axis=1)
        Div[:, k] += Nx * np.log(Nmean[k]/Nx)
    return Div


"""
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
"""
