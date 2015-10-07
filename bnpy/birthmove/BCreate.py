import numpy as np
import os
import sys
import bnpy.init.FromTruth
import BLogger

from scipy.special import digamma, gammaln
from BCleanup import cleanupMergeClusters, cleanupDeleteSmallClusters
from BRefine import assignSplitStats, _calc_expansion_alphaEbeta
from BirthProposalError import BirthProposalError
from bnpy.viz.PlotComps import plotCompsFromSS
from bnpy.viz.ProposalViz import plotELBOtermsForProposal
from bnpy.viz.ProposalViz import plotDocUsageForProposal
from bnpy.viz.ProposalViz import makeSingleProposalHTMLStr
from bnpy.viz.PrintTopics import vec2str

from bnpy.init.FromScratchBregman import initSS_BregmanDiv

DefaultLPkwargs = dict(
    restartLP=1,
    convThrLP=0.001,
    nCoordAscentItersLP=50,
    )
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
        b_creationProposalName='bregmankmeans',
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
    if 'b_debugOutputDir' not in kwargs:
        kwargs['b_debugOutputDir'] = None
    if 'b_debugOutputDir' in kwargs and kwargs['b_debugOutputDir'] == 'None':
        kwargs['b_debugOutputDir'] = None
    b_debugOutputDir = kwargs['b_debugOutputDir']

    createSplitStatsMap = dict([
        (k,v) for (k,v) in globals().items()
        if str(k).count('createSplitStats')])
    funcName = 'createSplitStats' + '_' + b_creationProposalName
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

    return xSSslice, DebugInfo



def createSplitStats_bregmankmeans(
        Dslice, curModel, curLPslice, 
        curSSwhole=None,
        targetUID=None,
        ktarget=None,
        LPkwargs=DefaultLPkwargs,
        newUIDs=None,
        lapFrac=0,
        b_nRefineSteps=3,
        b_debugOutputDir=None,
        b_minNumAtomsForNewComp=None,
        returnPropSS=0,
        vocabList=None,
        **kwargs):
    ''' Reassign target cluster to new states using Bregman K-means.

    Returns
    -------
    xSSslice : stats for reassigned mass
        total count is equal to SS.getCountVec()[ktarget]
        number of components is Kx
    DebugInfo : dict with info for visualization, etc
    '''
    BLogger.pprint('Creating proposal for targetUID %s at lap %.2f' % (
        targetUID, lapFrac))
    # Parse some kwarg input
    if hasattr(Dslice, 'vocabList') and Dslice.vocabList is not None:
        vocabList = Dslice.vocabList
    if ktarget is None:
        ktarget = curSSwhole.uid2k(targetUID)
    # Debug mode: make plot of the current model's components
    if b_debugOutputDir:
        plotCompsFromSS(
            curModel, curSSwhole, 
            os.path.join(b_debugOutputDir, 'OrigComps.png'),
            vocabList=vocabList,
            compsToHighlight=[ktarget])
    # Determine exactly how many states we can make...
    xK = newUIDs.size
    if xK + curSSwhole.K > kwargs['Kmax']:
        xK = kwargs['Kmax'] - curSSwhole.K
        newUIDs = newUIDs[:xK]
        if xK <= 1:
            errorMsg = 'Cancelled.' + \
                'Adding 2 or more states would exceed budget of %d comps.' % (
                    kwargs['Kmax'])
            BLogger.pprint(errorMsg)
            return None, dict(errorMsg=errorMsg)

    # Create suff stats for some new states
    xSSfake, DebugInfo = initSS_BregmanDiv(
        Dslice, curModel, curLPslice, 
        K=xK, 
        ktarget=ktarget,
        lapFrac=lapFrac,
        seed=1000 * lapFrac,
        **kwargs)
    BLogger.pprint(DebugInfo['targetAssemblyMsg'])
    # EXIT: if proposal initialization fails, quit early
    if xSSfake is None:
        BLogger.pprint('Proposal Init Failed. ' + DebugInfo['errorMsg'])
        return None, DebugInfo
    # Describe the initialization.
    BLogger.pprint('  Initialized Bregman clustering with %d clusters %s.' % (
        xSSfake.K, '(--b_Kfresh=%d)' % kwargs['b_Kfresh']))
    BLogger.pprint('  Running %d refinement iterations (--b_nRefineSteps)' % (
        b_nRefineSteps))
    # Record some debug info about the new states
    xSSfake.setUIDs(newUIDs[:xSSfake.K])
    strUIDs = vec2str(xSSfake.uids)
    BLogger.pprint('   ' + strUIDs)
    if b_debugOutputDir:
        plotCompsFromSS(
            curModel, xSSfake, 
            os.path.join(b_debugOutputDir, 'NewComps_Init.png'),
            vocabList=vocabList)
        curLdict = curModel.calc_evidence(SS=curSSwhole, todict=1)
        propLdictList = list()

        docUsageByUID = dict()
        if curModel.getAllocModelName().count('HDP'):
            for k, uid in enumerate(xSSfake.uids):
                if 'targetZ' in DebugInfo:
                    initDocUsage_uid = np.sum(DebugInfo['targetZ'] == k)
                else:
                    initDocUsage_uid = 0.0
                docUsageByUID[uid] = [initDocUsage_uid]

    xSSslice = xSSfake
    # Make a function to help with logging
    pprintCountVec = BLogger.makeFunctionToPrettyPrintCounts(xSSfake)
    
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
        pprintCountVec(xSSslice)
        if b_debugOutputDir:
            plotCompsFromSS(
                curModel, xSSslice, 
                os.path.join(b_debugOutputDir, 'NewComps_Step%d.png' % (i+1)),
                vocabList=vocabList)
            propSS = curSSwhole.copy()
            propSS.transferMassFromExistingToExpansion(
                uid=targetUID, xSS=xSSslice)
            propModel = curModel.copy()
            propModel.update_global_params(propSS)
            propLdict = propModel.calc_evidence(SS=propSS, todict=1)
            propLdictList.append(propLdict)
            if curModel.getAllocModelName().count('HDP'):
                docUsageVec = xSSslice.getSelectionTerm('DocUsageCount')
                for k, uid in enumerate(xSSslice.uids):
                    docUsageByUID[uid].append(docUsageVec[k])

        # Cleanup by deleting small clusters 
        if i < b_nRefineSteps - 1:
            if i == b_nRefineSteps - 2:
                # After all but last step, 
                # delete small (but not empty) comps
                minNumAtomsToStay = b_minNumAtomsForNewComp
            else:
                # Always remove empty clusters. They waste our time.
                minNumAtomsToStay = 1
            xSSslice = cleanupDeleteSmallClusters(
                xSSslice, minNumAtomsToStay, pprintCountVec=pprintCountVec)
        # Cleanup by merging clusters
        if i == b_nRefineSteps - 2:
            DebugInfo['mergestep'] = i + 1
            xSSslice = cleanupMergeClusters(
                xSSslice, curModel,
                obsSS=xSSfake,
                vocabList=Dslice.vocabList,
                pprintCountVec=pprintCountVec,
                b_debugOutputDir=b_debugOutputDir, **kwargs)
        # Exit early if no promising new clusters are created
        nnzCount = np.sum(xSSslice.getCountVec() >= 1)
        if nnzCount < 2:
            break

    DebugInfo['Kfinal'] = xSSslice.K
    if b_debugOutputDir:
        savefilename = os.path.join(
            b_debugOutputDir, 'ProposalTrace_ELBO.png')
        plotELBOtermsForProposal(curLdict, propLdictList,
                                 savefilename=savefilename)
        if curModel.getAllocModelName().count('HDP'):
            savefilename = os.path.join(
                b_debugOutputDir, 'ProposalTrace_DocUsage.png')
            plotDocUsageForProposal(docUsageByUID,
                                    savefilename=savefilename)

    # Raise error if we didn't create enough "big-enough" states.
    nnzCount = np.sum(xSSslice.getCountVec() >= b_minNumAtomsForNewComp)
    if nnzCount < 2:
        DebugInfo['errorMsg'] = \
            "Could not create at least two comps" + \
            " with mass >= %.1f (--%s)" % (
                b_minNumAtomsForNewComp, 'b_minNumAtomsForNewComp')
        BLogger.pprint('Refinement Failed. ' + DebugInfo['errorMsg'])
        return None, DebugInfo

    # If here, we have a valid proposal. 
    # Need to verify mass conservation
    if hasattr(Dslice, 'word_count') and \
            curModel.obsModel.DataAtomType.count('word'):
        origMass = np.inner(Dslice.word_count, curLPslice['resp'][:,ktarget])
    else:
        origMass = curLPslice['resp'][:,ktarget].sum()
    newMass = xSSslice.getCountVec().sum()
    assert np.allclose(newMass, origMass, atol=1e-6, rtol=0)
    BLogger.pprint('Success. Created %d new comps.' % (
        DebugInfo['Kfinal']))

    if returnPropSS:
        raise NotImplementedError("TODO")
    return xSSslice, DebugInfo

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

"""
