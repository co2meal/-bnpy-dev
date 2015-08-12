import numpy as np
import os
import sys
import bnpy.init.FromTruth
import BLogger

from scipy.special import digamma, gammaln
from BCleanup import cleanupMergeClusters, cleanupDeleteSmallClusters
from BRefine import assignSplitStats, _calc_expansion_alphaEbeta
from BirthProposalError import BirthProposalError
from bnpy.viz.PlotComps import plotAndSaveCompsFromSS
from bnpy.viz.ProposalViz import plotELBOtermsForProposal
from bnpy.viz.ProposalViz import plotDocUsageForProposal
from bnpy.viz.ProposalViz import makeSingleProposalHTMLStr
from bnpy.viz.PrintTopics import vec2str

from bnpy.init import initSSByBregDiv

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
        BLogger.pprint("WROTE HTML: " + htmlfilepath, 'debug')

    # Raise error if we didn't create valid SS
    if xSSslice is None:
        raise BirthProposalError(DebugInfo['msg'])
    # Raise error if we didn't create enough "big-enough" states.
    nnzCount = np.sum(xSSslice.getCountVec() >= 1)
    if nnzCount < 2:
        raise BirthProposalError(
            "Could not create at least two comps with mass >= 1.")
    # If here, we have a valid proposal. 
    # Need to verify mass conservation
    ktarget = curSSwhole.uid2k(kwargs['targetUID'])
    if hasattr(Dslice, 'word_count') and \
            hmodel.obsModel.DataAtomType.count('word'):
        origMass = np.inner(Dslice.word_count, curLPslice['resp'][:,ktarget])
    else:
        origMass = curLPslice['resp'][:,ktarget].sum()
    newMass = xSSslice.getCountVec().sum()
    assert np.allclose(newMass, origMass, atol=1e-6, rtol=0)
    return xSSslice, DebugInfo



def createSplitStats_BregDiv(
        Dslice, curModel, curLPslice, 
        curSSwhole=None,
        targetUID=0,
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
    ''' Reassign target component to new states using bregman divergence.

    Returns
    -------
    xSSslice : stats for reassigned mass
        total count is equal to SS.getCountVec()[ktarget]
        number of components is Kx
    DebugInfo : dict with info for visualization, etc
    '''
    BLogger.pprint('targetUID ' + str(targetUID))
    # Parse some kwarg input
    if hasattr(Dslice, 'vocabList') and Dslice.vocabList is not None:
        vocabList = Dslice.vocabList
    if ktarget is None:
        ktarget = curSSwhole.uid2k(targetUID)
    if b_debugOutputDir:
        plotAndSaveCompsFromSS(
            curModel, curSSwhole, b_debugOutputDir, 'OrigComps.png',
            vocabList=vocabList,
            compsToHighlight=[ktarget])
    # Create suff stats for some new states
    xK = newUIDs.size
    xSSfake, DebugInfo = initSSByBregDiv(
        Dslice=Dslice,
        curModel=curModel,
        curLPslice=curLPslice,
        K=xK,
        ktarget=ktarget,
        lapFrac=lapFrac,
        seed=1000 * lapFrac,
        **kwargs)
    if xSSfake is None:
        return None, DebugInfo
    # Record some debug info about the new states
    xSSfake.setUIDs(newUIDs[:xSSfake.K])
    strUIDs = vec2str(xSSfake.uids)
    BLogger.pprint('   ' + strUIDs)
    if b_debugOutputDir:
        plotAndSaveCompsFromSS(
            curModel, xSSfake, b_debugOutputDir, 'NewComps_Init.png',
            vocabList=vocabList)
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
            plotAndSaveCompsFromSS(
                curModel, xSSslice, b_debugOutputDir,
                filename='NewComps_Step%d.png' % (i+1),
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
                # After all but last step, delete small (but not empty) comps
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

    if b_debugOutputDir:
        savefilename = os.path.join(
            b_debugOutputDir, 'ProposalTrace_ELBO.png')
        plotELBOtermsForProposal(curLdict, propLdictList,
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
        if curModel.getAllocModelName().count('HDP'):
            savefilename = os.path.join(
                b_debugOutputDir, 'ProposalTrace_DocUsage.png')
            plotDocUsageForProposal(docUsageByUID,
                                    savefilename=savefilename)
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
