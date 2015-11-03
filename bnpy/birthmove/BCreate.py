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
        BLogger.pprint('Proposal FAILED initialization. ' + \
                       DebugInfo['errorMsg'])
        return None, DebugInfo

    # Describe the initialization.
    BLogger.pprint('  Bregman k-means init delivered %d clusters %s.' % (
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

        # Compute the objective Lscore for the initialization!
        # Assume any junk token is assigned to most common new state in each doc
        #init_xLPslice = curModel.allocModel.initLPFromResp(
        #    Dslice, Z=DebugInfo['targetZ'])
        #init_xSSslice = curModel.get_global_suff_stats(
        #    Dslice, init_xLPslice, doPrecompEntropy=1)

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
                vocabList=vocabList,
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
        BLogger.pprint('Proposal FAILED refinement. ' + DebugInfo['errorMsg'])
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
    BLogger.pprint('Proposal SUCCESS. Created %d candidate clusters.' % (
        DebugInfo['Kfinal']))

    if returnPropSS:
        raise NotImplementedError("TODO")
    return xSSslice, DebugInfo

