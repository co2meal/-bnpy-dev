import numpy as np
import os
import sys
import bnpy.init.FromTruth
import BLogger

from scipy.special import digamma, gammaln

from bnpy.allocmodel.topics.HDPTopicRestrictedLocalStep \
    import summarizeRestrictedLocalStep_HDPTopicModel

from BCleanup import cleanupMergeClusters, cleanupDeleteSmallClusters
#from BRefine import assignSplitStats, _calc_expansion_alphaEbeta
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

def summarizeRestrictedLocalStep(**kwargs):
    ''' Wrapper that calls allocation-model specific restricted local function.

    Looks up the right function by allocation model name, and calls it.

    Returns
    -------
    xSSslice : SuffStatBag, with Kfresh new clusters
    DebugInfo : dict
    '''
    aModelName = kwargs['curModel'].getAllocModelName()
    summarizeFuncName = \
        "summarizeRestrictedLocalStep_" + aModelName
    summarizeFunc = globals()[summarizeFuncName]
    return summarizeFunc(**kwargs)

def makeSummaryForBirthProposal_HTMLWrapper(
        Dslice, curModel, curLPslice,        
        **kwargs):
    ''' Thin wrapper around makeSummaryForBirthProposal that produces HTML.

    Will produce HTML output regardless of if makeSummaryForBirthProposal
    succeeds or if it fails somewhere the construction process.

    Returns
    -------
    xSSslice : SuffStatBag
        Contains exact summaries for reassignment of target mass.
        * Total mass is equal to mass assigned to ktarget in curLPslice
        * Number of components is Kfresh
    Info : dict
        Contains info for detailed debugging of construction process.
    '''
    BLogger.startUIDSpecificLog(kwargs['targetUID'])

    # Make an output directory for HTML
    if kwargs['b_debugWriteHTML']:
       kwargs['b_debugOutputDir'] = createBirthProposalHTMLOutputDir(**kwargs)

    xSSslice, DebugInfo = makeSummaryForBirthProposal(
        Dslice, curModel, curLPslice, **kwargs)

    # Write output to HTML
    if 'b_debugOutputDir' in kwargs and kwargs['b_debugOutputDir']:
        htmlstr = makeSingleProposalHTMLStr(DebugInfo, **kwargs)
        htmlfilepath = os.path.join(kwargs['b_debugOutputDir'], 'index.html')
        with open(htmlfilepath, 'w') as f:
            f.write(htmlstr)
    BLogger.stopUIDSpecificLog(kwargs['targetUID'])
    return xSSslice, DebugInfo


def makeSummaryForBirthProposal(
        Dslice, curModel, curLPslice,
        curSSwhole=None,
        b_creationProposalName='bregmankmeans',
        targetUID=None,
        ktarget=None,
        newUIDs=None,
        LPkwargs=DefaultLPkwargs,
        lapFrac=0,
        b_nRefineSteps=3,
        b_debugOutputDir=None,
        b_minNumAtomsForNewComp=None,
        vocabList=None,
        **kwargs):
    ''' Create summary that reassigns mass from target comp to Kfresh new comps.

    TODO support other options than bregman???

    Returns
    -------
    xSSslice : SuffStatBag
        Contains exact summaries for reassignment of target mass.
        * Total mass is equal to mass assigned to ktarget in curLPslice
        * Number of components is Kfresh
    Info : dict
        Contains info for detailed debugging of construction process.
    '''
    # Parse input to decide which cluster to target
    # * targetUID is the unique ID of this cluster
    # * ktarget is its position in the current cluster ordering
    if targetUID is None:
        targetUID = curSSwhole.k2uid(ktarget)
    if ktarget is None:
        ktarget = curSSwhole.uid2k(targetUID)
    # START log for this birth proposal
    BLogger.pprint('Creating proposal for targetUID %s at lap %.2f' % (
        targetUID, lapFrac))
    # Grab vocabList, if available.
    if hasattr(Dslice, 'vocabList') and Dslice.vocabList is not None:
        vocabList = Dslice.vocabList
    # Parse input to decide where to save HTML output
    if b_debugOutputDir == 'None':
        b_debugOutputDir = None
    if b_debugOutputDir:
        BLogger.pprint(
            'HTML output:' + b_debugOutputDir)
        # Create snapshot of current model comps
        plotCompsFromSS(
            curModel, curSSwhole, 
            os.path.join(b_debugOutputDir, 'OrigComps.png'),
            vocabList=vocabList,
            compsToHighlight=[ktarget])

    # Determine exactly how many states we can make...
    xK = len(newUIDs)
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
    xInitSS, DebugInfo = initSS_BregmanDiv(
        Dslice, curModel, curLPslice, 
        K=xK, 
        ktarget=ktarget,
        lapFrac=lapFrac,
        seed=1000 * lapFrac,
        **kwargs)
    BLogger.pprint(DebugInfo['targetAssemblyMsg'])
    # EXIT EARLY: if proposal initialization fails (not enough data).
    if xInitSS is None:
        BLogger.pprint('Proposal FAILED initialization. ' + \
                       DebugInfo['errorMsg'])
        return None, DebugInfo

    # If here, we have a valid set of initial stats.
    xInitSS.setUIDs(newUIDs[:xInitSS.K])
    # Log messages to describe the initialization.
    BLogger.pprint('  Bregman k-means init delivered %d clusters %s.' % (
        xInitSS.K, '(--b_Kfresh=%d)' % kwargs['b_Kfresh']))
    BLogger.pprint('  Running %d refinement iterations (--b_nRefineSteps)' % (
        b_nRefineSteps))
    BLogger.pprint('   ' + vec2str(xInitSS.uids))
    if b_debugOutputDir:
        plotCompsFromSS(
            curModel, xInitSS, 
            os.path.join(b_debugOutputDir, 'NewComps_Init.png'),
            vocabList=vocabList)
        curLdict = curModel.calc_evidence(SS=curSSwhole, todict=1)
        propLdictList = list()
        docUsageByUID = dict()
        if curModel.getAllocModelName().count('HDP'):
            for k, uid in enumerate(xInitSS.uids):
                if 'targetZ' in DebugInfo:
                    initDocUsage_uid = np.sum(DebugInfo['targetZ'] == k)
                else:
                    initDocUsage_uid = 0.0
                docUsageByUID[uid] = [initDocUsage_uid]
    # Make a function to pretty-print counts as we refine the initialization
    pprintCountVec = BLogger.makeFunctionToPrettyPrintCounts(xInitSS)
    # Create the initial stats and the initial observation model
    xSSslice = xInitSS
    xObsModel = curModel.obsModel.copy()
    # Run several refinement steps. 
    # Each step does a restricted local step to improve
    # the proposed cluster assignments.
    for i in range(b_nRefineSteps):
        # Restricted local step!
        # * xInitSS : specifies obs-model stats used for initialization
        xSSslice, Info = summarizeRestrictedLocalStep(
            Dslice=Dslice, 
            curModel=curModel,
            curLPslice=curLPslice,
            ktarget=ktarget,
            xUIDs=xSSslice.uids,
            xObsModel=xObsModel,
            xInitSS=xSSslice,
            LPkwargs=LPkwargs,
            emptyPiFrac=0.01)
        DebugInfo.update(Info)

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
                obsSS=xInitSS,
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

    # EXIT EARLY: error if we didn't create enough "big-enough" states.
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

    return xSSslice, DebugInfo


def createBirthProposalHTMLOutputDir(
        taskoutpath='/tmp/', 
        lapFrac=0, batchPos=0, nBatch=0, targetUID=0, **kwargs):
    ''' Create string that is absolute path to dir for saving birth HTML logs.

    Returns
    -------
    b_debugOutputDir : string filepath
    '''
    if taskoutpath is None:
        raise ValueError("Need taskoutpath to not be None")
    b_debugOutputDir = os.path.join(
        taskoutpath,
        'html-birth-logs',
        'lap=%04d_batchPos%04dof%d_targetUID=%04d' % (
            np.ceil(lapFrac),
            batchPos,
            nBatch,
            targetUID))
    if not os.path.exists(b_debugOutputDir):
       os.makedirs(b_debugOutputDir)
    return b_debugOutputDir


## DEPRECATED. HISTORICALLY INTERESTING CODE.
'''
    createSplitStatsMap = dict([
        (k,v) for (k,v) in globals().items() if str(k).count('createSplitStats')])
    funcName = 'createSplitStats' + '_' + b_creationProposalName
    if funcName not in createSplitStatsMap:
        raise NotImplementedError('Unrecognized function: ' + funcName)    
    # Execute model-specific function to make expansion stats
    # This call may return early if expansion failed,
    # due to creating too few states that are big-enough.
    # Need to finalize debug html before raising error.
    createSplitStatsFunc = createSplitStatsMap[funcName]
'''
