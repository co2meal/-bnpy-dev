import numpy as np
import os
import sys
import bnpy.init.FromTruth
import BLogger

from scipy.special import digamma, gammaln

from bnpy.allocmodel.topics.HDPTopicRestrictedLocalStep \
    import summarizeRestrictedLocalStep_HDPTopicModel

from BCleanup import cleanupMergeClusters, cleanupDeleteSmallClusters
from BirthProposalError import BirthProposalError
from bnpy.viz.PlotComps import plotCompsFromSS
from bnpy.viz.ProposalViz import plotELBOtermsForProposal
from bnpy.viz.ProposalViz import plotDocUsageForProposal
from bnpy.viz.ProposalViz import makeSingleProposalHTMLStr
from bnpy.viz.PrintTopics import vec2str

from BRestrictedLocalStep import \
    summarizeRestrictedLocalStep, \
    makeExpansionSSFromZ

from bnpy.init.FromScratchBregman import initSS_BregmanDiv

DefaultLPkwargs = dict(
    restartLP=1,
    convThrLP=0.001,
    nCoordAscentItersLP=50,
    )

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
    targetUID = kwargs['targetUID']
    BLogger.startUIDSpecificLog(kwargs['targetUID'])

    # Make an output directory for HTML
    if kwargs['b_debugWriteHTML']:
       kwargs['b_debugOutputDir'] = createBirthProposalHTMLOutputDir(**kwargs)
    else:
        if 'b_debugOutputDir' in kwargs:
            if kwargs['b_debugOutputDir'].lower() == 'none':
                kwargs['b_debugOutputDir'] = None

    doExtendExistingProposal = False
    if 'curSSwhole' in kwargs:
        curSSwhole = kwargs['curSSwhole']
        if hasattr(curSSwhole, 'propXSS'):
            if targetUID in curSSwhole.propXSS:
                doExtendExistingProposal = True

    if doExtendExistingProposal:
        xSSslice, DebugInfo = makeSummaryForExistingBirthProposal(
            Dslice, curModel, curLPslice, **kwargs)
    else:
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
        seed=0,
        b_nRefineSteps=3,
        b_debugOutputDir=None,
        b_minNumAtomsForNewComp=None,
        b_doInitCompleteLP=1,
        b_method_initCoordAscent='fromprevious',
        vocabList=None,
        **kwargs):
    ''' Create summary that reassigns mass from target to Kfresh new comps.

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
    xInitSStarget, Info = initSS_BregmanDiv(
        Dslice, curModel, curLPslice, 
        K=xK, 
        ktarget=ktarget,
        lapFrac=lapFrac,
        seed=seed + int(1000 * lapFrac),
        logFunc=BLogger.pprint,
        NiterForBregmanKMeans=kwargs['b_NiterForBregmanKMeans'],
        **kwargs)
    # EXIT EARLY: if proposal initialization fails (not enough data).
    if xInitSStarget is None:
        BLogger.pprint('Proposal initialization FAILED. ' + \
                       Info['errorMsg'])
        return None, Info

    # If here, we have a valid set of initial stats.
    xInitSStarget.setUIDs(newUIDs[:xInitSStarget.K])
    if b_doInitCompleteLP:
        # Create valid whole-dataset clustering from hard init
        xInitSSslice, tempInfo = makeExpansionSSFromZ(
            Dslice=Dslice, curModel=curModel, curLPslice=curLPslice,
            ktarget=ktarget,
            xInitSS=xInitSStarget,
            atomType=Info['atomType'],
            targetZ=Info['targetZ'],
            chosenDataIDs=Info['chosenDataIDs'],
            **kwargs)
        Info.update(tempInfo)

        xSSslice = xInitSSslice
    else:
        xSSslice = xInitSStarget

    if b_debugOutputDir:
        plotCompsFromSS(
            curModel, xSSslice, 
            os.path.join(b_debugOutputDir, 'NewComps_Init.png'),
            vocabList=vocabList)

        # Determine current model objective score
        curModelFWD = curModel.copy()
        curModelFWD.update_global_params(SS=curSSwhole)
        curLdict = curModelFWD.calc_evidence(SS=curSSwhole, todict=1)
        # Track proposal ELBOs as refinement improves things
        propLdictList = list()
        # Create initial proposal
        if b_doInitCompleteLP:
            propSS = curSSwhole.copy()
            propSS.transferMassFromExistingToExpansion(
                uid=targetUID, xSS=xSSslice)
            # Verify quality
            assert np.allclose(propSS.getCountVec().sum(),
                               curSSwhole.getCountVec().sum())
            propModel = curModel.copy()
            propModel.update_global_params(propSS)
            propLdict = propModel.calc_evidence(SS=propSS, todict=1)
            BLogger.pprint(
                "init %d/%d  gainL % .3e  propL % .3e  curL % .3e" % (
                    0, b_nRefineSteps,
                    propLdict['Ltotal'] - curLdict['Ltotal'],
                    propLdict['Ltotal'],
                    curLdict['Ltotal']))
            propLdictList.append(propLdict)

        docUsageByUID = dict()
        if curModel.getAllocModelName().count('HDP'):
            for k, uid in enumerate(xInitSStarget.uids):
                if 'targetZ' in Info:
                    if Info['atomType'].count('doc'):
                        initDocUsage_uid = np.sum(Info['targetZ'] == k)
                    else:
                        initDocUsage_uid = 0.0
                        for d in xrange(Dslice.nDoc):
                            start = Dslice.doc_range[d]
                            stop = Dslice.doc_range[d+1]
                            initDocUsage_uid += np.any(
                                Info['targetZ'][start:stop] == k)
                else:
                    initDocUsage_uid = 0.0
                docUsageByUID[uid] = [initDocUsage_uid]

    # Create initial observation model
    xObsModel = curModel.obsModel.copy()
    if b_method_initCoordAscent == 'fromprevious' and 'xLPslice' in Info:
        xInitLPslice = Info['xLPslice']
    else:
        xInitLPslice = None

    # Log messages to describe the initialization.
    BLogger.pprint(' Running %d refinement iterations (--b_nRefineSteps)' % (
        b_nRefineSteps))

    # Make a function to pretty-print counts as we refine the initialization
    pprintCountVec = BLogger.makeFunctionToPrettyPrintCounts(xSSslice)

    # Run several refinement steps. 
    # Each step does a restricted local step to improve
    # the proposed cluster assignments.
    for rstep in range(b_nRefineSteps):

        # Restricted local step!
        # * xInitSS : specifies obs-model stats used for initialization
        xSSslice, refineInfo = summarizeRestrictedLocalStep(
            Dslice=Dslice, 
            curModel=curModel,
            curLPslice=curLPslice,
            curSSwhole=curSSwhole,
            ktarget=ktarget,
            xUIDs=xSSslice.uids,
            xObsModel=xObsModel,
            xInitSS=xSSslice,
            xInitLPslice=xInitLPslice,
            LPkwargs=LPkwargs,
            **kwargs)
        Info.update(refineInfo)
        
        # Get most recent xLPslice for initialization
        if b_method_initCoordAscent == 'fromprevious' and 'xLPslice' in Info:
            xInitLPslice = Info['xLPslice']

        # Show diagnostics for new states
        if rstep == 0:
            targetPi = refineInfo['emptyPi'] + refineInfo['xPiVec'].sum()
            BLogger.pprint(
                " target prob redistributed by policy %s (--b_method_xPi)" % (
                    kwargs['b_method_xPi']))
            msg = " pi[ktarget] before %.4f  after %.4f." % (
                targetPi, refineInfo['emptyPi'])
            BLogger.pprint(msg)
            BLogger.pprint(" pi[new comps]: "  + \
                vec2str(
                    refineInfo['xPiVec'],
                    width=6, minVal=0.0001))

            logLPConvergenceDiagnostics(
                refineInfo, rstep=rstep, b_nRefineSteps=b_nRefineSteps)

            BLogger.pprint("   " + vec2str(xInitSStarget.uids))



        pprintCountVec(xSSslice)
        # Write HTML debug info
        if b_debugOutputDir:
            plotCompsFromSS(
                curModel, xSSslice, 
                os.path.join(b_debugOutputDir,
                             'NewComps_Step%d.png' % (rstep+1)),
                vocabList=vocabList)
            propSS = curSSwhole.copy()
            propSS.transferMassFromExistingToExpansion(
                uid=targetUID, xSS=xSSslice)
            # Reordering only lifts score by small amount. Not worth it.
            # propSS.reorderComps(np.argsort(-1 * propSS.getCountVec()))
            propModel = curModel.copy()
            propModel.update_global_params(propSS)
            propLdict = propModel.calc_evidence(SS=propSS, todict=1)
            BLogger.pprint(
                "step %d/%d  gainL % .3e  propL % .3e  curL % .3e" % (
                    rstep+1, b_nRefineSteps,
                    propLdict['Ltotal'] - curLdict['Ltotal'],
                    propLdict['Ltotal'],
                    curLdict['Ltotal']))
            propLdictList.append(propLdict)
            if curModel.getAllocModelName().count('HDP'):
                docUsageVec = xSSslice.getSelectionTerm('DocUsageCount')
                for k, uid in enumerate(xSSslice.uids):
                    docUsageByUID[uid].append(docUsageVec[k])

        # Cleanup by deleting small clusters 
        if rstep < b_nRefineSteps - 1:
            if rstep == b_nRefineSteps - 2:
                # After all but last step, 
                # delete small (but not empty) comps
                minNumAtomsToStay = b_minNumAtomsForNewComp
            else:
                # Always remove empty clusters. They waste our time.
                minNumAtomsToStay = np.minimum(1, b_minNumAtomsForNewComp)
            xSSslice, xInitLPslice = cleanupDeleteSmallClusters(
                xSSslice, minNumAtomsToStay,
                xInitLPslice=xInitLPslice, 
                pprintCountVec=pprintCountVec)
        # Cleanup by merging clusters
        if rstep == b_nRefineSteps - 2:
            Info['mergestep'] = rstep + 1
            xSSslice, xInitLPslice = cleanupMergeClusters(
                xSSslice, curModel,
                obsSSkeys=xInitSStarget._Fields._FieldDims.keys(),
                vocabList=vocabList,
                pprintCountVec=pprintCountVec,
                xInitLPslice=xInitLPslice,
                b_debugOutputDir=b_debugOutputDir, **kwargs)

    Info['Kfinal'] = xSSslice.K
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
        Info['errorMsg'] = \
            "Could not create at least two comps" + \
            " with mass >= %.1f (--%s)" % (
                b_minNumAtomsForNewComp, 'b_minNumAtomsForNewComp')
        BLogger.pprint('Proposal FAILED refinement. ' + Info['errorMsg'])
        return None, Info

    # If here, we have a valid proposal. 
    # Need to verify mass conservation
    if hasattr(Dslice, 'word_count') and \
            curModel.obsModel.DataAtomType.count('word') and \
            curModel.getObsModelName().count('Mult'):
        origMass = np.inner(Dslice.word_count, curLPslice['resp'][:,ktarget])
    else:
        origMass = curLPslice['resp'][:,ktarget].sum()
    newMass = xSSslice.getCountVec().sum()
    assert np.allclose(newMass, origMass, atol=1e-6, rtol=0)
    BLogger.pprint('Proposal construction DONE.' + \
        ' Created %d candidate clusters.' % (Info['Kfinal']))
    return xSSslice, Info


def makeSummaryForExistingBirthProposal(
        Dslice, curModel, curLPslice,
        curSSwhole=None,
        targetUID=None,
        ktarget=None,
        LPkwargs=DefaultLPkwargs,
        lapFrac=0,
        b_nRefineSteps=3,
        b_debugOutputDir=None,
        b_method_initCoordAscent='fromprevious',
        vocabList=None,
        **kwargs):
    ''' Create summary that reassigns mass from target given set of comps

    Given set of comps is a fixed proposal from a previously-seen batch.

    Returns
    -------
    xSSslice : SuffStatBag
        Contains exact summaries for reassignment of target mass.
        * Total mass is equal to mass assigned to ktarget in curLPslice
        * Number of components is Kfresh
    Info : dict
        Contains info for detailed debugging of construction process.
    '''
    if targetUID is None:
        targetUID = curSSwhole.k2uid(ktarget)
    if ktarget is None:
        ktarget = curSSwhole.uid2k(targetUID)
    # START log for this birth proposal
    BLogger.pprint(
        'Extending previous proposal for targetUID %s at lap %.2f' % (
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

    assert targetUID in curSSwhole.propXSS
    xinitSS = curSSwhole.propXSS[targetUID]
    xK = xinitSS.K
    if xK + curSSwhole.K > kwargs['Kmax']:
        errorMsg = 'Cancelled.' + \
            'Adding 2 or more states would exceed budget of %d comps.' % (
                kwargs['Kmax'])
        BLogger.pprint(errorMsg)
        return None, dict(errorMsg=errorMsg)

    # Log messages to describe the initialization.
    # Make a function to pretty-print counts as we refine the initialization
    pprintCountVec = BLogger.makeFunctionToPrettyPrintCounts(xinitSS)
    BLogger.pprint('  Using previous proposal with %d clusters %s.' % (
        xinitSS.K, '(--b_Kfresh=%d)' % kwargs['b_Kfresh']))
    BLogger.pprint("  Initial uid/counts from previous proposal:")
    BLogger.pprint('   ' + vec2str(xinitSS.uids))
    pprintCountVec(xinitSS)
    BLogger.pprint('  Running %d refinement iterations (--b_nRefineSteps)' % (
        b_nRefineSteps))

    xSSinitPlusSlice = xinitSS.copy()
    if b_debugOutputDir:
        plotCompsFromSS(
            curModel, xinitSS, 
            os.path.join(b_debugOutputDir, 'NewComps_Init.png'),
            vocabList=vocabList)

        # Determine current model objective score
        curModelFWD = curModel.copy()
        curModelFWD.update_global_params(SS=curSSwhole)
        curLdict = curModelFWD.calc_evidence(SS=curSSwhole, todict=1)
        # Track proposal ELBOs as refinement improves things
        propLdictList = list()
        docUsageByUID = dict()
        if curModel.getAllocModelName().count('HDP'):
            for k, uid in enumerate(xinitSS.uids):
                initDocUsage_uid = 0.0
                docUsageByUID[uid] = [initDocUsage_uid]

    # Create initial observation model
    xObsModel = curModel.obsModel.copy()
    xInitLPslice = None
    Info = dict()
    # Run several refinement steps. 
    # Each step does a restricted local step to improve
    # the proposed cluster assignments.
    for rstep in range(b_nRefineSteps):
        # Restricted local step!
        # * xInitSS : specifies obs-model stats used for initialization
        xSSslice, refineInfo = summarizeRestrictedLocalStep(
            Dslice=Dslice, 
            curModel=curModel,
            curLPslice=curLPslice,
            ktarget=ktarget,
            xUIDs=xSSinitPlusSlice.uids,
            xObsModel=xObsModel,
            xInitSS=xSSinitPlusSlice, # first time in loop <= xinitSS
            xInitLPslice=xInitLPslice,
            LPkwargs=LPkwargs,
            **kwargs)

        xSSinitPlusSlice += xSSslice
        if rstep >= 1:
            xSSinitPlusSlice -= prevSSslice
        prevSSslice = xSSslice

        Info.update(refineInfo)
        # Show diagnostics for new states
        pprintCountVec(xSSslice)
        logLPConvergenceDiagnostics(
            refineInfo, rstep=rstep, b_nRefineSteps=b_nRefineSteps)
        # Get most recent xLPslice for initialization
        if b_method_initCoordAscent == 'fromprevious' and 'xLPslice' in Info:
            xInitLPslice = Info['xLPslice']
        if b_debugOutputDir:
            plotCompsFromSS(
                curModel, xSSslice, 
                os.path.join(b_debugOutputDir,
                             'NewComps_Step%d.png' % (rstep+1)),
                vocabList=vocabList)
            propSS = curSSwhole.copy()
            propSS.transferMassFromExistingToExpansion(
                uid=targetUID, xSS=xSSslice)
            propModel = curModel.copy()
            propModel.update_global_params(propSS)
            propLdict = propModel.calc_evidence(SS=propSS, todict=1)
            BLogger.pprint(
                "step %d/%d  gainL % .3e  propL % .3e  curL % .3e" % (
                    rstep+1, b_nRefineSteps,
                    propLdict['Ltotal'] - curLdict['Ltotal'],
                    propLdict['Ltotal'],
                    curLdict['Ltotal']))
            propLdictList.append(propLdict)
            if curModel.getAllocModelName().count('HDP'):
                docUsageVec = xSSslice.getSelectionTerm('DocUsageCount')
                for k, uid in enumerate(xSSslice.uids):
                    docUsageByUID[uid].append(docUsageVec[k])

    Info['Kfinal'] = xSSslice.K
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

    # If here, we have a valid proposal. 
    # Need to verify mass conservation
    if hasattr(Dslice, 'word_count') and \
            curModel.obsModel.DataAtomType.count('word') and \
            curModel.getObsModelName().count('Mult'):
        origMass = np.inner(Dslice.word_count, curLPslice['resp'][:,ktarget])
    else:
        origMass = curLPslice['resp'][:,ktarget].sum()
    newMass = xSSslice.getCountVec().sum()
    assert np.allclose(newMass, origMass, atol=1e-6, rtol=0)
    BLogger.pprint('Proposal SUCCESS. Created %d candidate clusters.' % (
        Info['Kfinal']))
    return xSSslice, Info


def createBirthProposalHTMLOutputDir(
        taskoutpath='/tmp/', 
        lapFrac=0, batchPos=None, nBatch=None, targetUID=0,
        dataName=None, **kwargs):
    ''' Create string that is absolute path to dir for saving birth HTML logs.

    Returns
    -------
    b_debugOutputDir : string filepath
    '''
    if taskoutpath is None:
        raise ValueError("Need taskoutpath to not be None")

    if batchPos is None:
        subdirname = 'lap=%04d_targetUID=%04d_Kfresh=%d_%s' % (
            np.ceil(lapFrac),
            targetUID,
            kwargs['b_Kfresh'],
            kwargs['b_method_initCoordAscent'])
    else:
        subdirname = 'lap=%04d_batchPos%04dof%d_targetUID=%04d' % (
            np.ceil(lapFrac),
            batchPos,
            nBatch,
            targetUID)
    if dataName:
        b_debugOutputDir = os.path.join(
            taskoutpath, 'html-birth-logs', dataName, subdirname)        
    else:
        b_debugOutputDir = os.path.join(
            taskoutpath, 'html-birth-logs', subdirname)        
    # Create this directory if it doesn't exist already
    if not os.path.exists(b_debugOutputDir):
       os.makedirs(b_debugOutputDir)
    # Clear out any previous files from this directory
    from bnpy.Run import deleteAllFilesFromDir
    deleteAllFilesFromDir(b_debugOutputDir)
    return b_debugOutputDir


def logLPConvergenceDiagnostics(refineInfo, rstep=0, b_nRefineSteps=0):
    if 'xLPslice' not in refineInfo:
        return
    xLPslice = refineInfo['xLPslice']
    if '_maxDiff' not in xLPslice:
        return

    msg = "step %d/%d " % (rstep + 1, b_nRefineSteps)
    target_docs = np.flatnonzero(xLPslice['_maxDiff'] >= 0)
    if target_docs.size == 0:
        BLogger.pprint(msg + "No docs with active local step.")
        return

    msg += "nCAIters "
    for p in [0, 10, 50, 90, 100]:
        if p > 0:
            msg += "|"
        ip = np.percentile(xLPslice['_nIters'][target_docs], p)
        msg += " %3d%% %7d" % (p, ip)
    msg += "\n         Ndiff    "
    for p in [0, 10, 50, 90, 100]:
        if p > 0:
            msg += "|"
        md = np.percentile(xLPslice['_maxDiff'][target_docs], p)
        msg += " %3d%% %7.3f" % (p, md)
    BLogger.pprint(msg)

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
