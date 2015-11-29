'''
Implementation of parallel memoized variational algorithm for bnpy models.
'''
import numpy as np
import multiprocessing
import os
import ElapsedTimeLogger 

from collections import defaultdict

from bnpy.birthmove.BCreateManyProposals \
    import makeSummariesForManyBirthProposals

from bnpy.birthmove import \
    BLogger, \
    selectShortListForBirthAtLapStart, \
    summarizeRestrictedLocalStep, \
    selectCompsForBirthAtCurrentBatch
from bnpy.mergemove import MLogger, SLogger
from bnpy.mergemove import selectCandidateMergePairs, ELBO_GAP_ACCEPT_TOL
from bnpy.deletemove import DLogger, selectCandidateDeleteComps
from bnpy.util import sharedMemDictToNumpy, sharedMemToNumpyArray
from bnpy.util import argsort_bigtosmall_stable
from LearnAlg import makeDictOfAllWorkspaceVars
from LearnAlg import LearnAlg
from SharedMemWorker import SharedMemWorker
from bnpy.viz.PrintTopics import count2str

class MemoVBMovesAlg(LearnAlg):

    def __init__(self, **kwargs):
        ''' Constructor for LearnAlg.
        '''
        # Initialize instance vars related to
        # birth / merge / delete records
        LearnAlg.__init__(self, **kwargs)
        self.SSmemory = dict()
        self.LastUpdateLap = dict()

    def makeNewUIDs(self, nMoves=1, b_Kfresh=0, **kwargs):
        newUIDs = np.arange(self.maxUID + 1,
                            self.maxUID + nMoves * b_Kfresh + 1)
        self.maxUID += newUIDs.size
        return newUIDs

    def fit(self, hmodel, DataIterator, **kwargs):
        ''' Run learning algorithm that fits parameters of hmodel to Data.

        Returns
        --------
        Info : dict of run information.

        Post Condition
        --------
        hmodel updated in place with improved global parameters.
        '''
        origmodel = hmodel
        self.maxUID = hmodel.obsModel.K - 1

        # Initialize Progress Tracking vars like nBatch, lapFrac, etc.
        iterid, lapFrac = self.initProgressTrackVars(DataIterator)

        # Save initial state
        self.saveParams(lapFrac, hmodel)

        # Custom func hook
        self.eval_custom_func(
            isInitial=1, **makeDictOfAllWorkspaceVars(**vars()))

        # Begin loop over batches of data...
        SS = None
        isConverged = False
        Lscore = -np.inf
        self.set_start_time_now()
        MoveLog = list()
        MoveRecordsByUID = dict()
        ConvStatus = np.zeros(DataIterator.nBatch)
        while DataIterator.has_next_batch():

            batchID = DataIterator.get_next_batch(batchIDOnly=1)

            # Update progress-tracking variables
            iterid += 1
            lapFrac = (iterid + 1) * self.lapFracInc
            self.lapFrac = lapFrac
            self.set_random_seed_at_lap(lapFrac)

            # Debug print header
            if self.doDebugVerbose():
                self.print_msg('========================== lap %.2f batch %d'
                               % (lapFrac, batchID))

            # Reset at top of every lap
            if self.isFirstBatch(lapFrac):
                MovePlans = dict()
                if SS is not None and SS.hasSelectionTerms():
                    SS._SelectTerms.setAllFieldsToZero()
            MovePlans = self.makeMovePlans(
                hmodel, SS, 
                MovePlans=MovePlans,
                MoveRecordsByUID=MoveRecordsByUID,
                lapFrac=lapFrac)

            # Local/Summary step for current batch
            SSbatch = self.calcLocalParamsAndSummarize_withExpansionMoves(
                DataIterator, hmodel,
                SS=SS,
                batchID=batchID,
                lapFrac=lapFrac,
                MovePlans=MovePlans,
                MoveRecordsByUID=MoveRecordsByUID,
                MoveLog=MoveLog)

            self.saveDebugStateAtBatch(
                'Estep', batchID, SSchunk=SSbatch, SS=SS, hmodel=hmodel)

            # Incremental update of whole-data SS given new SSbatch
            oldSSbatch = self.loadBatchAndFastForward(
                batchID, lapFrac, MoveLog)
            SS = self.incrementWholeDataSummary(
                SS, SSbatch, oldSSbatch, lapFrac=lapFrac, hmodel=hmodel)
            self.SSmemory[batchID] = SSbatch
            self.LastUpdateLap[batchID] = lapFrac

            # Global step
            hmodel = self.globalStep(hmodel, SS, lapFrac)

            # ELBO calculation
            Lscore = hmodel.calc_evidence(SS=SS)

            # Birth moves!
            if self.hasMove('birth') and hasattr(SS, 'propXSS'):
                hmodel, SS, Lscore, MoveLog, MoveRecordsByUID = \
                    self.runMoves_Birth(
                        hmodel, SS, Lscore, MovePlans,
                        MoveLog=MoveLog,
                        MoveRecordsByUID=MoveRecordsByUID,
                        lapFrac=lapFrac)

            if self.isLastBatch(lapFrac):
                # Merge move!
                if self.hasMove('merge') and 'm_UIDPairs' in MovePlans:
                    hmodel, SS, Lscore, MoveLog, MoveRecordsByUID = \
                        self.runMoves_Merge(
                            hmodel, SS, Lscore, MovePlans,
                            MoveLog=MoveLog,
                            MoveRecordsByUID=MoveRecordsByUID,
                            lapFrac=lapFrac,)
                # Afterwards, always discard any tracked merge terms
                SS.removeMergeTerms()

                # Delete move!
                if self.hasMove('delete') and 'd_targetUIDs' in MovePlans:
                    hmodel, SS, Lscore, MoveLog, MoveRecordsByUID = \
                        self.runMoves_Delete(
                            hmodel, SS, Lscore, MovePlans,
                            MoveLog=MoveLog,
                            MoveRecordsByUID=MoveRecordsByUID,
                            lapFrac=lapFrac,)
                # Afterwards, always discard any tracking terms
                SS.removeMergeTerms()
                if hasattr(SS, 'propXSS'):
                    del SS.propXSS

                # Shuffle : Rearrange order (big to small)
                if self.hasMove('shuffle'):
                    hmodel, SS, Lscore, MoveLog, MoveRecordsByUID = \
                        self.runMoves_Shuffle(
                            hmodel, SS, Lscore, MovePlans,
                            MoveLog=MoveLog,
                            MoveRecordsByUID=MoveRecordsByUID,
                            lapFrac=lapFrac,)

            nLapsCompleted = lapFrac - self.algParams['startLap']
            if nLapsCompleted > 1.0:
                # Lscore increases monotonically AFTER first lap
                # verify_evidence warns if this isn't happening
                self.verify_evidence(Lscore, prevLscore, lapFrac)

            # Debug
            if self.doDebug() and lapFrac >= 1.0:
                self.verifyELBOTracking(hmodel, SS, Lscore, MoveLog)
            self.saveDebugStateAtBatch(
                'Mstep', batchID, SSchunk=SSbatch, SS=SS, hmodel=hmodel)

            # Assess convergence
            countVec = SS.getCountVec()
            if nLapsCompleted > 1.0:
                ConvStatus[batchID] = self.isCountVecConverged(
                    countVec, prevCountVec, batchID=batchID)
                isConverged = np.min(ConvStatus) and not \
                    self.hasMoreReasonableMoves(SS, MoveRecordsByUID, lapFrac)
                self.setStatus(lapFrac, isConverged)

            # Display progress
            if self.isLogCheckpoint(lapFrac, iterid):
                self.printStateToLog(hmodel, Lscore, lapFrac, iterid)

            # Save diagnostics and params
            if self.isSaveDiagnosticsCheckpoint(lapFrac, iterid):
                self.saveDiagnostics(lapFrac, SS, Lscore)
            if self.isSaveParamsCheckpoint(lapFrac, iterid):
                self.saveParams(lapFrac, hmodel, SS)

            # Custom func hook
            self.eval_custom_func(**makeDictOfAllWorkspaceVars(**vars()))

            if self.isLastBatch(lapFrac):
                ElapsedTimeLogger.writeToLogOnLapCompleted(lapFrac)

                if isConverged and \
                    nLapsCompleted >= self.algParams['minLaps']:
                    break
            prevCountVec = countVec.copy()
            prevLscore = Lscore
            # .... end loop over data

        # Finished! Save, print and exit
        self.printStateToLog(hmodel, Lscore, lapFrac, iterid, isFinal=1)
        self.saveParams(lapFrac, hmodel, SS)
        self.eval_custom_func(
            isFinal=1, **makeDictOfAllWorkspaceVars(**vars()))

        # Births and merges require copies of original model object
        #  we need to make sure original reference has updated parameters, etc.
        if id(origmodel) != id(hmodel):
            origmodel.allocModel = hmodel.allocModel
            origmodel.obsModel = hmodel.obsModel

        # Return information about this run
        return self.buildRunInfo(DataIterator, evBound=Lscore, SS=SS,
                                 SSmemory=self.SSmemory)

    def calcLocalParamsAndSummarize_withExpansionMoves(
            self, DataIterator, curModel,
            SS=None,
            batchID=0,
            lapFrac=0,
            MovePlans=None,
            MoveRecordsByUID=dict(),
            MoveLog=None,
            **kwargs):
        ''' Execute local step and summary step, with expansion proposals.

        Returns
        -------
        SSbatch : bnpy.suffstats.SuffStatBag
        '''
        # Fetch the current batch of data
        Dbatch = DataIterator.getBatch(batchID=batchID)
        # Prepare the kwargs for the local and summary steps
        # including args for the desired merges/deletes/etc.
        if not isinstance(MovePlans, dict):
            MovePlans = dict()
        LPkwargs = self.algParamsLP
        # MovePlans indicates which merge pairs to track in local step.
        LPkwargs.update(MovePlans)
        trackDocUsage = 0
        if self.hasMove('birth'):
            if self.algParams['birth']['b_debugWriteHTML']:
                trackDocUsage = 1
        # Do the real work here: calc local params
        # Pass lap and batch info so logging happens
        ElapsedTimeLogger.startEvent('local', 'update')
        LPbatch = curModel.calc_local_params(Dbatch, 
            lapFrac=lapFrac, batchID=batchID, **LPkwargs)
        ElapsedTimeLogger.stopEvent('local', 'update')
        # Summary time!
        ElapsedTimeLogger.startEvent('local', 'summary')
        SSbatch = curModel.get_global_suff_stats(
            Dbatch, LPbatch,
            doPrecompEntropy=1,
            doTrackTruncationGrowth=1,
            trackDocUsage=trackDocUsage,
            **MovePlans)
        if 'm_UIDPairs' in MovePlans:
            SSbatch.setMergeUIDPairs(MovePlans['m_UIDPairs'])
        ElapsedTimeLogger.stopEvent('local', 'summary')

        # Prepare current snapshot of whole-dataset stats
        # These must reflect the latest assignment to this batch,
        # AND all previous batches
        if SS is None:
            curSSwhole = SSbatch.copy()
        else:
            SSbatch.setUIDs(SS.uids)
            curSSwhole = SS.copy(includeELBOTerms=1, includeMergeTerms=0)
            curSSwhole += SSbatch
            if lapFrac > 1.0:
                oldSSbatch = self.loadBatchAndFastForward(
                    batchID, lapFrac, MoveLog, doCopy=1)
                curSSwhole -= oldSSbatch

        # Prepare plans for which births to try,
        # using recently updated stats.
        if self.hasMove('birth'):
            ElapsedTimeLogger.startEvent('birth', 'plan')
            MovePlans = self.makeMovePlans_Birth_AtBatch(
                curModel, curSSwhole,
                SSbatch=SSbatch,
                lapFrac=lapFrac, 
                MovePlans=MovePlans,
                MoveRecordsByUID=MoveRecordsByUID,
                **kwargs)
            ElapsedTimeLogger.stopEvent('birth', 'plan')

        # Prepare some logging stats        
        batchPos = np.round((lapFrac - np.floor(lapFrac)) / self.lapFracInc)
        if 'b_nFailedProp' not in MovePlans:
            MovePlans['b_nFailedProp'] = 0
        if 'b_nTrial' not in MovePlans:
            MovePlans['b_nTrial'] = 0

        # Create a place to store each proposal, indexed by UID
        SSbatch.propXSS = dict()
        # Try each planned birth
        if 'b_targetUIDs' in MovePlans and len(MovePlans['b_targetUIDs']) > 0:
            ElapsedTimeLogger.startEvent('birth', 'localexpansion')
            newUIDs = self.makeNewUIDs(
                nMoves=len(MovePlans['b_targetUIDs']),
                **self.algParams['birth'])
            SSbatch.propXSS, MovePlans, MoveRecordsByUID = \
                 makeSummariesForManyBirthProposals(
                    Dslice=Dbatch,
                    curModel=curModel,
                    curLPslice=LPbatch,
                    curSSwhole=curSSwhole,
                    curSSslice=SSbatch,
                    LPkwargs=LPkwargs,
                    newUIDs=newUIDs,
                    MovePlans=MovePlans,
                    MoveRecordsByUID=MoveRecordsByUID,
                    taskoutpath=self.savedir,
                    lapFrac=lapFrac,
                    seed=self.seed,
                    nBatch=self.nBatch,
                    batchPos=batchPos,
                    **self.algParams['birth'])
            ElapsedTimeLogger.stopEvent('birth', 'localexpansion')


        # Prepare deletes
        if 'd_targetUIDs' in MovePlans:
            ElapsedTimeLogger.startEvent('delete', 'localexpansion')
            targetUID = MovePlans['d_targetUIDs'][0]
            # Make copy of current suff stats (minus target state)
            # to inspire reclustering of junk state.
            propRemSS = curSSwhole.copy(
                includeELBOTerms=False, includeMergeTerms=False)
            for uid in propRemSS.uids:
                if uid not in MovePlans['d_absorbingUIDSet']:
                    propRemSS.removeComp(uid=uid)
            # Give each absorbing UID a termporary new UID
            oldUIDs = propRemSS.uids.copy()
            if self.isFirstBatch(lapFrac):
                newUIDs = self.makeNewUIDs(b_Kfresh=len(oldUIDs))
            else:
                assert targetUID in SS.propXSS
                newUIDs = SS.propXSS[targetUID].uids
            propRemSS.setUIDs(newUIDs)
            mUIDPairs = list()
            for ii, oldUID in enumerate(oldUIDs):
                mUIDPairs.append((oldUID, newUIDs[ii]))

            # # For topic models, consider inflating some word counts
            # # for the temporary suff stats used to do restricted steps
            '''
            absorbingIDs = np.sort([curSSwhole.uid2k(uid)
                    for uid in MovePlans['d_absorbingUIDSet']])

            if curModel.getAllocModelName().count('HDPTopic'):
                if curModel.getObsModelName().count('Mult'):
                    ktarget = curSSwhole.uid2k(targetUID)
                    topTargetWords = np.flatnonzero(
                        curSSwhole.WordCounts[ktarget,:] > 5)
                    # TODO inflate with topTargetWords on other topics??
            '''
            # Run restricted local step
            SSbatch.propXSS[targetUID], rInfo = summarizeRestrictedLocalStep(
                Dbatch, curModel, LPbatch, 
                curSSwhole=curSSwhole,
                xInitSS=propRemSS,
                xUIDs=newUIDs,
                targetUID=targetUID,
                mUIDPairs=mUIDPairs,
                LPkwargs=LPkwargs,
                emptyPiFrac=0,
                lapFrac=lapFrac)
            ElapsedTimeLogger.stopEvent('delete', 'localexpansion')
        return SSbatch

    def incrementWholeDataSummary(
            self, SS, SSbatch, oldSSbatch,
            hmodel=None,
            lapFrac=0):
        ''' Update whole dataset sufficient stats object.

        Returns
        -------
        SS : SuffStatBag
            represents whole dataset seen thus far.
        '''
        ElapsedTimeLogger.startEvent('global', 'increment')
        if SS is None:
            SS = SSbatch.copy()
        else:
            if oldSSbatch is not None:
                SS -= oldSSbatch
            SS += SSbatch
            if hasattr(SSbatch, 'propXSS'):
                if not hasattr(SS, 'propXSS'):
                    SS.propXSS = dict()

                for uid in SSbatch.propXSS:
                    if uid in SS.propXSS:
                        SS.propXSS[uid] += SSbatch.propXSS[uid]
                    else:
                        SS.propXSS[uid] = SSbatch.propXSS[uid].copy()
        # Force aggregated suff stats to obey required constraints.
        # This avoids numerical issues caused by incremental updates
        if hmodel is not None:
            if hasattr(hmodel.allocModel, 'forceSSInBounds'):
                hmodel.allocModel.forceSSInBounds(SS)
            if hasattr(hmodel.obsModel, 'forceSSInBounds'):
                hmodel.obsModel.forceSSInBounds(SS)
        ElapsedTimeLogger.stopEvent('global', 'increment')
        return SS

    def loadBatchAndFastForward(self, batchID, lapFrac, MoveLog, doCopy=0):
        ''' Retrieve batch from memory, and apply any relevant moves to it.

        Returns
        -------
        oldSSbatch : SuffStatBag, or None if specified batch not in memory.

        Post Condition
        --------------
        LastUpdateLap attribute will indicate batchID was updated at lapFrac,
        unless working with a copy not raw memory (doCopy=1).
        '''
        ElapsedTimeLogger.startEvent('global', 'loadbatch')
        try:
            SSbatch = self.SSmemory[batchID]
        except KeyError:
            return None

        if doCopy:
            SSbatch = SSbatch.copy()

        for (lap, op, kwargs, beforeUIDs, afterUIDs) in MoveLog:
            if lap < self.LastUpdateLap[batchID]:
                continue
            assert np.allclose(SSbatch.uids, beforeUIDs)
            if op == 'merge':
                SSbatch.mergeComps(**kwargs)
            elif op == 'shuffle':
                SSbatch.reorderComps(kwargs['bigtosmallorder'])
            elif op == 'prune':
                for uid in kwargs['emptyCompUIDs']:
                    SSbatch.removeComp(uid=uid)
            elif op == 'birth':
                targetUID = kwargs['targetUID']
                hasStoredProposal = hasattr(SSbatch, 'propXSS') and \
                    targetUID in SSbatch.propXSS
                if hasStoredProposal:
                    cur_newUIDs = SSbatch.propXSS[targetUID].uids
                    expected_newUIDs = np.setdiff1d(afterUIDs, beforeUIDs)
                    sameSize = cur_newUIDs.size == expected_newUIDs.size
                    if sameSize and np.all(cur_newUIDs == expected_newUIDs):
                        SSbatch.transferMassFromExistingToExpansion(
                           uid=targetUID, xSS=SSbatch.propXSS[targetUID])
                    else:
                        hasStoredProposal = False

                if not hasStoredProposal:
                    Kfresh = afterUIDs.size - beforeUIDs.size
                    SSbatch.insertEmptyComps(Kfresh)
                    SSbatch.setUIDs(afterUIDs)
            elif op == 'delete':
                SSbatch.removeMergeTerms()
                targetUID = kwargs['targetUID']
                hasStoredProposal = hasattr(SSbatch, 'propXSS') and \
                    targetUID in SSbatch.propXSS
                assert hasStoredProposal
                SSbatch.replaceCompWithExpansion(
                    uid=targetUID, xSS=SSbatch.propXSS[targetUID])
                for (uidA, uidB) in SSbatch.mUIDPairs:
                    SSbatch.mergeComps(uidA=uidA, uidB=uidB)
            else:
                raise NotImplementedError("TODO")
            assert np.allclose(SSbatch.uids, afterUIDs)
        # Discard merge terms, since all accepted merges have been incorporated
        SSbatch.removeMergeTerms()
        if not doCopy:
            self.LastUpdateLap[batchID] = lapFrac
        ElapsedTimeLogger.stopEvent('global', 'loadbatch')
        return SSbatch

    def globalStep(self, hmodel, SS, lapFrac):
        ''' Do global update, if appropriate at current lap.

        Post Condition
        ---------
        hmodel global parameters updated in place.
        '''
        ElapsedTimeLogger.startEvent('global', 'update')
        doFullPass = self.algParams['doFullPassBeforeMstep']
        if self.algParams['doFullPassBeforeMstep'] == 1:
            if lapFrac >= 1.0:
                hmodel.update_global_params(SS)
        elif doFullPass > 1.0:
            if lapFrac >= 1.0 or (doFullPass < SS.nDoc):
                # update if we've seen specified num of docs, not before
                hmodel.update_global_params(SS)
        else:
            hmodel.update_global_params(SS)
        ElapsedTimeLogger.stopEvent('global', 'update')
        return hmodel

    def makeMovePlans(self, hmodel, SS,
                      MovePlans=dict(),
                      MoveRecordsByUID=dict(), 
                      lapFrac=-1,
                      **kwargs):
        ''' Plan which comps to target for each possible move.

        Returns
        -------
        MovePlans : dict
        '''
        isFirst = self.isFirstBatch(lapFrac)
        if isFirst:
            MovePlans = dict()
        if isFirst and self.hasMove('birth'):
           ElapsedTimeLogger.startEvent('birth', 'plan')
           MovePlans = self.makeMovePlans_Birth_AtLapStart(
               hmodel, SS, 
               lapFrac=lapFrac,
               MovePlans=MovePlans,
               MoveRecordsByUID=MoveRecordsByUID,
               **kwargs)
           ElapsedTimeLogger.stopEvent('birth', 'plan')
        if isFirst and self.hasMove('merge'):
            ElapsedTimeLogger.startEvent('merge', 'plan')
            MovePlans = self.makeMovePlans_Merge(
                hmodel, SS, 
                lapFrac=lapFrac,
                MovePlans=MovePlans,
                MoveRecordsByUID=MoveRecordsByUID,
                **kwargs)
            ElapsedTimeLogger.stopEvent('merge', 'plan')
        if isFirst and self.hasMove('delete'):
            ElapsedTimeLogger.startEvent('delete', 'plan')
            MovePlans = self.makeMovePlans_Delete(
                hmodel, SS, 
                lapFrac=lapFrac,
                MovePlans=MovePlans,
                MoveRecordsByUID=MoveRecordsByUID,
                **kwargs)
            ElapsedTimeLogger.stopEvent('delete', 'plan')
        return MovePlans

    def makeMovePlans_Merge(self, hmodel, SS,
                            MovePlans=dict(),
                            MoveRecordsByUID=dict(),
                            lapFrac=0,
                            **kwargs):
        ''' Plan out which merges to attempt in current lap.

        Returns
        -------
        MovePlans : dict
            * m_UIDPairs : list of pairs of uids to merge
        '''
        ceilLap = np.ceil(lapFrac)
        if SS is None:
            msg = "MERGE @ lap %.2f: Disabled." + \
                " Cannot plan merge on first lap." + \
                " Need valid SS that represent whole dataset."
            MLogger.pprint(msg % (ceilLap), 'info')
            return MovePlans

        startLap = self.algParams['merge']['m_startLap']
        if np.ceil(lapFrac) < startLap:
            msg = "MERGE @ lap %.2f: Disabled." + \
                " Waiting for lap >= %d (--m_startLap)."
            MLogger.pprint(msg % (ceilLap, startLap), 'info')
            return MovePlans
        stopLap = self.algParams['merge']['m_stopLap']
        if stopLap > 0 and np.ceil(lapFrac) >= stopLap:
            msg = "MERGE @ lap %.2f: Disabled." + \
                " Beyond lap %d (--m_stopLap)."
            MLogger.pprint(msg % (ceilLap, stopLap), 'info')
            return MovePlans

        MArgs = self.algParams['merge']
        MPlan = selectCandidateMergePairs(
            hmodel, SS,
            MovePlans=MovePlans,
            MoveRecordsByUID=MoveRecordsByUID,
            lapFrac=lapFrac,
            **MArgs)
        # Do not track m_UIDPairs field unless it is non-empty
        if len(MPlan['m_UIDPairs']) < 1:
            del MPlan['m_UIDPairs']
            del MPlan['mPairIDs']
            msg = "MERGE @ lap %.2f: Ineligible." + \
                " No promising candidates."
            MLogger.pprint(msg % (ceilLap), 'info')

        else:
            MPlan['doPrecompMergeEntropy'] = 1
        MovePlans.update(MPlan)
        return MovePlans

    def makeMovePlans_Delete(self, hmodel, SS,
                            MovePlans=dict(),
                            MoveRecordsByUID=dict(),
                            lapFrac=0,
                            **kwargs):
        ''' Plan out which deletes to attempt in current lap.

        Returns
        -------
        MovePlans : dict
            * d_targetUIDs : list of uids to delete
        '''
        ceilLap = np.ceil(lapFrac)
        if SS is None:
            msg = "DELETE @ lap %.2f: Disabled." + \
                " Cannot delete before first complete lap," + \
                " because SS that represents whole dataset is required."
            DLogger.pprint(msg % (ceilLap), 'info')
            return MovePlans

        startLap = self.algParams['delete']['d_startLap']
        if ceilLap < startLap:
            msg = "DELETE @ lap %.2f: Disabled." + \
                " Waiting for lap >= %d (--d_startLap)."
            DLogger.pprint(msg % (ceilLap, startLap), 'info')
            return MovePlans
        stopLap = self.algParams['delete']['d_stopLap']
        if stopLap > 0 and ceilLap >= stopLap:
            msg = "DELETE @ lap %.2f: Disabled." + \
                " Beyond lap %d (--d_stopLap)."
            DLogger.pprint(msg % (ceilLap, stopLap), 'info')
            return MovePlans

        if self.hasMove('birth'):
            BArgs = self.algParams['birth']
        else:
            BArgs = dict()
        DArgs = self.algParams['delete']
        DArgs.update(BArgs)
        DPlan = selectCandidateDeleteComps(
            hmodel, SS,
            MovePlans=MovePlans,
            MoveRecordsByUID=MoveRecordsByUID,
            lapFrac=lapFrac,
            **DArgs)
        if 'failMsg' in DPlan:
            DLogger.pprint(
                'DELETE @ lap %.2f: %s' % (ceilLap, DPlan['failMsg']),
                'info')
        else:
            MovePlans.update(DPlan)
        return MovePlans

    def makeMovePlans_Birth_AtLapStart(
            self, hmodel, SS,
            MovePlans=dict(),
            MoveRecordsByUID=dict(),
            lapFrac=-2,
            batchID=-1,
            **kwargs):
        ''' Select comps to target with birth at start of current lap.

        Returns
        -------
        MovePlans : dict
            * b_shortlistUIDs : list of uids (ints) off limits to other moves.
        '''
        ceilLap = np.ceil(lapFrac)
        startLap = self.algParams['birth']['b_startLap']
        stopLap = self.algParams['birth']['b_stopLap']

        assert self.isFirstBatch(lapFrac)

        if ceilLap < startLap:
            msg = "BIRTH @ lap %.2f: Disabled." + \
                " Waiting for lap >= %d (--b_startLap)."
            BLogger.pprint(msg % (ceilLap, startLap), 'info')
            return MovePlans
        if stopLap > 0 and ceilLap >= stopLap:
            msg = "BIRTH @ lap %.2f: Disabled." + \
                " Beyond lap %d (--b_stopLap)."
            BLogger.pprint(msg % (ceilLap, stopLap), 'info')
            return MovePlans

        BArgs = self.algParams['birth']    
        msg = "PLANNING birth shortlist at lap %.3f"
        BLogger.pprint(msg % (lapFrac))
        MovePlans = selectShortListForBirthAtLapStart(
            hmodel, SS,
            MoveRecordsByUID=MoveRecordsByUID,
            MovePlans=MovePlans,
            lapFrac=lapFrac,
            **BArgs)
        assert 'b_shortlistUIDs' in MovePlans
        assert isinstance(MovePlans['b_shortlistUIDs'], list)
        return MovePlans


    def makeMovePlans_Birth_AtBatch(
            self, hmodel, SS,
            SSbatch=None,
            MovePlans=dict(),
            MoveRecordsByUID=dict(),
            lapFrac=-2,
            batchID=0,
            **kwargs):
        ''' Select comps to target with birth at current batch.

        Returns
        -------
        MovePlans : dict
            * b_targetUIDs : list of uids (ints) indicating comps to target
        '''
        ceilLap = np.ceil(lapFrac)
        startLap = self.algParams['birth']['b_startLap']
        stopLap = self.algParams['birth']['b_stopLap']

        if ceilLap < startLap:
            return MovePlans
        if stopLap > 0 and ceilLap >= stopLap:
            return MovePlans

        if self.hasMove('birth'):
            BArgs = self.algParams['birth']    
            msg = "PLANNING birth at lap %.3f, batch %d"
            BLogger.pprint(msg % (lapFrac, batchID))
            MovePlans = selectCompsForBirthAtCurrentBatch(
                hmodel, SS,
                SSbatch=SSbatch,
                MoveRecordsByUID=MoveRecordsByUID,
                MovePlans=MovePlans,
                lapFrac=lapFrac,
                **BArgs)
            if 'b_targetUIDs' in MovePlans:
                assert isinstance(MovePlans['b_targetUIDs'], list)
        return MovePlans

    def runMoves_Birth(self, hmodel, SS, Lscore, MovePlans,
                       MoveLog=list(),
                       MoveRecordsByUID=dict(),
                       lapFrac=0,
                       **kwargs):
        ''' Execute planned birth/split moves.

        Returns
        -------
        hmodel
        SS
        Lscore
        MoveLog
        MoveRecordsByUID
        '''
        ElapsedTimeLogger.startEvent('birth', 'eval')
        if 'b_targetUIDs' in MovePlans and len(MovePlans['b_targetUIDs']) > 0:
            b_targetUIDs = [u for u in MovePlans['b_targetUIDs']]
            BLogger.pprint(
                'EVALUATING birth proposals at lap %.2f' % (lapFrac))
            MovePlans['b_retainedUIDs'] = list()
        else:
            b_targetUIDs = list()

        if 'b_nFailedEval' in MovePlans:
            nFailedEval = MovePlans['b_nFailedEval']
        else:
            nFailedEval = 0
        if 'b_nAccept' in MovePlans:
            nAccept = MovePlans['b_nAccept']
        else:
            nAccept = 0
        if 'b_nTrial' in MovePlans:
            nTrial = MovePlans['b_nTrial']
        else:
            nTrial = 0
        if 'b_Knew' in MovePlans:
            totalKnew = MovePlans['b_Knew']
        else:
            totalKnew = 0
        nRetainedForNextLap = 0
        acceptedUIDs = list()
        curLdict = hmodel.calc_evidence(SS=SS, todict=1)
        for targetUID in b_targetUIDs:
            # Skip delete proposals, which are handled differently
            if 'd_targetUIDs' in MovePlans:
                if targetUID in MovePlans['d_targetUIDs']:
                    raise ValueError("WHOA! Cannot delete and birth same uid.")
            nTrial += 1

            BLogger.startUIDSpecificLog(targetUID)
            # Prepare record-keeping            
            if targetUID not in MoveRecordsByUID:
                MoveRecordsByUID[targetUID] = defaultdict(int)
            ktarget = SS.uid2k(targetUID)
            targetCount = SS.getCountVec()[ktarget]
            MoveRecordsByUID[targetUID]['b_nTrial'] += 1
            MoveRecordsByUID[targetUID]['b_latestLap'] = lapFrac
            MoveRecordsByUID[targetUID]['b_latestCount'] = targetCount
            MoveRecordsByUID[targetUID]['b_latestBatchCount'] = \
                SS.propXSS[targetUID].getCountVec().sum()
            # Construct proposal statistics
            BLogger.pprint(
                'Evaluating targetUID %d at lap %.2f' % (
                    targetUID, lapFrac))
            propSS = SS.copy()
            propSS.transferMassFromExistingToExpansion(
                uid=targetUID, xSS=SS.propXSS[targetUID])
            # Create model via global step from proposed stats
            propModel = hmodel.copy()
            propModel.update_global_params(propSS)
            # Compute score of proposal
            propLdict = propModel.calc_evidence(SS=propSS, todict=1)
            propLscore = propLdict['Ltotal']
            msg = "   gainL % .3e" % (propLscore-Lscore)
            msg += "\n    curL % .3e" % (Lscore)
            msg += "\n   propL % .3e" % (propLscore)
            for key in sorted(curLdict.keys()):
                if key.count('_') or key.count('total'):
                    continue
                msg += "\n   gain_%8s % .3e" % (
                    key, propLdict[key] - curLdict[key])
            BLogger.pprint(msg)
            assert propLdict['Lentropy'] >= - 1e-6
            assert curLdict['Lentropy'] >= - 1e-6
            assert propLdict['Lentropy'] >= curLdict['Lentropy'] - 1e-6
            if propLscore > Lscore:
                nAccept += 1
                BLogger.pprint(
                    '   Accepted. Jump up to Lscore % .3e ' % (propLscore))
                BLogger.pprint(
                    "    Mass transfered to new comps: %.2f" % (
                        SS.getCountVec()[ktarget] - \
                            propSS.getCountVec()[ktarget]))
                BLogger.pprint(
                    "    Remaining mass at targetUID %d: %.2f" % (
                        targetUID, propSS.getCountVec()[ktarget]))
                totalKnew += propSS.K - SS.K
                MoveRecordsByUID[targetUID]['b_nSuccess'] += 1
                MoveRecordsByUID[targetUID]['b_nFailRecent'] = 0
                MoveRecordsByUID[targetUID]['b_nSuccessRecent'] += 1
                MoveRecordsByUID[targetUID]['b_latestLapAccept'] = lapFrac
                # Write necessary information to the log
                MoveArgs = dict(targetUID=targetUID,
                                newUIDs=SS.propXSS[targetUID].uids)
                infoTuple = (
                    lapFrac, 'birth', MoveArgs,
                    SS.uids.copy(), propSS.uids.copy())
                MoveLog.append(infoTuple)
                # Set proposal values as new "current" values
                hmodel = propModel
                Lscore = propLscore
                SS = propSS
                curLdict = propLdict
                MovePlans['b_targetUIDs'].remove(targetUID)
                del SS.propXSS[targetUID]
            else:
                nSubset = SS.propXSS[targetUID].getCountVec().sum()
                nTotal = SS.getCountVec()[ktarget]
                if nSubset + 1.0 < nTotal and self.nBatch > 1:
                    couldUseMoreData = True
                else:
                    couldUseMoreData = False

                BLogger.pprint(
                    '   Rejected. Remain at Lscore %.3e' % (Lscore))
                gainLdata = propLdict['Ldata'] - curLdict['Ldata']

                if couldUseMoreData:
                    # Route to redemption #2:
                    # If Ldata for subset of data reassigned so far looks good
                    # we hold onto this proposal for next time! 
                    propSSsubset = SS.propXSS[targetUID].copy(
                        includeELBOTerms=False, includeMergeTerms=False)
                    tmpModel = propModel
                    tmpModel.obsModel.update_global_params(propSSsubset)
                    propLdata_subset = tmpModel.obsModel.calcELBO_Memoized(
                        propSSsubset)
                    curSSsubset = propSSsubset
                    while curSSsubset.K > 1:
                        curSSsubset.mergeComps(0, 1)
                    tmpModel.obsModel.update_global_params(curSSsubset)
                    curLdata_subset = tmpModel.obsModel.calcELBO_Memoized(
                        curSSsubset)
                    gainLdata_subset = propLdata_subset - curLdata_subset
                else:
                    gainLdata_subset = -42.0

                if gainLdata_subset > 1e-6 and not self.isLastBatch(lapFrac):
                    nTrial -= 1
                    BLogger.pprint(
                        '   Retained. Promising gainLdata_subset % .2f' % (
                            gainLdata_subset))
                    assert targetUID in SS.propXSS
                    MovePlans['b_retainedUIDs'].append(targetUID)

                elif gainLdata > 1e-6 and not self.isLastBatch(lapFrac):
                    nTrial -= 1
                    BLogger.pprint(
                        '   Retained. Promising value of gainLdata % .2f' % (
                            gainLdata))
                    assert targetUID in SS.propXSS
                    MovePlans['b_retainedUIDs'].append(targetUID)
                elif gainLdata_subset > 1e-6 and \
                        self.isLastBatch(lapFrac) and couldUseMoreData:
                    nRetainedForNextLap += 1
                    BLogger.pprint(
                        '   Retain uid %d next lap! gainLdata_subset %.3e' % (
                            targetUID, gainLdata_subset))
                    assert targetUID in SS.propXSS
                    MoveRecordsByUID[targetUID]['b_tryAgainFutureLap'] = 1
                    MovePlans['b_retainedUIDs'].append(targetUID)
                else:
                    nFailedEval += 1
                    MovePlans['b_targetUIDs'].remove(targetUID)
                    del SS.propXSS[targetUID]
                    MoveRecordsByUID[targetUID]['b_nFail'] += 1
                    MoveRecordsByUID[targetUID]['b_nFailRecent'] += 1
                    MoveRecordsByUID[targetUID]['b_nSuccessRecent'] = 0
                    MoveRecordsByUID[targetUID]['b_tryAgainFutureLap'] = 0

            BLogger.stopUIDSpecificLog(targetUID)

        if 'b_retainedUIDs' in MovePlans:
            assert np.allclose(MovePlans['b_retainedUIDs'],
                MovePlans['b_targetUIDs'])
            for uid in MovePlans['b_targetUIDs']:
                assert uid in SS.propXSS
        MovePlans['b_Knew'] = totalKnew
        MovePlans['b_nAccept'] = nAccept
        MovePlans['b_nTrial'] = nTrial
        MovePlans['b_nFailedEval'] = nFailedEval
        if self.isLastBatch(lapFrac):
            if nTrial > 0:
                msg = "BIRTH @ lap %.2f : Added %d states." + \
                    " %d/%d succeeded. %d/%d failed eval phase. " + \
                    "%d/%d failed build phase."
                msg = msg % (
                    lapFrac, totalKnew, 
                    nAccept, nTrial,
                    MovePlans['b_nFailedEval'], nTrial,
                    MovePlans['b_nFailedProp'], nTrial)
                if nRetainedForNextLap > 0:
                    msg += " %d retained!" % (nRetainedForNextLap)
                BLogger.pprint(msg, 'info')
            elif 'b_shortlistUIDs' in MovePlans:
                # Birth was eligible, but did not make it to eval stage.
                if len(MovePlans['b_shortlistUIDs']) > 0:
                    msg = "BIRTH @ lap %.2f : No proposals attempted." + \
                        " Shortlist had %d possible clusters," + \
                        " but none met minimum requirements."
                    msg = msg % (
                        lapFrac, len(MovePlans['b_shortlistUIDs']))
                    BLogger.pprint(msg, 'info')
                else:
                    msg = "BIRTH @ lap %.2f : No shortlist."
                    msg +=  " Could have added %d clusters this lap."
                    msg +=  " But %d too small. %d had past rejections."
                    msg = msg % (
                        lapFrac,
                        MovePlans['b_roomToGrow'],
                        MovePlans['b_nDQ_toosmall'],
                        MovePlans['b_nDQ_pastfail'])
                    BLogger.pprint(msg, 'info')
            else:
                pass

            # If any short-listed uids did not get tried in this lap
            # there are two possible reasons:
            # 1) No batch contains a sufficient size of that uid.
            # 2) Other uids were prioritized due to budget constraints.
            # We need to mark uids that failed for reason 1,
            # so that we don't avoid deleting/merging them in the future.
            if 'b_shortlistUIDs' in MovePlans:
                for uid in MovePlans['b_shortlistUIDs']:
                    if uid not in MoveRecordsByUID:
                        MoveRecordsByUID[uid] = defaultdict(int)
                    Rec = MoveRecordsByUID[uid]

                    lastEligibleLap = Rec['b_latestEligibleLap']
                    if np.ceil(lastEligibleLap) < np.ceil(lapFrac):
                        msg = "Marked uid %d ineligible for future shortlists."
                        msg += " It was never eligible this lap."
                        BLogger.pprint(msg % (uid))
                        k = SS.uid2k(uid)
                        Rec['b_latestLap'] = lapFrac
                        Rec['b_nFail'] += 1
                        Rec['b_nFailRecent'] += 1
                        Rec['b_nSuccessRecent'] = 0
                        Rec['b_latestCount'] = SS.getCountVec()[k]
                        Rec['b_latestBatchCount'] = \
                            self.SSmemory[0].getCountVec()[k]

        ElapsedTimeLogger.stopEvent('birth', 'eval')
        return hmodel, SS, Lscore, MoveLog, MoveRecordsByUID

    def runMoves_Merge(self, hmodel, SS, Lscore, MovePlans,
                       MoveLog=list(),
                       MoveRecordsByUID=dict(),
                       lapFrac=0,
                       **kwargs):
        ''' Execute planned merge moves.

        Returns
        -------
        hmodel
        SS : SuffStatBag
            Contains updated fields and ELBO terms for K-Kaccepted comps.
            All merge terms will be set to zero.
        Lscore
        MoveLog
        MoveRecordsByUID
        '''
        ElapsedTimeLogger.startEvent('merge', 'eval')
        acceptedUIDs = set()
        nTrial = 0
        nAccept = 0
        nSkip = 0
        Ndiff = 0.0
        MLogger.pprint("EVALUATING merges at lap %.2f" % (
            lapFrac), 'debug')
        for ii, (uidA, uidB) in enumerate(MovePlans['m_UIDPairs']):
            # Skip uids that we have already accepted in a previous merge.
            if uidA in acceptedUIDs or uidB in acceptedUIDs:
                nSkip += 1
                MLogger.pprint("%4d, %4d : skipped." % (
                    uidA, uidB), 'debug')
                continue
            nTrial += 1           
            # Update records for when each uid was last attempted
            for u in [uidA, uidB]:
                if u not in MoveRecordsByUID:
                    MoveRecordsByUID[u] = defaultdict(int)

                targetCount = SS.getCountVec()[SS.uid2k(u)]
                MoveRecordsByUID[u]['m_nTrial'] += 1
                MoveRecordsByUID[u]['m_latestLap'] = lapFrac
                MoveRecordsByUID[u]['m_latestCount'] = targetCount
            propSS = SS.copy()
            propSS.mergeComps(uidA=uidA, uidB=uidB)
            propModel = hmodel.copy()
            propModel.update_global_params(propSS)
            propLscore = propModel.calc_evidence(SS=propSS)
            assert np.isfinite(propLscore)

            propSizeStr = count2str(propSS.getCountForUID(uidA))
            if propLscore > Lscore - ELBO_GAP_ACCEPT_TOL:
                nAccept += 1
                Ndiff += targetCount
                MLogger.pprint(
                    "%4d, %4d : accepted." % (uidA, uidB) +
                    " gain %.3e  " % (propLscore - Lscore) +
                    " size %s  " % (propSizeStr),
                    'debug')

                acceptedUIDs.add(uidA)
                acceptedUIDs.add(uidB)
                MoveRecordsByUID[uidA]['m_nSuccess'] += 1
                MoveRecordsByUID[uidA]['m_nSuccessRecent'] += 1
                MoveRecordsByUID[uidA]['m_nFailRecent'] = 0
                MoveRecordsByUID[uidA]['m_latestLapAccept'] = lapFrac
                # Write necessary information to the log
                MoveArgs = dict(uidA=uidA, uidB=uidB)
                infoTuple = (lapFrac, 'merge', MoveArgs,
                             SS.uids.copy(), propSS.uids.copy())
                MoveLog.append(infoTuple)
                # Set proposal values as new "current" values
                SS = propSS
                hmodel = propModel
                Lscore = propLscore
            else:
                MLogger.pprint(
                    "%4d, %4d : rejected." % (uidA, uidB) +
                    " gain %.3f  " % (propLscore - Lscore) +
                    " size %s  " % (propSizeStr),
                    'debug')

                for u in [uidA, uidB]:
                    MoveRecordsByUID[u]['m_nFail'] += 1
                    MoveRecordsByUID[u]['m_nFailRecent'] += 1
                    MoveRecordsByUID[u]['m_nSuccessRecent'] = 0
        if nTrial > 0:
            msg = "MERGE @ lap %.2f : %d/%d accepted." + \
                " Ndiff %.2f. %d skipped."
            msg = msg % (
                lapFrac, nAccept, nTrial, Ndiff, nSkip)
            MLogger.pprint(msg, 'info')
        # Finally, set all merge fields to zero,
        # since all possible merges have been accepted
        SS.removeMergeTerms()
        assert not hasattr(SS, 'M')
        ElapsedTimeLogger.stopEvent('merge', 'eval')
        return hmodel, SS, Lscore, MoveLog, MoveRecordsByUID

    def runMoves_Shuffle(self, hmodel, SS, Lscore, MovePlans,
                         MoveLog=list(),
                         MoveRecordsByUID=dict(),
                         lapFrac=0,
                         **kwargs):
        ''' Execute shuffle move, which need not be planned in advance.

        Returns
        -------
        hmodel
            Reordered copies of the K input states.
        SS : SuffStatBag
            Reordered copies of the K input states.
        Lscore
        MoveLog
        MoveRecordsByUID
        '''
        prevLscore = Lscore
        emptyCompLocs = np.flatnonzero(SS.getCountVec() < 0.001)
        emptyCompUIDs = [SS.uids[k] for k in emptyCompLocs]
        if emptyCompLocs.size > 0 and self.algParams['shuffle']['s_doPrune']:
            beforeUIDs = SS.uids.copy()
            for uid in emptyCompUIDs:
                SS.removeComp(uid=uid)
            afterUIDs = SS.uids.copy()
            moveTuple = (
                lapFrac, 'prune',
                dict(emptyCompUIDs=emptyCompUIDs),
                beforeUIDs,
                afterUIDs)
            MoveLog.append(moveTuple)

        if hasattr(SS, 'sumLogPi'):
            bigtosmallorder = argsort_bigtosmall_stable(SS.sumLogPi)
        else:
            bigtosmallorder = argsort_bigtosmall_stable(SS.getCountVec())
        sortedalready = np.arange(SS.K)
        if not np.allclose(bigtosmallorder, sortedalready):
            moveTuple = (
                lapFrac, 'shuffle',
                dict(bigtosmallorder=bigtosmallorder),
                SS.uids, SS.uids[bigtosmallorder])
            MoveLog.append(moveTuple)
            SS.reorderComps(bigtosmallorder)
            hmodel.update_global_params(SS, sortorder=bigtosmallorder)
            Lscore = hmodel.calc_evidence(SS=SS)
            # TODO Prevent shuffle if ELBO does not improve??
            SLogger.pprint(
                "SHUFFLED at lap %.3f." % (lapFrac) + \
                " Lgain % .4e   Lbefore % .4e   Lafter % .4e" % (
                    Lscore - prevLscore, prevLscore, Lscore))

        elif emptyCompLocs.size > 0:
            hmodel.update_global_params(SS)
            Lscore = hmodel.calc_evidence(SS=SS)

        return hmodel, SS, Lscore, MoveLog, MoveRecordsByUID


    def runMoves_Delete(self, hmodel, SS, Lscore, MovePlans,
                        MoveLog=list(),
                        MoveRecordsByUID=dict(),
                        lapFrac=0,
                        **kwargs):
        ''' Execute planned delete move.

        Returns
        -------
        hmodel
        SS
        Lscore
        MoveLog
        MoveRecordsByUID
        '''
        ElapsedTimeLogger.startEvent('delete', 'eval')

        if len(MovePlans['d_targetUIDs']) > 0:
            DLogger.pprint('EVALUATING delete @ lap %.2f' % (lapFrac))

        nAccept = 0
        nTrial = 0
        Ndiff = 0.0
        curLdict = hmodel.calc_evidence(SS=SS, todict=1)
        for targetUID in MovePlans['d_targetUIDs']:
            nTrial += 1
            assert targetUID in SS.propXSS
            # Prepare record keeping
            if targetUID not in MoveRecordsByUID:
                MoveRecordsByUID[targetUID] = defaultdict(int)
            targetCount = SS.getCountVec()[SS.uid2k(targetUID)]
            MoveRecordsByUID[targetUID]['d_nTrial'] += 1
            MoveRecordsByUID[targetUID]['d_latestLap'] = lapFrac
            MoveRecordsByUID[targetUID]['d_latestCount'] = targetCount
            # Construct proposed stats
            propSS = SS.copy()
            propSS.replaceCompWithExpansion(uid=targetUID,
                                            xSS=SS.propXSS[targetUID])
            for (uidA, uidB) in propSS.mUIDPairs:
                propSS.mergeComps(uidA=uidA, uidB=uidB)
            # Construct proposed model and its ELBO score
            propModel = hmodel.copy()
            propModel.update_global_params(propSS)
            propLdict = propModel.calc_evidence(SS=propSS, todict=1)
            propLscore = propLdict['Ltotal']
            msg = 'targetUID %d' % (targetUID)
            msg += '\n   gainL % .3e' % (propLscore-Lscore)
            msg += "\n    curL % .3e" % (Lscore)
            msg += "\n   propL % .3e" % (propLscore)
            for key in sorted(curLdict.keys()):
                if key.count('_') or key.count('total'):
                    continue
                msg += "\n   gain_%8s % .3e" % (
                    key, propLdict[key] - curLdict[key])

            DLogger.pprint(msg)
            # Make decision
            if propLscore > Lscore:
                # Accept
                nAccept += 1
                Ndiff += targetCount
                MoveRecordsByUID[targetUID]['d_nFailRecent'] = 0
                MoveRecordsByUID[targetUID]['d_latestLapAccept'] = lapFrac
                # Write necessary information to the log
                MoveArgs = dict(targetUID=targetUID)
                infoTuple = (lapFrac, 'delete', MoveArgs,
                             SS.uids.copy(), propSS.uids.copy())
                MoveLog.append(infoTuple)
                # Set proposal values as new "current" values
                hmodel = propModel
                Lscore = propLscore
                SS = propSS
                curLdict = propLdict
            else:
                # Reject!
                MoveRecordsByUID[targetUID]['d_nFail'] += 1
                MoveRecordsByUID[targetUID]['d_nFailRecent'] += 1
            # Always cleanup evidence of the proposal
            del SS.propXSS[targetUID]

        if nTrial > 0:
            msg = 'DELETE @ lap %.2f: %d/%d accepted. Ndiff %.2f.' % (
                lapFrac, nAccept, nTrial, Ndiff)
            DLogger.pprint(msg, 'info')
        # Discard plans, because they have come to fruition.
        for key in MovePlans.keys():
            if key.startswith('d_'):
                del MovePlans[key]
        ElapsedTimeLogger.stopEvent('delete', 'eval')
        return hmodel, SS, Lscore, MoveLog, MoveRecordsByUID

    def initProgressTrackVars(self, DataIterator):
        ''' Initialize internal attributes tracking how many steps we've taken.

        Returns
        -------
        iterid : int
        lapFrac : float

        Post Condition
        --------------
        Creates attributes nBatch, lapFracInc
        '''
        # Define how much of data we see at each mini-batch
        nBatch = float(DataIterator.nBatch)
        self.nBatch = nBatch
        self.lapFracInc = 1.0 / nBatch

        # Set-up progress-tracking variables
        iterid = -1
        lapFrac = np.maximum(0, self.algParams['startLap'] - 1.0 / nBatch)
        if lapFrac > 0:
            # When restarting an existing run,
            #  need to start with last update for final batch from previous lap
            DataIterator.lapID = int(np.ceil(lapFrac)) - 1
            DataIterator.curLapPos = nBatch - 2
            iterid = int(nBatch * lapFrac) - 1
        return iterid, lapFrac

    def doDebug(self):
        debug = self.algParams['debug']
        return debug.count('q') or debug.count('on') or debug.count('interact')

    def doDebugVerbose(self):
        return self.doDebug() and self.algParams['debug'].count('q') == 0

    def hasMoreReasonableMoves(self, SS, MoveRecordsByUID, lapFrac, **kwargs):
        ''' Decide if more moves will feasibly change current configuration.

        Returns
        -------
        hasMovesLeft : boolean
            True means further iterations likely see births/merges accepted.
            False means all possible moves likely to be rejected.
        '''
        if lapFrac - self.algParams['startLap'] >= self.algParams['nLap']:
            # Time's up, so doesn't matter what other moves are possible.
            return False

        if self.hasMove('birth'):
            nStuck = self.algParams['birth']['b_nStuckBeforeQuit']
            startLap = self.algParams['birth']['b_startLap']
            stopLap = self.algParams['birth']['b_stopLap']
            if stopLap < 0:
                stopLap = np.inf
            if lapFrac > stopLap:
                hasMovesLeft_Birth = False
            elif (lapFrac > startLap + nStuck):
                # If tried for at least nStuck laps without accepting,
                # we consider the method exhausted and exit early.
                b_lapLastAcceptedVec = np.asarray(
                    [MoveRecordsByUID[u]['b_latestLapAccept']
                        for u in MoveRecordsByUID])
                if b_lapLastAcceptedVec.size == 0:
                    lapLastAccepted = 0
                else:
                    lapLastAccepted = np.max(b_lapLastAcceptedVec)
                if (lapFrac - lapLastAccepted) > nStuck:
                    hasMovesLeft_Birth = False
                else:
                    hasMovesLeft_Birth = True
            else:
                hasMovesLeft_Birth = True
        else:
            hasMovesLeft_Birth = False

        if self.hasMove('merge'):
            nStuck = self.algParams['merge']['m_nStuckBeforeQuit']
            startLap = self.algParams['merge']['m_startLap']
            stopLap = self.algParams['merge']['m_stopLap']
            if stopLap < 0:
                stopLap = np.inf
            if lapFrac > stopLap:
                hasMovesLeft_Merge = False
            elif (lapFrac > startLap + nStuck):
                # If tried for at least nStuck laps without accepting,
                # we consider the method exhausted and exit early.
                m_lapLastAcceptedVec = np.asarray(
                    [MoveRecordsByUID[u]['m_latestLapAccept']
                        for u in MoveRecordsByUID])
                if m_lapLastAcceptedVec.size == 0:
                    lapLastAccepted = 0
                else:
                    lapLastAccepted = np.max(m_lapLastAcceptedVec)
                if (lapFrac - lapLastAccepted) > nStuck:
                    hasMovesLeft_Merge = False
                else:
                    hasMovesLeft_Merge = True
            else:
                hasMovesLeft_Merge = True
        else:
            hasMovesLeft_Merge = False

        if self.hasMove('delete'):
            nStuck = self.algParams['delete']['d_nStuckBeforeQuit']
            startLap = self.algParams['delete']['d_startLap']
            stopLap = self.algParams['delete']['d_stopLap']
            if stopLap < 0:
                stopLap = np.inf
            if lapFrac > stopLap:
                hasMovesLeft_Delete = False
            elif lapFrac > startLap + nStuck:
                # If tried for at least nStuck laps without accepting,
                # we consider the method exhausted and exit early.
                d_lapLastAcceptedVec = np.asarray(
                    [MoveRecordsByUID[u]['d_latestLapAccept']
                        for u in MoveRecordsByUID])
                if d_lapLastAcceptedVec.size == 0:
                    lapLastAccepted = 0
                else:
                    lapLastAccepted = np.max(d_lapLastAcceptedVec)
                if (lapFrac - lapLastAccepted) > nStuck:
                    hasMovesLeft_Delete = False
                else:
                    hasMovesLeft_Delete = True
            else:
                hasMovesLeft_Delete = True
        else:
            hasMovesLeft_Delete = False
        return hasMovesLeft_Birth or hasMovesLeft_Merge or hasMovesLeft_Delete
        # ... end function hasMoreReasonableMoves
