'''
Implementation of parallel memoized variational algorithm for bnpy models.
'''
import numpy as np
import multiprocessing
from collections import defaultdict

from bnpy.birthmove import createSplitStats, assignSplitStats
from bnpy.birthmove import BirthProposalError, selectTargetCompsForBirth
from bnpy.mergemove import selectCandidateMergePairs, ELBO_GAP_ACCEPT_TOL
from bnpy.deletemove import selectCandidateDeleteComps
from bnpy.util import sharedMemDictToNumpy, sharedMemToNumpyArray
from LearnAlg import makeDictOfAllWorkspaceVars
from LearnAlg import LearnAlg
from SharedMemWorker import SharedMemWorker


class MemoVBMovesAlg(LearnAlg):

    def __init__(self, **kwargs):
        ''' Constructor for LearnAlg.
        '''
        # Initialize instance vars related to
        # birth / merge / delete records
        LearnAlg.__init__(self, **kwargs)
        self.SSmemory = dict()
        self.LastUpdateLap = dict()

    def makeNewUIDs(self, b_Kfresh=0, **kwargs):
        newUIDs = np.arange(self.maxUID + 1, self.maxUID + b_Kfresh + 1)
        self.maxUID += b_Kfresh
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
                hmodel, SS, MovePlans,
                MoveRecordsByUID=MoveRecordsByUID,
                lapFrac=lapFrac)

            # Local/Summary step for current batch
            SSbatch = self.calcLocalParamsAndSummarize_withExpansionMoves(
                DataIterator, hmodel,
                SS=SS,
                batchID=batchID,
                lapFrac=lapFrac,
                MovePlans=MovePlans,
                MoveRecordsByUID=MoveRecordsByUID)

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
                else:
                    SS.removeMergeTerms()

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
            if lapFrac > 1.0:
                ConvStatus[batchID] = self.isCountVecConverged(
                    countVec, prevCountVec, batchID=batchID)
                isConverged = np.min(ConvStatus) and \
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

            if isConverged and \
                    self.isLastBatch(lapFrac) and \
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
        LPkwargs.update(MovePlans)
        # Do the real work here: calc local params and summaries
        LPbatch = curModel.calc_local_params(Dbatch, **LPkwargs)
        SSbatch = curModel.get_global_suff_stats(
            Dbatch, LPbatch, doPrecompEntropy=1,
            doTrackTruncationGrowth=1, **MovePlans)

        if 'm_UIDPairs' in MovePlans:
            SSbatch.setMergeUIDPairs(MovePlans['m_UIDPairs'])

        # Prepare whole-dataset stats
        if SS is None:
            curSSwhole = SSbatch.copy()
        else:
            SSbatch.setUIDs(SS.uids)
            curSSwhole = SS

        # Try each planned birth
        SSbatch.propXSS = dict()
        if 'BirthTargetUIDs' in MovePlans:
            # Loop thru copy of the target comp UID list
            # So that we can remove elements from it within the loop
            for ii, targetUID in enumerate(list(MovePlans['BirthTargetUIDs'])):
                if hasattr(SS, 'propXSS') and targetUID in SS.propXSS:
                    SSbatch.propXSS[targetUID] = assignSplitStats(
                        Dbatch, curModel, LPbatch,
                        SS.propXSS[targetUID],
                        curSSwhole=curSSwhole,
                        targetUID=targetUID,
                        LPkwargs=LPkwargs,
                        **self.algParams['birth'])
                else:
                    try:
                        newUIDs = self.makeNewUIDs(**self.algParams['birth'])
                        SSbatch.propXSS[targetUID], Info = \
                            createSplitStats(
                                Dbatch, curModel, LPbatch,
                                curSSwhole=curSSwhole,
                                targetUID=targetUID,
                                newUIDs=newUIDs,
                                LPkwargs=LPkwargs,
                                **self.algParams['birth'])
                    except BirthProposalError as e:
                        print '  ', str(e)
                        MovePlans['BirthTargetUIDs'].remove(targetUID)
                        if targetUID not in MoveRecordsByUID:
                            MoveRecordsByUID[targetUID] = defaultdict(int)
                        MoveRecordsByUID[targetUID]['b_nTrial'] += 1
                        MoveRecordsByUID[targetUID]['b_nFail'] += 1
                        MoveRecordsByUID[targetUID]['b_nFailRecent'] += 1
                        MoveRecordsByUID[targetUID]['b_nSuccessRecent'] = 0
                        MoveRecordsByUID[targetUID]['b_latestLap'] = lapFrac
                        targetCount = SS.getCountVec()[SS.uid2k(targetUID)]
                        MoveRecordsByUID[targetUID]['b_latestCount'] = \
                            targetCount
        if 'd_targetUIDs' in MovePlans:
            from IPython import embed; embed()
            xSS = curSSwhole.copy()

            SSbatch.propXSS[targetUID] = assignSplitStats(
                Dbatch, curModel, LPbatch, xSS,
                curSSwhole=curSSwhole,
                targetUID=targetUID,
                LPkwargs=LPkwargs,
                **self.algParams['birth'])
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
                    SSbatch.transferMassFromExistingToExpansion(
                        uid=targetUID, xSS=SSbatch.propXSS[targetUID])
                else:
                    Kfresh = afterUIDs.size - beforeUIDs.size
                    SSbatch.insertEmptyComps(Kfresh)
                    SSbatch.setUIDs(afterUIDs)
            else:
                raise NotImplementedError("TODO")
            assert np.allclose(SSbatch.uids, afterUIDs)
        # Discard merge terms, since all accepted merges have been incorporated
        SSbatch.removeMergeTerms()
        if not doCopy:
            self.LastUpdateLap[batchID] = lapFrac
        return SSbatch

    def globalStep(self, hmodel, SS, lapFrac):
        ''' Do global update, if appropriate at current lap.

        Post Condition
        ---------
        hmodel global parameters updated in place.
        '''
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
        return hmodel

    def makeMovePlans(self, hmodel, SS, MovePlans=dict(), **kwargs):
        isFirst = self.isFirstBatch(kwargs['lapFrac'])
        if isFirst:
            MovePlans = dict()
        if self.hasMove('birth'):
            MovePlans = self.makeMovePlans_Birth(
                hmodel, SS, MovePlans=MovePlans, **kwargs)
        if isFirst and self.hasMove('delete'):
            MovePlans = self.makeMovePlans_Delete(
                hmodel, SS, MovePlans=MovePlans, **kwargs)
        if isFirst and self.hasMove('merge'):
            MovePlans = self.makeMovePlans_Merge(
                hmodel, SS, MovePlans=MovePlans, **kwargs)
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
        if SS is None:
            return MovePlans
        if lapFrac < self.algParams['merge']['mergeStartLap']:
            return MovePlans

        MArgs = self.algParams['merge']
        MPlan = selectCandidateMergePairs(
            hmodel, SS, MovePlans=MovePlans, **MArgs)
        # Do not track m_UIDPairs field unless it is non-empty
        if len(MPlan['m_UIDPairs']) < 1:
            del MPlan['m_UIDPairs']
            del MPlan['mPairIDs']
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
        if SS is None:
            return MovePlans
        if lapFrac < self.algParams['delete']['d_startLap']:
            return MovePlans
        if self.hasMove('birth'):
            BArgs = self.algParams['birth']
        else:
            BArgs = dict()
        DArgs = self.algParams['delete']
        DArgs.update(BArgs)
        DPlan = selectCandidateDeleteComps(
            hmodel, SS, MovePlans=MovePlans, **DArgs)
        MovePlans.update(DPlan)
        return MovePlans

    def makeMovePlans_Birth(self, hmodel, SS,
                            MovePlans=dict(),
                            MoveRecordsByUID=dict(),
                            lapFrac=0,
                            **kwargs):
        ''' Select comps to target with birth in current batch (or lap).

        Returns
        -------
        MovePlans : dict
            * BirthTargetUIDs : list of uids (ints) indicating comps to target
        '''
        if self.hasMove('birth'):
            print 'EXPANSION STAGE ======================='
            BArgs = self.algParams['birth']    
            if SS is None:
                MovePlans['BirthTargetUIDs'] = \
                    np.arange(hmodel.obsModel.K).tolist()
            else:
                MovePlans = selectTargetCompsForBirth(
                    hmodel, SS,
                    MoveRecordsByUID=MoveRecordsByUID,
                    MovePlans=MovePlans,
                    lapFrac=lapFrac,
                    **BArgs)
            assert isinstance(MovePlans['BirthTargetUIDs'], list)

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
        if len(SS.propXSS.keys()) > 0:
            print 'EVALUATION STAGE ======================='
        acceptedUIDs = list()
        for targetUID in SS.propXSS.keys():
            if 'd_targetUIDs' in MovePlans:
                if targetUID in MovePlans['d_targetUIDs']:
                    continue

            print 'targetUID', targetUID
            if targetUID not in MoveRecordsByUID:
                MoveRecordsByUID[targetUID] = defaultdict(int)

            targetCount = SS.getCountVec()[SS.uid2k(targetUID)]
            MoveRecordsByUID[targetUID]['b_nTrial'] += 1
            MoveRecordsByUID[targetUID]['b_latestLap'] = lapFrac
            MoveRecordsByUID[targetUID]['b_latestCount'] = targetCount
            # Construct proposal statistics
            propSS = SS.copy()
            propSS.transferMassFromExistingToExpansion(
                uid=targetUID, xSS=SS.propXSS[targetUID])
            # Create model via global step from proposed stats
            propModel = hmodel.copy()
            propModel.update_global_params(propSS)
            # Compute score of proposal
            propLscore = propModel.calc_evidence(SS=propSS)

            if propLscore > Lscore:
                print '   ACCEPTED. gainLtotal % .2f' % (propLscore-Lscore)
                MoveRecordsByUID[targetUID]['b_nSuccess'] += 1
                MoveRecordsByUID[targetUID]['b_nFailRecent'] = 0
                MoveRecordsByUID[targetUID]['b_nSuccessRecent'] += 1
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
                MovePlans['BirthTargetUIDs'].remove(targetUID)
                del SS.propXSS[targetUID]

            else:
                propLdata = propModel.obsModel.calc_evidence(None, propSS, None)
                curLdata = hmodel.obsModel.calc_evidence(None, SS, None)
                nAtoms = hmodel.obsModel.getDatasetScale(SS)
                gainLdata = (propLdata - curLdata) / nAtoms
                print '   REJECTED. gainLdata % .2f' % (gainLdata)

                if gainLdata > 0.01:
                    # Track for next time!
                    assert targetUID in SS.propXSS
                else:
                    MovePlans['BirthTargetUIDs'].remove(targetUID)
                    del SS.propXSS[targetUID]
                    MoveRecordsByUID[targetUID]['b_nFail'] += 1
                    MoveRecordsByUID[targetUID]['b_nFailRecent'] += 1
                    MoveRecordsByUID[targetUID]['b_nSuccessRecent'] = 0

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
        if lapFrac < self.algParams['merge']['mergeStartLap']:
            return hmodel, SS, Lscore, MoveLog, MoveRecordsByUID
        assert SS.hasMergeTerms()
        acceptedUIDs = set()
        for (uidA, uidB) in MovePlans['m_UIDPairs']:
            # Skip uids that we have already accepted in a previous merge.
            if uidA in acceptedUIDs or uidB in acceptedUIDs:
                continue
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
            if propLscore > Lscore - ELBO_GAP_ACCEPT_TOL:
                acceptedUIDs.add(uidA)
                acceptedUIDs.add(uidB)
                MoveRecordsByUID[uidA]['m_nSuccess'] += 1
                MoveRecordsByUID[uidA]['m_nSuccessRecent'] += 1
                MoveRecordsByUID[uidA]['m_nFailRecent'] = 0

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
                for u in [uidA, uidB]:
                    MoveRecordsByUID[u]['m_nFail'] += 1
                    MoveRecordsByUID[u]['m_nFailRecent'] += 1
                    MoveRecordsByUID[u]['m_nSuccessRecent'] = 0

        # Finally, set all merge fields to zero,
        # since all possible merges have been accepted
        SS.removeMergeTerms()
        assert not hasattr(SS, 'M')
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

        bigtosmallorder = np.argsort(-1 * SS.getCountVec())
        sortedalready = np.arange(SS.K)
        if not np.allclose(bigtosmallorder, sortedalready):
            moveTuple = (
                lapFrac, 'shuffle',
                dict(bigtosmallorder=bigtosmallorder),
                SS.uids, SS.uids[bigtosmallorder])
            MoveLog.append(moveTuple)
            SS.reorderComps(bigtosmallorder)
            hmodel.update_global_params(SS)
            Lscore = hmodel.calc_evidence(SS=SS)
        elif emptyCompLocs.size > 0:
            hmodel.update_global_params(SS)
            Lscore = hmodel.calc_evidence(SS=SS)

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
            hasMovesLeft_Birth = True

        if self.hasMove('merge'):
            hasMovesLeft_Merge = True
            nStuck = self.algParams['merge']['mergeNumStuckBeforeQuit']
            mergeStartLap = self.algParams['merge']['mergeStartLap']
            if lapFrac > mergeStartLap + nStuck:
                # If tried merges for at least nStuck laps without accepting,
                # we consider all possible merges exhausted and exit early.
                lapLastAcceptedMerge = np.max(
                    [MoveRecordsByUID[u]['m_latestLap'] for u in SS.uids])
                if (lapFrac - lapLastAcceptedMerge) > nStuckBeforeQuit:
                    hasMovesLeft_Merge = False

        return hasMovesLeft_Merge or hasMovesLeft_Birth
        # ... end function hasMoreReasonableMoves
