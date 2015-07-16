'''
Implementation of parallel memoized variational algorithm for bnpy models.
'''
import numpy as np
import multiprocessing

from bnpy.util import sharedMemDictToNumpy, sharedMemToNumpyArray
from LearnAlg import makeDictOfAllWorkspaceVars
from MOVBBirthMergeAlg import MOVBBirthMergeAlg
from SharedMemWorker import SharedMemWorker


class MemoVBMovesAlg(LearnAlg):

    def __init__(self, **kwargs):
        ''' Constructor for LearnAlg.
        '''
        # Initialize instance vars related to
        # birth / merge / delete records
        LearnAlg.__init__(self, **kwargs)
        self.nWorkers = self.algParams['nWorkers']
        if self.nWorkers < 0:
            self.nWorkers = maxWorkers + self.nWorkers
        if self.nWorkers > maxWorkers:
            self.nWorkers = np.maximum(self.nWorkers, maxWorkers)

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

            # Prepare for merges
            if self.hasMove('merge') and self.doMergeAtLap(lapFrac+1):
                MovePlans = self.makePlanForMerge(
                    hmodel, SS, MovePlans)

            # Reset selection terms to zero
            if self.isFirstBatch(lapFrac):
                if SS is not None and SS.hasSelectionTerms():
                    SS._SelectTerms.setAllFieldsToZero()

            # TODO shared memory??

            # Local/Summary step for current batch
            self.algParamsLP['lapFrac'] = lapFrac  # for logging
            self.algParamsLP['batchID'] = batchID
            if self.nWorkers > 0:
                SSchunk = self.calcLocalParamsAndSummarize_parallel(
                    DataIterator, hmodel,
                    MergePrepInfo=MergePrepInfo,
                    batchID=batchID, lapFrac=lapFrac)
            else:
                SSchunk = self.calcLocalParamsAndSummarize_main(
                    DataIterator, hmodel,
                    MergePrepInfo=MergePrepInfo,
                    batchID=batchID, lapFrac=lapFrac)

            self.saveDebugStateAtBatch(
                'Estep', batchID, SSchunk=SSchunk, SS=SS, hmodel=hmodel)

            # Summary step for whole-dataset stats
            # (does incremental update)
            SS = self.memoizedSummaryStep(hmodel, SS, SSchunk, batchID)

            # Global step
            hmodel = self.globalStep(hmodel, SS, lapFrac)

            # ELBO calculation
            Lscore = hmodel.calc_evidence(SS=SS)

            if self.hasMove('birth') and hasattr(SS, 'propXSS'):
                hmodel, SS, Lscore, MoveLog = self.splitMoves(
                    hmodel, SS, Lscore, MoveLog, MovePlans)

            # Merge move!
            if self.hasMove('merge') and self.doMergeAtLap(lapFrac):
                hmodel, SS, Lscore, MoveLog = self.mergeMoves(
                    hmodel, SS, Lscore, MoveLog, MovePlans)

            # Shuffle : Rearrange order (big to small)
            if self.hasMove('shuffle') and self.isLastBatch(lapFrac):
                hmodel, SS, Lscore, MoveLog = self.shuffleMove(
                    hmodel, SS, Lscore, MoveLog, MovePlans)

            # ELBO calculation
            nLapsCompleted = lapFrac - self.algParams['startLap']
            if nLapsCompleted > 1.0:
                # evBound increases monotonically AFTER first lap
                # verify_evidence warns if this isn't happening
                self.verify_evidence(Lscore, prevLscore, lapFrac)

            # Debug
            if self.doDebug() and lapFrac >= 1.0:
                self.verifyELBOTracking(hmodel, SS, Lscore, MoveLog)
            self.saveDebugStateAtBatch(
                'Mstep', batchID, SSchunk=SSchunk, SS=SS, hmodel=hmodel)

            # Assess convergence
            countVec = SS.getCountVec()
            if lapFrac > 1.0:
                convergeStatusByBatch[batchID] = self.isCountVecConverged(
                    countVec, prevCountVec)
                isConverged = np.min(convStatusByBatch) and \
                    self.hasMoreReasonableMoves(SS, MoveLog)
                self.setStatus(lapFrac, isConverged)

            # Display progress
            if self.isLogCheckpoint(lapFrac, iterid):
                self.printStateToLog(hmodel, evBound, lapFrac, iterid)

            # Save diagnostics and params
            if self.isSaveDiagnosticsCheckpoint(lapFrac, iterid):
                self.saveDiagnostics(lapFrac, SS, evBound)
            if self.isSaveParamsCheckpoint(lapFrac, iterid):
                self.saveParams(lapFrac, hmodel, SS)

            # Custom func hook
            self.eval_custom_func(**makeDictOfAllWorkspaceVars(**vars()))

            if isConverged and \
                    self.isLastBatch(lapFrac) and \
                    nLapsCompleted >= self.algParams['minLaps']:
                break
            prevCountVec = countVec.copy()
            prevBound = evBound
            # .... end loop over data

        # Finished! Save, print and exit
        for workerID in range(self.nWorkers):
            # Passing None to JobQ is shutdown signal
            self.JobQ.put(None)

        self.printStateToLog(hmodel, evBound, lapFrac, iterid, isFinal=1)
        self.saveParams(lapFrac, hmodel, SS)
        self.eval_custom_func(
            isFinal=1, **makeDictOfAllWorkspaceVars(**vars()))

        # Births and merges require copies of original model object
        #  we need to make sure original reference has updated parameters, etc.
        if id(origmodel) != id(hmodel):
            origmodel.allocModel = hmodel.allocModel
            origmodel.obsModel = hmodel.obsModel

        # Return information about this run
        return self.buildRunInfo(evBound=evBound, SS=SS,
                                 SSmemory=self.SSmemory)

    def memoizedSummaryStep(self, hmodel, SS, SSchunk, batchID,
                            MergePrepInfo=None,
                            order=None,
                            **kwargs):
        ''' Execute summary step on current batch and update aggregated SS.

        Returns
        --------
        SS : updated aggregate suff stats
        '''
        if batchID in self.SSmemory:
            oldSSchunk = self.load_batch_suff_stat_from_memory(
                batchID, doCopy=0, Kfinal=SS.K, order=order)
            assert not oldSSchunk.hasMergeTerms()
            assert oldSSchunk.K == SS.K
            assert np.allclose(SS.uIDs, oldSSchunk.uIDs)
            SS -= oldSSchunk

        # UIDs are not set by parallel workers. Need to do this here
        SSchunk.setUIDs(self.ActiveIDVec.copy())
        if SS is None:
            SS = SSchunk.copy()
        else:
            assert SSchunk.K == SS.K
            assert np.allclose(SSchunk.uIDs, self.ActiveIDVec)
            assert np.allclose(SS.uIDs, self.ActiveIDVec)
            SS += SSchunk
            if not SS.hasSelectionTerms() and SSchunk.hasSelectionTerms():
                SS._SelectTerms = SSchunk._SelectTerms
        assert hasattr(SS, 'uIDs')
        self.save_batch_suff_stat_to_memory(batchID, SSchunk)

        # Force aggregated suff stats to obey required constraints.
        # This avoids numerical issues caused by incremental updates
        if hasattr(hmodel.allocModel, 'forceSSInBounds'):
            hmodel.allocModel.forceSSInBounds(SS)
        if hasattr(hmodel.obsModel, 'forceSSInBounds'):
            hmodel.obsModel.forceSSInBounds(SS)
        return SS

    def calcLocalParamsAndSummarize_withFixedTruncation(
            self, DataIterator, hmodel,
            MovePlans=None,
            batchID=0, **kwargs):
        ''' Execute local step and summary step, single-threaded.

        Returns
        -------
        SSbatch : bnpy.suffstats.SuffStatBag
        '''
        if not isinstance(MovePlans, dict):
            MovePlans = dict()
        LPkwargs = self.algParamsLP
        LPkwargs.update(MovePlans)

        Dbatch = DataIterator.getBatch(batchID=batchID)
        LPbatch = hmodel.calc_local_params(Dbatch, **LPkwargs)
        SSbatch = hmodel.get_global_suff_stats(
            Dbatch, LPbatch, doPrecompEntropy=1, **MovePlans)
        return SSbatch

    def calcLocalParamsAndSummarize_withExpansionMoves(
            self, DataIterator, curModel,
            MovePlans=dict(),
            SS=None,
            batchID=0, **kwargs):
        ''' Execute local step and summary step, with expansion proposals.

        Returns
        -------
        SSbatch : bnpy.suffstats.SuffStatBag
        '''
        if not isinstance(MovePlans, dict):
            MovePlans = dict(SplitPlans=[])
        LPkwargs = self.algParamsLP
        LPkwargs.update(MovePlans)

        Dbatch = DataIterator.getBatch(batchID=batchID)
        LPbatch = curModel.calc_local_params(Dbatch, **LPkwargs)
        SSbatch = curModel.get_global_suff_stats(
            Dbatch, LPbatch, doPrecompEntropy=1, **MovePlans)

        if SS is None:
            SSwhole = SSbatch.copy()
        else:
            SSwhole = SS
        for ii, SplitPlan in enumerate(MovePlans['SplitPlans']):
            xSSbatch, MovePlans['SplitPlans'][ii] = \
                bnpy.birthmove.makeExpansionStatsForPlan(
                    Dbatch, LPbatch, curModel, SSwhole,
                    Plan=SplitPlan, **LPkwargs)
            SSbatch.propXSS[uid] = xSSbatch

        return SSbatch

    def calcLocalParamsAndSummarize_parallel(self,
                                             DataIterator, hmodel,
                                             MergePrepInfo=None,
                                             batchID=0, lapFrac=-1, **kwargs):
        ''' Execute local step and summary step in parallel via workers.

        Returns
        -------
        SSagg : bnpy.suffstats.SuffStatBag
            Aggregated suff stats from all processed slices of the data.
        '''
        # Map Step
        # Create several tasks (one per worker) and add to job queue
        nWorkers = self.algParams['nWorkers']
        for workerID in xrange(nWorkers):
            sliceArgs = DataIterator.calcSliceArgs(
                batchID, workerID, nWorkers, lapFrac)
            aArgs = hmodel.allocModel.getSerializableParamsForLocalStep()
            aArgs.update(MergePrepInfo)
            oArgs = hmodel.obsModel.getSerializableParamsForLocalStep()
            self.JobQ.put((sliceArgs, aArgs, oArgs))

        # Pause at this line until all jobs are marked complete.
        self.JobQ.join()

        # Reduce step
        # Aggregate results across across all workers
        SSagg = self.ResultQ.get()
        while not self.ResultQ.empty():
            SSslice = self.ResultQ.get()
            SSagg += SSslice
        return SSagg


def setupQueuesAndWorkers(DataIterator, hmodel,
                          algParamsLP=None,
                          nWorkers=0,
                          **kwargs):
    ''' Create pool of worker processes for provided dataset and model.

    Returns
    -------
    JobQ : multiprocessing task queue
        Used for passing tasks to workers
    ResultQ : multiprocessing task Queue
        Used for receiving SuffStatBags from workers
    '''
    # Create a JobQ (to hold tasks to be done)
    # and a ResultsQ (to hold results of completed tasks)
    manager = multiprocessing.Manager()
    JobQ = manager.Queue()
    ResultQ = manager.Queue()

    # Get the function handles
    makeDataSliceFromSharedMem = DataIterator.getDataSliceFunctionHandle()
    o_calcLocalParams, o_calcSummaryStats = hmodel.obsModel.\
        getLocalAndSummaryFunctionHandles()
    a_calcLocalParams, a_calcSummaryStats = hmodel.allocModel.\
        getLocalAndSummaryFunctionHandles()

    # Create the shared memory
    try:
        dataSharedMem = DataIterator.getRawDataAsSharedMemDict()
    except AttributeError as e:
        dataSharedMem = None
    aSharedMem = hmodel.allocModel.fillSharedMemDictForLocalStep()
    oSharedMem = hmodel.obsModel.fillSharedMemDictForLocalStep()

    # Create multiple workers
    for uid in range(nWorkers):
        worker = SharedMemWorker(uid, JobQ, ResultQ,
                                 makeDataSliceFromSharedMem,
                                 o_calcLocalParams,
                                 o_calcSummaryStats,
                                 a_calcLocalParams,
                                 a_calcSummaryStats,
                                 dataSharedMem,
                                 aSharedMem,
                                 oSharedMem,
                                 LPkwargs=algParamsLP,
                                 verbose=1)
        worker.start()

    return JobQ, ResultQ, aSharedMem, oSharedMem
