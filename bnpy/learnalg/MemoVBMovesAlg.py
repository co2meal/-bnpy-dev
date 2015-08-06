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
                MovePlans = self.makePlan_Merge(
                    hmodel, SS, MovePlans)

            # Reset selection terms to zero
            if self.isFirstBatch(lapFrac):
                if SS is not None and SS.hasSelectionTerms():
                    SS._SelectTerms.setAllFieldsToZero()

            # Local/Summary step for current batch
            SSbatch = self.calcLocalParamsAndSummarize_withExpansionMoves(
                DataIterator, hmodel,
                SS=SS,
                batchID=batchID,
                lapFrac=lapFrac,
                MovePlans=MovePlans)

            self.saveDebugStateAtBatch(
                'Estep', batchID, SSbatch=SSbatch, SS=SS, hmodel=hmodel)

            # Summary step for whole-dataset stats
            # Incremental update of SS given new SSbatch
            SS = self.memoizedSummaryStep(hmodel, SS, SSbatch, batchID)

            # Global step
            hmodel = self.globalStep(hmodel, SS, lapFrac)

            # ELBO calculation
            Lscore = hmodel.calc_evidence(SS=SS)

            # Birth moves!
            if self.hasMove('birth') and hasattr(SS, 'propXSS'):
                hmodel, SS, Lscore, MoveLog = self.runMoves_Birth(
                    hmodel, SS, Lscore, MoveLog, MovePlans)

            if self.isLastBatch(lapFrac):
                # Merge move!
                if self.hasMove('merge'):
                    hmodel, SS, Lscore, MoveLog = self.runMoves_Merge(
                        hmodel, SS, Lscore, MoveLog, MovePlans)

                # Shuffle : Rearrange order (big to small)
                if self.hasMove('shuffle'):
                    hmodel, SS, Lscore, MoveLog = self.runMoves_Shuffle(
                        hmodel, SS, Lscore, MoveLog, MovePlans)

            nLapsCompleted = lapFrac - self.algParams['startLap']
            if nLapsCompleted > 1.0:
                # evBound increases monotonically AFTER first lap
                # verify_evidence warns if this isn't happening
                self.verify_evidence(Lscore, prevLscore, lapFrac)

            # Debug
            if self.doDebug() and lapFrac >= 1.0:
                self.verifyELBOTracking(hmodel, SS, Lscore, MoveLog)
            self.saveDebugStateAtBatch(
                'Mstep', batchID, SSbatch=SSbatch, SS=SS, hmodel=hmodel)

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


    def memoizedSummaryStep(self, hmodel, SS, SSbatch, batchID,
                            MergePrepInfo=None,
                            order=None,
                            **kwargs):
        ''' Execute summary step on current batch and update aggregated SS.

        Returns
        --------
        SS : updated aggregate suff stats
        '''
        if batchID in self.SSmemory:
            oldSSbatch = self.load_batch_suff_stat_from_memory(
                batchID, doCopy=0, Kfinal=SS.K, order=order)
            assert not oldSSbatch.hasMergeTerms()
            assert oldSSbatch.K == SS.K
            assert np.allclose(SS.uIDs, oldSSbatch.uIDs)
            SS -= oldSSbatch

        # UIDs are not set by parallel workers. Need to do this here
        SSbatch.setUIDs(self.ActiveIDVec.copy())
        if SS is None:
            SS = SSbatch.copy()
        else:
            assert SSbatch.K == SS.K
            assert np.allclose(SSbatch.uIDs, self.ActiveIDVec)
            assert np.allclose(SS.uIDs, self.ActiveIDVec)
            SS += SSbatch
            if not SS.hasSelectionTerms() and SSbatch.hasSelectionTerms():
                SS._SelectTerms = SSbatch._SelectTerms
        assert hasattr(SS, 'uIDs')
        self.save_batch_suff_stat_to_memory(batchID, SSbatch)

        # Force aggregated suff stats to obey required constraints.
        # This avoids numerical issues caused by incremental updates
        if hasattr(hmodel.allocModel, 'forceSSInBounds'):
            hmodel.allocModel.forceSSInBounds(SS)
        if hasattr(hmodel.obsModel, 'forceSSInBounds'):
            hmodel.obsModel.forceSSInBounds(SS)
        return SS

    def calcLocalParamsAndSummarize_withFixedTruncation(
            self, DataIterator, hmodel,
            batchID=0,
            MovePlans=None,
            **kwargs):
        ''' Execute local step and summary step, single-threaded.

        Returns
        -------
        SSbatch : bnpy.suffstats.SuffStatBag
            exact summary of local params for data in specified batch.
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
        LPbatch = hmodel.calc_local_params(Dbatch, **LPkwargs)
        SSbatch = hmodel.get_global_suff_stats(
            Dbatch, LPbatch, doPrecompEntropy=1, **MovePlans)
        return SSbatch

    def calcLocalParamsAndSummarize_withExpansionMoves(
            self, DataIterator, curModel,
            SS=None,
            batchID=0,
            MovePlans=None,
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
            Dbatch, LPbatch, doPrecompEntropy=1, **MovePlans)
        # Prepare whole-dataset stats
        if SS is None:
            curSSwhole = SSbatch.copy()
        else:
            curSSwhole = SS

        for ii, targetUID in enumerate(MovePlans['SplitTargetUIDs']):
            if ii == 0:
                SSbatch.propXSS = dict()
            xSSbatch, Info = \
                createSplitStats(
                    Dbatch, LPbatch, curModel,
                    curSSwhole=curSSwhole,
                    targetUID=targetUID, 
                    LPkwargs=LPkwargs)
            SSbatch.propXSS[targetUID] = xSSbatch

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
