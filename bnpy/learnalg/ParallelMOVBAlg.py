'''
ParallelVBAlg.py

Implementation of parallel variational bayes learning algorithm for bnpy models.
'''
import numpy as np
import multiprocessing

from LearnAlg import makeDictOfAllWorkspaceVars
from MOVBAlg import MOVBAlg
from bnpy.util import sharedMemDictToNumpy, sharedMemToNumpyArray
from SharedMemWorker import SharedMemWorker


class ParallelMOVBAlg(MOVBAlg):

    def __init__(self, **kwargs):
        ''' Constructor for ParallelMOVBAlg
        '''
        MOVBAlg.__init__(self, **kwargs)
        self.nWorkers = self.algParams['nWorkers']
        if self.nWorkers <= 0:
            self.nWorkers = multiprocessing.cpu_count()
        self.memoLPkeys = []

    def fit(self, hmodel, DataIterator, LP=None, **kwargs):
        ''' Run learning algorithm that fits parameters of hmodel to Data.

        Returns
        --------
        Info : dict of run information.

        Post Condition
        --------
        hmodel updated in place with improved global parameters.
        '''
         # Initialize Progress Tracking vars like nBatch, lapFrac, etc.
        iterid, lapFrac = self.initProgressTrackVars(DataIterator)

        # Save initial state
        self.saveParams(lapFrac, hmodel)

        # Custom func hook
        self.eval_custom_func(
            isInitial=1, **makeDictOfAllWorkspaceVars(**vars()))

        # Create a JobQ (to hold tasks to be done)
        # and a ResultsQ (to hold results of completed tasks)
        manager = multiprocessing.Manager()
        self.JobQ = manager.Queue()
        self.ResultQ = manager.Queue()

        # Get the function handles
        makeDataSliceFromSharedMem = DataIterator.getDataSliceFunctionHandle()
        o_calcLocalParams, o_calcSummaryStats = hmodel.obsModel.getLocalAndSummaryFunctionHandles()
        a_calcLocalParams, a_calcSummaryStats = hmodel.allocModel.getLocalAndSummaryFunctionHandles()

        # Create the shared memory
        try:
            dataSharedMem = DataIterator.getRawDataAsSharedMemDict()
        except AttributeError as e:
            dataSharedMem = None
        aSharedMem = hmodel.allocModel.fillSharedMemDictForLocalStep()
        oSharedMem = hmodel.obsModel.fillSharedMemDictForLocalStep()

        # Create multiple workers
        for uid in range(self.nWorkers):
            SharedMemWorker(uid, self.JobQ, self.ResultQ,
                            makeDataSliceFromSharedMem,
                            o_calcLocalParams,
                            o_calcSummaryStats,
                            a_calcLocalParams,
                            a_calcSummaryStats,
                            dataSharedMem,
                            aSharedMem,
                            oSharedMem,
                            LPkwargs=self.algParamsLP,
                            verbose=1).start()
        
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
            if self.doDebugVerbose():
                self.print_msg('========================== lap %.2f batch %d'
                               % (lapFrac, batchID))

            # Local/E step
            self.algParamsLP['lapFrac'] = lapFrac  # for logging
            self.algParamsLP['batchID'] = batchID
            SSchunk = self.calcLocalParamsAndSummarize(
                DataIterator,
                hmodel,
                batchID=batchID)

            self.saveDebugStateAtBatch('Estep', batchID, 
                                       SS=SS, hmodel=hmodel)

            # Summary step
            SS = self.memoizedSummaryStep(hmodel, SS, SSchunk, batchID)
            
            # Global step
            self.GlobalStep(hmodel, SS, lapFrac)
            aSharedMem = hmodel.allocModel.fillSharedMemDictForLocalStep(aSharedMem)
            oSharedMem = hmodel.obsModel.fillSharedMemDictForLocalStep(oSharedMem)

            # ELBO calculation
            evBound = hmodel.calc_evidence(SS=SS)
            nLapsCompleted = lapFrac - self.algParams['startLap']
            if nLapsCompleted > 1.0:
                # evBound increases monotonically AFTER first lap
                # verify_evidence warns if this isn't happening
                self.verify_evidence(evBound, prevBound, lapFrac)

            if self.doDebug() and lapFrac >= 1.0:
                self.verifyELBOTracking(hmodel, SS, evBound)

            self.saveDebugStateAtBatch(
                'Mstep', batchID, SSchunk=SSchunk,
                SS=SS, hmodel=hmodel)

            # Assess convergence
            countVec = SS.getCountVec()
            if lapFrac > 1.0:
                isConverged = self.isCountVecConverged(countVec, prevCountVec)
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

            if isConverged and self.isLastBatch(lapFrac) \
               and nLapsCompleted >= self.algParams['minLaps']:
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

        return self.buildRunInfo(evBound=evBound, SS=SS,
                                 SSmemory=self.SSmemory)


    def memoizedSummaryStep(self, hmodel, SS, SSchunk, batchID):
        ''' Execute summary step on current batch and update aggregated SS.

        Returns
        --------
        SS : updated aggregate suff stats
        '''
        if batchID in self.SSmemory:
            oldSSchunk = self.load_batch_suff_stat_from_memory(batchID)
            assert oldSSchunk.K == SS.K
            SS -= oldSSchunk
        if SS is None:
            SS = SSchunk.copy()
        else:
            assert SSchunk.K == SS.K
            SS += SSchunk
        self.save_batch_suff_stat_to_memory(batchID, SSchunk)

        # Force aggregated suff stats to obey required constraints.
        # This avoids numerical issues caused by incremental updates
        if hasattr(hmodel.allocModel, 'forceSSInBounds'):
            hmodel.allocModel.forceSSInBounds(SS)
        if hasattr(hmodel.obsModel, 'forceSSInBounds'):
            hmodel.obsModel.forceSSInBounds(SS)
        return SS

    def calcLocalParamsAndSummarize(self, DataIterator, hmodel, 
                                    batchID=0, **kwargs):
        # MAP!
        # Create several tasks (one per worker) and add to job queue
        nWorkers = self.algParams['nWorkers']
        for workerID in xrange(nWorkers):
            sliceArgs = DataIterator.calcSliceArgs(
                batchID, workerID, nWorkers)
            aArgs = hmodel.allocModel.getSerializableParamsForLocalStep()
            oArgs = hmodel.obsModel.getSerializableParamsForLocalStep()
            self.JobQ.put((sliceArgs,aArgs,oArgs))

        # Pause at this line until all jobs are marked complete.
        self.JobQ.join()

        # REDUCE!
        # Aggregate results across across all workers
        SSbigslice = self.ResultQ.get()
        while not self.ResultQ.empty():
            SSslice = self.ResultQ.get()
            SSbigslice += SSslice
        return SSbigslice

