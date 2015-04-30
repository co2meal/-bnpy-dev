'''
SOVBAlg.py

Implementation of stochastic online VB (soVB) for bnpy models
'''
import numpy as np
from LearnAlg import LearnAlg
from LearnAlg import makeDictOfAllWorkspaceVars


class SOVBAlg(LearnAlg):

    def __init__(self, **kwargs):
        ''' Creates stochastic online learning algorithm,
            with fields rhodelay, rhoexp that define learning rate schedule.
        '''
        super(type(self), self).__init__(**kwargs)
        self.rhodelay = self.algParams['rhodelay']
        self.rhoexp = self.algParams['rhoexp']

    def fit(self, hmodel, DataIterator, SS=None):
        ''' Run stochastic variational to fit hmodel parameters to Data.

        Returns
        --------
        Info : dict of run information.

        Post Condition
        --------
        hmodel updated in place with improved global parameters.
        '''

        LP = None
        rho = 1.0  # Learning rate
        nBatch = float(DataIterator.nBatch)

        # Set-up progress-tracking variables
        iterid = -1
        lapFrac = np.maximum(0, self.algParams['startLap'] - 1.0 / nBatch)
        if lapFrac > 0:
            # When restarting an existing run,
            #  need to start with last update for final batch from previous lap
            DataIterator.lapID = int(np.ceil(lapFrac)) - 1
            DataIterator.curLapPos = nBatch - 2
            iterid = int(nBatch * lapFrac) - 1

        # Save initial state
        self.saveParams(lapFrac, hmodel)

        # Custom func hook
        self.eval_custom_func(
            isInitial=1, **makeDictOfAllWorkspaceVars(**vars()))

        if self.algParams['doMemoELBO']:
            SStotal = None
            SSPerBatch = dict()
        else:
            EvRunningSum = 0
            EvMemory = np.zeros(nBatch)

        self.set_start_time_now()
        while DataIterator.has_next_batch():

            # Grab new data
            Dchunk = DataIterator.get_next_batch()
            batchID = DataIterator.batchID
            Dchunk.batchID = batchID

            # Update progress-tracking variables
            iterid += 1
            lapFrac += 1.0 / nBatch
            self.lapFrac = lapFrac
            nLapsCompleted = lapFrac - self.algParams['startLap']
            self.set_random_seed_at_lap(lapFrac)

            # E step
            self.algParamsLP['batchID'] = batchID
            self.algParamsLP['lapFrac'] = lapFrac  # logging
            LP = hmodel.calc_local_params(Dchunk, **self.algParamsLP)

            rho = (1 + iterid + self.rhodelay) ** (-1.0 * self.rhoexp)
            if self.algParams['doMemoELBO']:
                # SS step. Scale at size of current batch.
                SS = hmodel.get_global_suff_stats(Dchunk, LP,
                                                  doPrecompEntropy=True)
                # Incremental updates for whole-dataset stats
                # Must happen before applification.
                if batchID in SSPerBatch:
                    SStotal -= SSPerBatch[batchID]
                if SStotal is None:
                    SStotal = SS.copy()
                else:
                    SStotal += SS
                SSPerBatch[batchID] = SS.copy()

                # Scale up to size of whole dataset.
                if hasattr(Dchunk, 'nDoc'):
                    ampF = Dchunk.nDocTotal / float(Dchunk.nDoc)
                    SS.applyAmpFactor(ampF)
                else:
                    ampF = Dchunk.nObsTotal / float(Dchunk.nObs)
                    SS.applyAmpFactor(ampF)

                # M step with learning rate
                hmodel.update_global_params(SS, rho)

                # ELBO step
                assert not SStotal.hasAmpFactor()
                evBound = hmodel.calc_evidence(SS=SStotal)
            else:
                # SS step. Scale at size of current batch.
                SS = hmodel.get_global_suff_stats(Dchunk, LP)

                # Scale up to size of whole dataset.
                if hasattr(Dchunk, 'nDoc'):
                    ampF = Dchunk.nDocTotal / float(Dchunk.nDoc)
                    SS.applyAmpFactor(ampF)
                else:
                    ampF = Dchunk.nObsTotal / float(Dchunk.nObs)
                    SS.applyAmpFactor(ampF)

                # M step with learning rate
                hmodel.update_global_params(SS, rho)

                # ELBO step
                assert SS.hasAmpFactor()
                EvChunk = hmodel.calc_evidence(Dchunk, SS, LP)
                if EvMemory[batchID] != 0:
                    EvRunningSum -= EvMemory[batchID]
                EvRunningSum += EvChunk
                EvMemory[batchID] = EvChunk
                evBound = EvRunningSum / nBatch

            # Display progress
            self.updateNumDataProcessed(Dchunk.get_size())
            if self.isLogCheckpoint(lapFrac, iterid):
                self.printStateToLog(hmodel, evBound, lapFrac, iterid, rho=rho)

            # Save diagnostics and params
            if self.isSaveDiagnosticsCheckpoint(lapFrac, iterid):
                self.saveDiagnostics(lapFrac, SS, evBound)
            if self.isSaveParamsCheckpoint(lapFrac, iterid):
                self.saveParams(lapFrac, hmodel, SS)

            self.eval_custom_func(**makeDictOfAllWorkspaceVars(**vars()))
            # .... end loop over data

        # Finished! Save, print and exit
        self.printStateToLog(hmodel, evBound, lapFrac, iterid, isFinal=1)
        self.saveParams(lapFrac, hmodel, SS)
        self.eval_custom_func(
            isFinal=1, **makeDictOfAllWorkspaceVars(**vars()))

        return self.buildRunInfo(evBound=evBound, SS=SS)
