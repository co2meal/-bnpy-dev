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
    super(type(self),self).__init__(**kwargs)
    self.rhodelay = self.algParams['rhodelay']
    self.rhoexp = self.algParams['rhoexp']

  def fit(self, hmodel, DataIterator, SS=None):
    ''' Run soVB learning algorithm, fit global parameters of hmodel to Data
        Returns
        --------
        LP : local params from final pass of Data
        Info : dict of run information, with fields
              evBound : final ELBO evidence bound
              status : str message indicating reason for termination
                        {'all data processed'}
    '''

    LP = None
    rho = 1.0 # Learning rate
    nBatch = float(DataIterator.nBatch)

    # Set-up progress-tracking variables
    iterid = -1
    lapFrac = np.maximum(0, self.algParams['startLap'] - 1.0/nBatch)
    if lapFrac > 0:
      # When restarting an existing run,
      #  need to start with last update for final batch from previous lap
      DataIterator.lapID = int(np.ceil(lapFrac)) - 1
      DataIterator.curLapPos = nBatch - 2
      iterid = int(nBatch * lapFrac) - 1

    ## Save initial state
    self.saveParams(lapFrac, hmodel)

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
      lapFrac += 1.0/nBatch
      self.lapFrac = lapFrac
      nLapsCompleted = lapFrac - self.algParams['startLap']
      self.set_random_seed_at_lap(lapFrac)
      
      # E step
      LP = hmodel.calc_local_params(Dchunk, **self.algParamsLP)

      # ELBO calculation
      if self.algParams['doMemoELBO']:
        SS = hmodel.get_global_suff_stats(Dchunk, LP, doAmplify=False,
                                                      doPrecompEntropy=True)
        if batchID in SSPerBatch:
          SStotal -= SSPerBatch[batchID]
        if SStotal is None:
          SStotal = SS.copy()
        else:
          SStotal += SS
        SSPerBatch[batchID] = SS.copy()
        evBound = hmodel.calc_evidence(SS=SStotal)
        if hasattr(Dchunk, 'nDoc'):
          ampF = Dchunk.nDocTotal / Dchunk.nDoc
          SS.applyAmpFactor(ampF)
        else:
          ampF = Dchunk.nObsTotal / Dchunk.nObs
          SS.applyAmpFactor(ampF)
      else:
        SS = hmodel.get_global_suff_stats(Dchunk, LP, doAmplify=True)
        EvChunk = hmodel.calc_evidence(Dchunk, SS, LP)      

        if EvMemory[batchID] != 0:
          EvRunningSum -= EvMemory[batchID]
        EvRunningSum += EvChunk
        EvMemory[batchID] = EvChunk
        evBound = EvRunningSum / nBatch

      ## M step with learning rate
      if SS is not None:
        rho = (iterid + self.rhodelay) ** (-1.0 * self.rhoexp)
        hmodel.update_global_params(SS, rho)

      ## Display progress
      self.updateNumDataProcessed(Dchunk.get_size())
      if self.isLogCheckpoint(lapFrac, iterid):
        self.printStateToLog(hmodel, evBound, lapFrac, iterid)

      ## Save diagnostics and params
      if self.isSaveDiagnosticsCheckpoint(lapFrac, iterid):
        self.saveDiagnostics(lapFrac, SS, evBound)
      if self.isSaveParamsCheckpoint(lapFrac, iterid):
        self.saveParams(lapFrac, hmodel, SS)
      #.................................................... end loop over data

    # Finished! Save, print and exit
    self.printStateToLog(hmodel, evBound, lapFrac, iterid, isFinal=1)
    self.saveParams(lapFrac, hmodel, SS)
    self.eval_custom_func(isFinal=1, **makeDictOfAllWorkspaceVars(**vars()))

    return self.buildRunInfo(evBound=evBound, SS=SS)
