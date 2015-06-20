'''
EMAlg.py

Implementation of expectation maximization learning algorithm for bnpy models.
'''
import numpy as np

from LearnAlg import LearnAlg, makeDictOfAllWorkspaceVars

class EMAlg( LearnAlg ):

  def __init__( self, **kwargs ):
    ''' Create EMAlg instance, subtype of generic LearnAlg
    '''
    super(type(self), self).__init__( **kwargs )
    
  def fit(self, hmodel, Data, LP=None):
    ''' Fit point estimates of global parameters of hmodel to Data
        Returns
        --------
        LP : local params from final pass of Data
        Info : dict of run information, with fields
              evBound : final ELBO evidence bound
              status : str message indicating reason for termination
                        {'converged', 'max passes exceeded'}
    '''
    prevBound = -np.inf
    isConverged = False

    ## Save initial state
    self.saveParams(0, hmodel)

    ## Custom func hook
    self.eval_custom_func(isInitial=1, **makeDictOfAllWorkspaceVars(**vars()))

    self.set_start_time_now()
    for iterid in xrange(1, self.algParams['nLap']+1):
      lap = self.algParams['startLap'] + iterid
      nLapsCompleted = lap - self.algParams['startLap']
      self.set_random_seed_at_lap(lap)

      ## Local/E step
      LP = hmodel.calc_local_params(Data, LP, **self.algParamsLP)

      ## Summary step
      SS = hmodel.get_global_suff_stats(Data, LP)

      ## ELBO calculation (needs to be BEFORE Mstep for EM)
      evBound = hmodel.calc_evidence(Data, SS, LP)
      if lap > 1.0:
        ## Report warning if bound isn't increasing monotonically
        self.verify_evidence(evBound, prevBound)

      ## Global/M step
      hmodel.update_global_params(SS) 

      ## Check convergence of expected counts
      countVec = SS.getCountVec()
      if lap > 1.0:
        isConverged = self.isCountVecConverged(countVec, prevCountVec)
        self.setStatus(lap, isConverged)

      ## Display progress
      self.updateNumDataProcessed(Data.get_size())
      if self.isLogCheckpoint(lap, iterid):
        self.printStateToLog(hmodel, evBound, lap, iterid)

      ## Save diagnostics and params
      if self.isSaveDiagnosticsCheckpoint(lap, iterid):
        self.saveDiagnostics(lap, SS, evBound)
      if self.isSaveParamsCheckpoint(lap, iterid):
        self.saveParams(lap, hmodel, SS)

      ## Custom func hook
      self.eval_custom_func(**makeDictOfAllWorkspaceVars(**vars()))

      if nLapsCompleted >= self.algParams['minLaps'] and isConverged:
        break
      prevBound = evBound
      prevCountVec = countVec.copy()
      # ................................................... end loop over laps


    ## Finished! Save, print and exit
    self.saveParams(lap, hmodel, SS)
    self.printStateToLog(hmodel, evBound, lap, iterid, isFinal=1)
    self.eval_custom_func(isFinal=1, **makeDictOfAllWorkspaceVars(**vars()))

    return self.buildRunInfo(evBound=evBound, SS=SS, LP=LP)
