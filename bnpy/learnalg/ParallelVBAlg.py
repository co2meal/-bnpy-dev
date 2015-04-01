'''
ParallelVBAlg.py

Implementation of parallel variational bayes learning algorithm for bnpy models.
'''
import numpy as np
import multiprocessing

from LearnAlg import LearnAlg, makeDictOfAllWorkspaceVars

class ParallelVBAlg( LearnAlg ):

  def __init__( self, **kwargs ):
    ''' Create VBLearnAlg, subtype of generic LearnAlg
    '''
    LearnAlg.__init__(self, **kwargs)
    self.nWorkers = 4 #TODO: need to change this
    
  def fit(self, hmodel, Data, LP=None):
    ''' Run VB learning algorithm, fit global parameters of hmodel to Data
        Returns
        --------
        Info : dict of run information, with fields
        * evBound : final ELBO evidence bound
        * status : str message indicating reason for termination
                   {'converged', 'max laps exceeded'}
        * LP : dict of local parameters for final model
    '''
    prevBound = -np.inf
    isConverged = False

    ## Save initial state
    self.saveParams(0, hmodel)

    ## Custom func hook
    self.eval_custom_func(isInitial=1, **makeDictOfAllWorkspaceVars(**vars()))

    self.set_start_time_now()


    isParallel = False #TODO: delete this, this is simply for debugging purposes


    if isParallel:
      # Create a JobQ (to hold tasks to be done)
      # and a ResultsQ (to hold results of completed tasks)
      manager = multiprocessing.Manager()
      self.JobQ = manager.Queue()
      self.ResultQ = manager.Queue()

      #Get the function handles
      makeDataSliceFromSharedMem = hmodel.getHandleMakeDataSliceFromSharedMem()
      o_calcLocalParams = hmodel.obsModel.getHandleCalcLocalParams()
      o_calcSummaryStats = hmodel.obsModel.getHandleCalcSummaryStats()
      a_calcLocalParams = hmodel.allocModel.getHandleCalcLocalParams()
      a_calcSummaryStats = hmodel.allocModel.getHandleCalcSummaryStats()

      #Create the shared memory
      dataSharedMem = Data.converToSharedMem() 
      aSharedMem = hmodel.allocModel.fillInSharedMem() 
      oSharedMem = hmodel.obsModel.fillInSharedMem()

      #Create multiple workers
      for uid in range(self.nWorkers):
        sharedMemWorker(uid,self.JobQ, self.ResultQ,
        makeDataSliceFromSharedMem,
        o_calcLocalParams,
        o_calcSummaryStats,
        a_calcLocalParams,
        a_calcSummaryStats,
        dataSharedMem,
        aSharedMem,
        oSharedMem).start() #TODO: need to find the way to import that from where it is
        #TODO not passing in LPKwargs
    else:
      self.hmodel = hmodel

    for iterid in xrange(1, self.algParams['nLap']+1):
      lap = self.algParams['startLap'] + iterid
      nLapsCompleted = lap - self.algParams['startLap']
      self.set_random_seed_at_lap(lap)

      if isParallel:
        SS = self.calcLocalParamsAndSummarize(hmodel) #TODO fill in params

      else:
        SS = self.serialCalcLocalParamsAndSummarize() 
      ## Global/M step
      hmodel.update_global_params(SS) 

      #update the memory
      aSharedMem = hmodel.allocModel.fillInSharedMem(aSharedMem)
      oSharedMem = hmodel.obs<odel.fillInSharedMem(oSharedMem)

      ## ELBO calculation
      evBound = hmodel.calc_evidence(Data=Data, SS=SS)

      if lap > 1.0:
        ## Report warning if bound isn't increasing monotonically
        self.verify_evidence(evBound, prevBound)

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

    return self.buildRunInfo(evBound=evBound, SS=SS)

  def calcLocalParamsAndSummarize(self,hmodel):
    # MAP!
    # Create several tasks (one per worker) and add to job queue
    for dataBatchID,start,stop in sliceGenerator(self.nDoc, self.nWorkers):
        sliceArgs = (dataBatchID,start,stop)
        aArgs = hmodel.allocModel.getSerializableParamsForLocalStep()
        oArgs = hmodel.obsModel.getSerializableParamsForLocalStep()
        self.JobQ.put((sliceArgs,aArgs,oArgs))

    # Pause at this line until all jobs are marked complete.
    self.JobQ.join()

    # REDUCE!
    # Aggregate results across across all workers
    SS = self.ResultQ.get()
    while not self.ResultQ.empty():
        SSchunk = self.ResultQ.get()
        SS += SSchunk
    return SS

  def serialCalcLocalParamsAndSummarize(self):
    pass

  def sliceGenerator(nDoc=0, nWorkers=0):
    """ Iterate over slices given problem size and num workers

    Yields
    --------
    (start,stop) : tuple
    """
    batchSize = int(np.floor(nDoc / nWorkers))
    for workerID in range(nWorkers):
        start = workerID * batchSize
        stop = (workerID + 1) * batchSize
        if workerID == nWorkers - 1:
            stop = nDoc
        yield start, stop


