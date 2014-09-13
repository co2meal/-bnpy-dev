'''
VBLearnAlg.py

Implementation of both EM and VB for bnpy models

Notes
-------
Essentially, EM and VB are the same iterative *algorithm*,
repeating the steps of a monotonic increasing objective function until convergence.

EM recovers the parameters for a *point-estimate* of quantities of interest
while VB learns the parameters of an approximate *distribution* over quantities of interest

For more info, see the documentation [TODO]
'''
import numpy as np
from collections import defaultdict
from LearnAlg import LearnAlg

class VBAlg( LearnAlg ):

  def __init__( self, **kwargs ):
    ''' Create VBLearnAlg, subtype of generic LearnAlg
    '''
    LearnAlg.__init__(self, **kwargs)
    #super(type(self), self).__init__( **kwargs )
    
  def fit(self, hmodel, Data):
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
    LP = None
    self.set_start_time_now()
    for iterid in xrange(self.algParams['nLap'] + 1):
      lap = self.algParams['startLap'] + iterid
      self.set_random_seed_at_lap(lap)

      ## Local/E step
      LP = hmodel.calc_local_params(Data, LP, **self.algParamsLP)

      ## Summary step
      if self.hasMove('merge'):
        SS = hmodel.get_global_suff_stats(Data, LP, **mergeFlags)
      else:
        SS = hmodel.get_global_suff_stats(Data, LP)

      ## Global/M step
      hmodel.update_global_params(SS) 

      ## ELBO calculation
      evBound = hmodel.calc_evidence(Data, SS, LP)

      ## Display progress
      self.add_nObs(Dchunk.get_size())
      self.print_state(hmodel, SS, iterid, lapFrac, evBound)

      ## Save diagnostics and params
      if self.isSaveDiagnosticsCheckpoint(lapFrac, iterid):
        self.saveDiagnostics(lapFrac, SS, evBound, self.ActiveIDVec)
      if self.isSaveParamsCheckpoint(lapFrac, iterid):
        self.saveParams(lapFrac, hmodel, SS)

      ## Check for Convergence!
      # Report warning if bound isn't increasing monotonically
      isConverged = self.verify_evidence( evBound, prevBound )
      if isConverged:
        break
      prevBound = evBound

    #Finally, save, print and exit
    if isConverged:
      status = "converged."
    else:
      status = "max laps exceeded."
    self.saveParams(lapFrac, hmodel, SS)
    self.print_state(hmodel, SS, iterid, lap, evBound, 
                     doFinal=True, status=status)
    return self.buildRunInfo(evBound, status, nLap=lap, LP=LP)



