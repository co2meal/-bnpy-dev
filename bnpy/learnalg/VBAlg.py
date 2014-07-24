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
    super(type(self), self).__init__( **kwargs )
    self.BirthLog = list()
    
  def fit(self, hmodel, Data):
    ''' Run EM/VB learning algorithm, fit global parameters of hmodel to Data
        Returns
        --------
        LP : local params from final pass of Data
        Info : dict of run information, with fields
              evBound : final ELBO evidence bound
              status : str message indicating reason for termination
                        {'converged', 'max passes exceeded'}
    '''
    prevBound = -np.inf
    LP = None
    mergeFlags = dict(doPrecompEntropy=True, doPrecompMergeEntropy=True)
    self.set_start_time_now()
    for iterid in xrange(self.algParams['nLap'] + 1):
      lap = self.algParams['startLap'] + iterid
      self.set_random_seed_at_lap(lap)

      # M step
      if iterid > 0:
        hmodel.update_global_params(SS) 
        from IPython import embed; embed()
      if self.hasMove('birth') and iterid > 1:
        hmodel, LP = self.run_birth_move(hmodel, Data, SS, LP, iterid)
        
      # E step 
      LP = hmodel.calc_local_params(Data, LP, **self.algParamsLP)

      # Suff Stat step
      if self.hasMove('merge'):
        SS = hmodel.get_global_suff_stats(Data, LP, **mergeFlags)
      else:
        SS = hmodel.get_global_suff_stats(Data, LP)

      # ELBO calculation
      evBound = hmodel.calc_evidence(Data, SS, LP)

      # Attempt merge move      
      if self.hasMove('merge'):
        hmodel, SS, LP, evBound = self.run_merge_move(
                                          hmodel, Data, SS, LP, evBound)

      # Save and display progress
      self.add_nObs(Data.nObs)
      self.save_state(hmodel, iterid, lap, evBound)
      self.print_state(hmodel, iterid, lap, evBound)

      # Check for Convergence!
      #  report warning if bound isn't increasing monotonically
      isConverged = self.verify_evidence( evBound, prevBound )
      if isConverged:
        break
      prevBound = evBound

    #Finally, save, print and exit
    if isConverged:
      status = "converged."
    else:
      status = "max passes thru data exceeded."
    self.save_state(hmodel,iterid, lap, evBound, doFinal=True)    
    self.print_state(hmodel,iterid, lap, evBound, doFinal=True, status=status)
    return LP, self.buildRunInfo(evBound, status, nLap=lap)



