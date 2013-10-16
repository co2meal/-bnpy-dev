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
from IPython import embed
import numpy as np
from bnpy.learn import LearnAlg

class VBLearnAlg( LearnAlg ):

  def __init__( self, **kwargs ):
    super(type(self), self).__init__( **kwargs )

  def fit( self, hmodel, Data ):
    self.set_start_time_now()
    prevBound = -np.inf
    LP = None
    for iterid in xrange(self.algParams['nLap'] + 1):
      # M step
      if iterid > 0:
        hmodel.update_global_params(SS) 
      
      # E step 
      LP = hmodel.calc_local_params(Data, LP)
      SS = hmodel.get_global_suff_stats(Data, LP)

      # ELBO calculation
      evBound = hmodel.calc_evidence(Data, SS, LP)
      
      # Save and display progress
      self.add_nObs(Data.nObsTotal)
      lap = iterid
      self.save_state(hmodel, iterid, lap, evBound)
      self.print_state(hmodel, iterid, lap, evBound)

      # Check for Convergence!
      #  report warning if bound isn't increasing monotonically
      isConverged = self.verify_evidence( evBound, prevBound )

      if isConverged:
        break
      prevBound = evBound

    #Finally, save, print and exit
    self.save_state(hmodel,iterid, lap, evBound, doFinal=True)
    self.plot_results(hmodel, Data, LP)
    
    if isConverged:
      status = "converged."
    else:
      status = "max passes thru data exceeded."
    self.print_state(hmodel,iterid, lap, evBound, doFinal=True, status=status)
    return LP
