'''
VBLearnAlg.py

Implementation of both EM and VB for bnpy models

Notes
-------
Essentially, EM and VB are the same iterative *algorithm*,
repeating the steps of a monotonic increasing objective function until convergence.

EM recovers the parameters for a *point-estimate* of quantities of interest
while VB learns the parameters of an approximate *distribution* over quantities of interest
'''
import numpy as np

from bnpy.learn import LearnAlg

class VBLearnAlg( LearnAlg ):

  def __init__( self, **kwargs ):
    super(type(self), self).__init__( **kwargs )

  def fit( self, hmodel, Data ):
    self.set_start_time_now()
    prevBound = -np.inf
    LP = None
    for iterid in xrange(self.args['maxPassThruData']):
      lap = iterid
      if iterid > 0:
        # M-step
        hmodel.update_global_params(SS) 
      
      # E-step 
      LP = hmodel.calc_local_params(Data, LP)
      SS = hmodel.get_global_suff_stats(Data, LP)

      # ELBO calculation
      evBound = hmodel.calc_evidence(Data, SS, LP)
      
      # Save and display progress
      self.add_nObs(Data['nObs'])
      self.save_state(hmodel, iterid, lap, evBound)
      self.print_state(hmodel, iterid, lap, evBound)

      # Check for Convergence!
      #  throw error if our bound calculation isn't working properly
      #    but only if the gap is greater than some tolerance
      isConverged = self.verify_evidence( evBound, prevBound )

      if isConverged:
        status = 'converged.'
        break
      prevBound = evBound

    #Finally, save, print and exit 
    status = "max passes thru data exceeded."
    self.save_state(hmodel,iterid, lap, evBound, doFinal=True) 
    self.print_state(hmodel,iterid, lap, evBound, doFinal=True, status=status)
    return LP
