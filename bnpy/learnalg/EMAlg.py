'''
EMAlg.py

Implementation of EM for bnpy models.
'''
import numpy as np
from collections import defaultdict
from LearnAlg import LearnAlg

class EMAlg( LearnAlg ):

  def __init__( self, **kwargs ):
    ''' Create EMAlg instance, subtype of generic LearnAlg
    '''
    super(type(self), self).__init__( **kwargs )
    
  def fit(self, hmodel, Data):
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
    LP = None
    self.set_start_time_now()

    for iterid in xrange(self.algParams['nLap'] + 1):
      lap = self.algParams['startLap'] + iterid

      # M step
      if iterid > 0:
        hmodel.update_global_params(SS) 
              
      # E step 
      LP = hmodel.calc_local_params(Data, LP, **self.algParamsLP)

      # Suff Stat step
      SS = hmodel.get_global_suff_stats(Data, LP)

      # ELBO calculation
      evBound = hmodel.calc_evidence(Data, SS, LP)

      # Save and display progress
      #self.add_nObs(Data.nObs)
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