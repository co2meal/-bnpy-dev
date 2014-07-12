'''
GSAlg.py

Implementation of Gibbs Sampling for bnpy models

For more info, see the documentation [TODO]
'''
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from LearnAlg import LearnAlg

class GSAlg( LearnAlg ):

  def __init__( self, **kwargs ):
    ''' Create GSAlg, subtype of generic LearnAlg
    '''
    super(type(self), self).__init__( **kwargs )
    
  def fit(self, hmodel, Data):
    ''' Run Gibbs sampling to fit hmodel to data
        Returns
        --------
        LP : local param samples
        Info : dict of run information, with fields
              ll: joint log probability  
              status : str message indicating reason for termination
                        {'max passes exceeded'}
    '''
    # get initial allocations and corresponding suff stats

    LP = hmodel.calc_local_params(Data)
    LP = hmodel.allocModel.make_hard_asgn_local_params(LP)
    SS = hmodel.get_global_suff_stats(Data, LP)
    
    self.set_start_time_now()
    for iterid in xrange(self.algParams['nLap'] + 1):
      lap = self.algParams['startLap'] + iterid
      self.set_random_seed_at_lap(lap)

      # sample posterior allocations
      LP, SS = hmodel.sample_local_params(Data, SS, LP, self.PRNG)
 
      # Make posterior params
      hmodel.update_global_params(SS)

      # Log prob of total sampler state
      ll = hmodel.calcLogLikCollapsedSamplerState(SS)

      # Save and display progress
      self.add_nObs(Data.nObsTotal)
      self.save_state(hmodel, iterid, lap, ll)
      self.print_state(hmodel, iterid, lap, ll)

    
    ## Finally, save, print and exit
    status = "max passes thru data exceeded."
    self.save_state(hmodel,iterid, lap, ll, doFinal=True)    
    self.print_state(hmodel,iterid, lap, ll, doFinal=True, status=status)
    return LP, self.buildRunInfo(ll, status)
