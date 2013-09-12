'''
 Variational Bayes learning algorithm

Author: Mike Hughes (mike@michaelchughes.com)
'''
from IPython import embed
import numpy as np
import time

from .LearnAlg import LearnAlg

class VBLearnAlg( LearnAlg ):

  def __init__( self, **kwargs ):
    super(type(self), self).__init__( **kwargs )

  def fit( self, hmodel, Data ):
    '''
        Notes
        -------
        *order* of Mstep, Estep, and ev calculation is very important
    '''
    self.start_time = time.time()
    status = "max iters reached."
    prevBound = -np.inf
    evBound = -1
    LP = None
    for iterid in xrange(self.Niter):
      if iterid > 0:
        # M-step
        hmodel.update_global_params( SS ) 
      
      # E-step 
      LP = hmodel.calc_local_params( Data, LP )
      SS = hmodel.get_global_suff_stats( Data, LP )

      evBound = hmodel.calc_evidence( Data, SS, LP )
      
      # Save and display progress
      self.save_state(hmodel, iterid+1, evBound, Data['nObs'])
      self.print_state(hmodel, iterid+1, evBound)

      # Check for Convergence!
      #  throw error if our bound calculation isn't working properly
      #    but only if the gap is greater than some tolerance
      isConverged = self.verify_evidence( evBound, prevBound )

      if isConverged:
        status = 'converged.'
        break
      prevBound = evBound

    #Finally, save, print and exit 
    self.save_state(hmodel,iterid+1, evBound, Data['nObs'], doFinal=True) 
    self.print_state(hmodel,iterid+1, evBound, doFinal=True, status=status)
    return LP
