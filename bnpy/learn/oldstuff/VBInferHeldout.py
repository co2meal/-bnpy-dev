'''
 Variational Bayes learning algorithm

Author: Mike Hughes (mike@michaelchughes.com)
'''
import numpy as np
import time

from .LearnAlg import LearnAlg

class VBInferHeldout( LearnAlg ):

  def __init__( self, **kwargs ):
    super(type(self), self).__init__( **kwargs )

  def infer( self, hmodel, Data ):
    self.start_time = time.time()
    status = "max iters reached."
    prevBound = -np.inf
    evBound = -1
    LP = None
    qType = hmodel.qType
    hmodel.set_qType('VB')
    for iterid in xrange(self.Niter):
      # Only update local parameters!
      LP = hmodel.calc_local_params( Data, LP )
      SS = hmodel.get_global_suff_stats( Data, LP )
      evBound = hmodel.calc_evidence( Data, SS, LP )

      # Display progress
      self.print_state(hmodel, iterid+1, evBound)

      # Check for Convergence!
      #  throw error if our bound calculation isn't working properly
      #    but only if the gap is greater than some tolerance
      isConverged = self.verify_evidence( evBound, prevBound )

      if isConverged:
        status = 'converged.'
        break
      prevBound = evBound

    #Finally, print and exit
    self.print_state(hmodel, iterid+1, evBound, doFinal=True, status=status)
    hmodel.set_qType(qType)
    return LP
