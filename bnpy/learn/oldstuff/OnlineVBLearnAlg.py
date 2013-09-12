'''
 Online/Stochastic Variational Bayes learning algorithm

Author: Mike Hughes (mike@michaelchughes.com)
'''
from IPython import embed
import numpy as np
import time

from .LearnAlg import LearnAlg

class OnlineVBLearnAlg( LearnAlg ):

  def __init__( self, **kwargs ):
    super(type(self),self).__init__( **kwargs )
    self.Niter = '' # empty

  def fit( self, hmodel, DataGenerator):
    self.start_time = time.time()
    rho = 1.0

    for iterid, Dchunk in enumerate(DataGenerator):
      # Mstep update with learning rate
      if iterid > 0:
        rho = ( iterid+1 + self.rhodelay )**(-1*self.rhoexp)
        hmodel.update_global_params( SS, rho )

      # E step
      LP = hmodel.calc_local_params( Dchunk )
      SS = hmodel.get_global_suff_stats( Dchunk, LP, Ntotal=Dchunk['nTotal'] )

      evBound = hmodel.calc_evidence( Dchunk, SS, LP)      

      # Save and display progress
      self.save_state( hmodel, iterid+1, evBound, Dchunk['nObs'])
      self.print_state(hmodel, iterid+1, evBound, rho=rho)

    #Finally, save, print and exit 
    status = 'all data gone.'
    try:
      self.save_state(hmodel, iterid+1, evBound, Dchunk['nObs'], doFinal=True) 
      self.print_state(hmodel, iterid+1, evBound, doFinal=True, status=status, rho=rho)
      return LP
    except UnboundLocalError:
      print 'No iters performed.  Perhaps DataGen empty. Rebuild DataGen and try again.'
