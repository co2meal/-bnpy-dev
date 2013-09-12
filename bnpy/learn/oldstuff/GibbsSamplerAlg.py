'''
 Gibbs Sampler learning algorithm

Author: Mike Hughes (mike@michaelchughes.com)
'''
from IPython import embed
import numpy as np
import time
import random

from .LearnAlg import LearnAlg

class GibbsSamplerAlg( LearnAlg ):

  def __init__( self, **kwargs ):
    super(type(self), self).__init__( **kwargs )

  def init_sampler_params( self, hmodel, Data, seed=None, nObs=None, SLP=None ):
    if self.printEvery >= 0:
      print 'Initialization: %s' % (self.initname)
    if nObs is None:
      nObs = Data['nObs']
    if self.initname == 'seq':
      SS = None
      SLP = dict()
      SLP['Z'] = -1*np.ones( nObs, dtype=np.int32)
      hmodel.reset_K( 2 )
    elif self.initname == 'truth' and 'TrueZ' in Data:
      Kmax = int( Data['TrueZ'].max())+1 + hmodel.allocModel.is_nonparametric()
      hmodel.reset_K( Kmax )
      SLP = dict(Z=np.int32(Data['TrueZ']) )
    elif SLP is not None:
      Kmax = SLP['Z'].max()+1 + doNP # include empty component
      hmodel.reset_K( Kmax )
    SS = hmodel.get_global_suff_stats( Data, SLP)
    hmodel.update_global_params( SS )
    return SS, SLP

  def fit( self, hmodel, Data, seed ):
    self.start_time = time.time()
    status = "max iters reached."
    random.seed( seed)
    np.random.seed( seed )
    
    X = Data['X']
    nObs = X.shape[0]
    permIDs = range(nObs)
    
    SS, SLP = self.init_sampler_params( hmodel, Data, seed=seed )
    
    random.shuffle(permIDs) # inplace shuffle
    for iterid in xrange(self.Niter):
      
      for cid, dataid in enumerate(permIDs):
        SLP, SS = hmodel.sample_local_params_collapsed( dataid, X[dataid], SS, SLP)       
        if iterid ==0 and self.initname == 'seq' and cid % (Data['nObs']/5) == 0:      
          print ' '.join( ['%4d'%(x) for x in SS['N'] ] )
        
      # Not necessary, except for later posterior analysis
      hmodel.allocModel.update_global_params( SS )

      if (iterid+1) % self.printEvery == 0:
        print ' '.join( ['%4d'%(x) for x in SS['N'] ] )


      if iterid > 0:
        assert np.sum( SS['N'] ) == Data['nObs']
       
      
      evidence = hmodel.calc_evidence( Data, SS, SLP )

      # Save and display progress
      self.save_state(hmodel, iterid+1, evidence, Data['nObs'])
      self.print_state(hmodel, iterid+1, evidence)

    #Finally, save, print and exit 
    self.save_state(hmodel, iterid+1, evidence, Data['nObs'], doFinal=True) 
    self.print_state(hmodel, iterid+1, evidence, doFinal=True, status=status)
    return SLP

