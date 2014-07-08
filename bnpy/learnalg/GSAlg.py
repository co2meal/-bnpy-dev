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
    ''' Create VBLearnAlg, subtype of generic LearnAlg
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
    LP = hmodel.initParams  
    LP['resp'] = self.unflatten(Data,LP)
    SS = hmodel.get_global_suff_stats(Data, LP, doAmplify=False)
    
    
    self.set_start_time_now()
    for iterid in xrange(self.algParams['nLap'] + 1):
      lap = self.algParams['startLap'] + iterid
      self.set_random_seed_at_lap(lap)
       
       
      # sample posterior allocations
      LP,SS = hmodel.sample_local_params(Data, SS, LP)
 
      # jointLL calculation [TODO]
      ll = hmodel.calc_jointll(Data, SS, LP)
      
      # Save and display progress
      self.add_nObs(Data.nObsTotal)
      self.save_state(hmodel, iterid, lap, ll)
      self.print_state(hmodel, iterid, lap, ll)

    
    #unflatten Z
    #LP['resp'] = self.unflatten(Data,LP)
     
    #Finally, save, print and exit

    status = "max passes thru data exceeded."
    self.save_state(hmodel,iterid, lap, ll, doFinal=True)    
    self.print_state(hmodel,iterid, lap, ll, doFinal=True, status=status)
    return LP, self.buildRunInfo(ll, status)

  '''
  def initialize(self,Data):
    # random initialization
    initZ = np.empty([Data.nObs,1],dtype=int)
    for dataindex in xrange(Data.nObs):
        initZ[dataindex] = np.random.choice(10)
    return self.unflatten(Data,initZ)    
  '''

  def unflatten(self,Data,LP):
    ''' Unflatten Z'''  
    row = np.reshape(np.asarray(xrange(Data.nObs)),[1,Data.nObs])
    col = LP['Z']
    val = np.ones([Data.nObs])
    #### scipy.sparse and numpy sum don't seem to play well
    #### Needs to be fixed -- for now just create a dense matrix
    return np.asarray(sp.coo_matrix((val.ravel(),(row.ravel(),col.ravel()))).todense())  



