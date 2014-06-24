'''
GSLearnAlg.py

Implementation of Gibbs Sampling for bnpy models

Notes
-------
Essentially, EM and VB are the same iterative *algorithm*,
repeating the steps of a monotonic increasing objective function until convergence.

EM recovers the parameters for a *point-estimate* of quantities of interest
while VB learns the parameters of an approximate *distribution* over quantities of interest

For more info, see the documentation [TODO]
'''
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from LearnAlg import LearnAlg
import pdb

class GSLearnAlg( LearnAlg ):

  def __init__( self, **kwargs ):
    ''' Create VBLearnAlg, subtype of generic LearnAlg
    '''
    super(type(self), self).__init__( **kwargs )
    self.BirthLog = list()
    
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
    LP = dict()
    LP['resp'] = np.asarray(self.initialize(Data)) 
    SS = hmodel.get_global_suff_stats(Data, LP, doAmplify=False)
    
    # flatten LP['resp'] to Z \in R^Data.nObs*1 matrix
    Z = np.argmax(LP['resp'],axis=1)
    
    self.set_start_time_now()
    for iterid in xrange(self.algParams['nLap'] + 1):
      lap = self.algParams['startLap'] + iterid
      self.set_random_seed_at_lap(lap)
       
       
      # sample posterior allocations
      Z,SS = hmodel.sample_local_params(Data, SS, Z)
 
      # jointLL calculation [TODO]
      ll = hmodel.calc_jointll(Data, SS, Z)
      
      # Save and display progress
      self.add_nObs(Data.nObsTotal)
      self.save_state(hmodel, iterid, lap, ll)
      self.print_state(hmodel, iterid, lap, ll)

    
    #unflatten Z
    LP['resp'] = self.unflatten(Data,Z)
     
    #Finally, save, print and exit

    status = "max passes thru data exceeded."
    self.save_state(hmodel,iterid, lap, ll, doFinal=True)    
    self.print_state(hmodel,iterid, lap, ll, doFinal=True, status=status)
    return LP, self.buildRunInfo(ll, status)


  def initialize(self,Data):
    # random initialization
    initZ = np.empty([Data.nObs,1],dtype=int)
    for dataindex in xrange(Data.nObs):
        initZ[dataindex] = np.random.choice(10)
    return self.unflatten(Data,initZ)    

  def unflatten(self,Data,Z):
    ''' Unflatten Z'''  
    row = np.reshape(np.asarray(xrange(Data.nObs)),[1,Data.nObs])
    col = Z
    val = np.ones([Data.nObs])
    #### scipy.sparse and numpy sum don't seem to play well
    #### Needs to be fixed -- for now just create a dense matrix
    return sp.coo_matrix((val.ravel(),(row.ravel(),col.ravel()))).todense()  



