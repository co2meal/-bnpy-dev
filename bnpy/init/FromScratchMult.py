'''
FromScratchGauss.py

Initialize params of a mixture model with gaussian observations from scratch.
'''
import numpy as np
from bnpy.util import discrete_single_draw

def init_global_params(hmodel, Data, initname='randexamples', seed=0, K=0, **kwargs):
  
  PRNG = np.random.RandomState(seed)
  word_count = Data.word_count
  total_obs = Data.nObsTotal
  groupid = Data.groupid
  D = Data.D
  if initname == 'randexamples':
    ''' Choose K items uniformly at random from the Data
        then component params by M-step given those single items
    '''
    phi = np.zeros( (K,total_obs) ) + .1 # initialize local word-level variational parameters phi
    theta = np.zeros( (K,D) ) # initialize local document-level variational parameters theta 
    for i in xrange(total_obs):
      k = np.round(np.random.rand()*K)-1
      phi[k,i] = 1.0
      phi[:,i] = phi[:,i] / phi[:,i].sum()
      
    for d in xrange(D):
        start,stop = groupid[d]
        theta[:,d] = np.sum(phi[:,start:stop],axis=1)
        
    
  LP = dict(theta=theta,phi=phi)
  SS = hmodel.get_global_suff_stats(Data, LP)
  hmodel.update_global_params(SS)
