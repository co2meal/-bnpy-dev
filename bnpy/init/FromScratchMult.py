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
    resp = np.zeros( (total_obs, K) ) + .1 # initialize local word-level variational parameters resp
    theta = np.zeros( (D,K) ) # initialize local document-level variational parameters theta 
    for i in xrange(total_obs):
      k = np.round(np.random.rand()*K)-1
      resp[i,k] = 1.0
      resp[i,:] = resp[i,:] / resp[i,:].sum()
      
    for d in xrange(D):
        start,stop = groupid[d]
        theta[d,:] = np.sum(resp[start:stop,:],axis=0)
        
    
  LP = dict(theta=theta,resp=resp)
  SS = hmodel.get_global_suff_stats(Data, LP)
  hmodel.update_global_params(SS)
