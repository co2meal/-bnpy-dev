'''
FromScratchBern.py

Initialize params of a SBM model with Bernoulli observations from scratch.
'''
import numpy as np
from bnpy.util import discrete_single_draw

def init_global_params(hmodel, Data, initname='randexamples', seed=0, K=0, **kwargs):
  PRNG = np.random.RandomState(seed)
  X = Data.X
  N = Data.N
  if initname == 'randexamples':
    ''' Choose K items uniformly at random from the Data
        then component params by M-step given those single items
    '''
    phi = np.zeros((K, N)) + .1
    #rhos = np.zeros((K, N)) + 1
    for i in xrange(N):
        #k = np.round(np.random.rand()*K)-1
        #rhos[k,i] += 1.0
        k = np.round(np.random.rand()*K)-1
        phi[k,i] += 10.0
        #rhos[:,i] = rhos[:,i] / rhos[:,i].sum()
        phi[:,i] = phi[:,i] / phi[:,i].sum()
    
  #Randomly assigned soft probabilities go into LP
 #LP = dict(phi=phi,rhos=rhos)
  LP = dict(phi=phi)
  SS = hmodel.get_global_suff_stats(Data, LP)
  hmodel.update_global_params(SS)
