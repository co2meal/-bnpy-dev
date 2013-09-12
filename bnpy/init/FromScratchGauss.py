'''
FromScratchGauss.py

Initialize params of a mixture model with gaussian obsModel from scratch.
'''
import numpy as np

def init_global_params(hmodel, Data, initname='randexamples', seed=0, K=0, **kwargs):
  PRNG = np.random.RandomState(seed)
  X = Data.X
  if initname == 'randexamples':
    ''' Choose K items at random from the Data
        then component params by M-step given those single items
    '''
    resp = np.zeros((Data.nObs, K))
    permIDs = PRNG.permutation(Data.nObs).tolist()
    for k in xrange(K):
      resp[permIDs[k],k] = 1.0
  elif initname == 'randsoftpartition':
    ''' Randomly assign all data items some mass in each of K components
        then create component params by M-step given that soft partition
    '''
    resp = PRNG.rand(Data.nObs, K)
    resp = resp/np.sum(resp,axis=1)[:,np.newaxis]
    
  LP = dict(resp=resp)
  SS = hmodel.get_global_suff_stats(Data, LP)
  hmodel.update_global_params(SS)
