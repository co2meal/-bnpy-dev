'''
FromScratchDiverse.py

Initialize params of a mixture model with different (Gaussian, Multinomial etc) 
observation models from scratch.
'''
import numpy as np
from bnpy.util import discrete_single_draw
from bnpy.data import XData

def init_global_params(hmodel, Data, initname='randexamples', seed=0, K=0, **kwargs):
  ############
  import pdb
  pdb.set_trace()
  ##############  
  PRNG = np.random.RandomState(seed)
  if initname == 'randexamples':
    ''' Choose K items uniformly at random from the Data
        then component params by M-step given those single items
    '''
    resp = np.zeros((Data.nObs, K))
    permIDs = PRNG.permutation(Data.nObs).tolist()
    for k in xrange(K):
      resp[permIDs[k],k] = 1.0
  elif initname == 'randexamplesbydist':
    ''' Choose K items from the Data,
        selecting the first at random,
        then subsequently proportional to euclidean distance to the closest item
    '''
    raise NotImplementedError('TODO?')
  elif initname == 'randsoftpartition':
    ''' Randomly assign all data items some mass in each of K components
        then create component params by M-step given that soft partition
    '''
    resp = PRNG.rand(Data.nObs, K)
    resp = resp/np.sum(resp,axis=1)[:,np.newaxis]

  elif initname == 'randomnaive':
    ''' Generate K "fake" examples from the diagonalized data covariance,
        creating params by assigning each "fake" example to a component.
    '''
    raise NotImplementedError('TODO?')
  
  LP = dict(resp=resp)
 
  SS = hmodel.get_global_suff_stats(Data, LP)
  hmodel.update_global_params(SS)
