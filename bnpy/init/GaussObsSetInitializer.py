'''  GaussObsSetInitializer.py
'''
from IPython import embed
import os
import sys

import numpy as np
import scipy.spatial
import scipy.cluster
from bnpy.util import logsumexp, EPS
from bnpy.ioutil.ModelReader import load_model

class GaussObsSetInitializer( object ):
  def __init__( self, initname='random', seed=0, K=0, **kwargs):
    self.initname = initname
    self.seed = seed
    self.K = K
    self.PRNG = np.random.RandomState(seed)
    
  def init_global_params(self, hmodel, Data):
    X = Data.X
    if self.initname == 'randexamples':
      resp = np.zeros((Data.nObs, self.K))
      permIDs = PRNG.permutation(Data.nObs).tolist()
      for k in xrange(self.K):
        resp[permIDs[k],k] = 1.0
    elif self.initname == 'randsoftpartition':
      resp = np.random.rand(Data.nObs, self.K)
      resp = resp/np.sum(resp,axis=1)[:,np.newaxis]
    
    LP = dict(resp=resp)
    SS = hmodel.get_global_suff_stats(Data, LP)
    hmodel.update_global_params(SS)
