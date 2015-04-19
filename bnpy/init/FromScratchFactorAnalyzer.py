'''
FromScratchFactorAnalyzer.py

Initialize params of an HModel with factor-analyzer observations from scratch.
'''

import numpy as np
from bnpy.util import discrete_single_draw
from bnpy.data import XData
from bnpy.suffstats import SuffStatBag
from scipy.cluster.vq import kmeans2

def init_global_params(obsModel, Data, K=0, seed=0,
                                       initname='randexamples',
                                       initBlockLen=20,
                                       **kwargs):
    ''' Initialize parameters for Factor-analyzer obsModel, in place.

      Returns
      -------
      Nothing. obsModel is updated in place.
    '''
    PRNG = np.random.RandomState(seed)
    X = Data.X
    if initname == 'randexamples':
        # Choose K items uniformly at random from the Data
        #    then component params by M-step given those single items
        N = Data.nObs
        C = obsModel.C
        resp = np.zeros((N, K))
        aMean = np.zeros((N, K, C))
        aCov =  np.zeros((K, C, C))
        permIDs = PRNG.permutation(Data.nObs).tolist()
        for k in xrange(K):
            resp[permIDs[k],k] = 1.0
            aMean[permIDs[k],k] = PRNG.normal(0, obsModel.Prior.f / obsModel.Prior.g, C)
    else:
        raise NotImplementedError('Unrecognized initname ' + initname)
    tempLP = dict(resp=resp, aMean=aMean, aCov=aCov)
    SS = SuffStatBag(K=K, D=Data.dim, C=C)
    SS = obsModel.get_global_suff_stats(Data, SS, tempLP)
    obsModel.update_global_params(SS)