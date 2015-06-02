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
    N = Data.nObs
    C = obsModel.C
    if initname == 'randexamples':
        # Choose K items uniformly at random from the Data
        #    then component params by M-step given those single items
        N = K
        resp = np.zeros((N, K))
        aMean = PRNG.normal(0, 1, (N,K,C))
        aCov = np.tile(np.eye(C), (K,1,1))
        permIDs = PRNG.permutation(Data.nObs).tolist()
        for k in xrange(K):
            # resp[permIDs[k],k] = 1.0
            # aMean[permIDs[k],k] = PRNG.normal(0, obsModel.Prior.f / obsModel.Prior.g, C)
            resp = np.eye(K)
            aMean[k,k] = PRNG.normal(0, obsModel.Prior.f / obsModel.Prior.g, C)
    elif initname == 'fixedResp':
        assert 'Z' in Data.TrueParams
        Z = Data.TrueParams['Z']
        resp = np.zeros((N, K))
        resp[xrange(N), Z.astype(int)] = 1.0
        aMean = PRNG.normal(0, 1, (N,K,C))
        aCov = np.tile(np.eye(C), (K,1,1)) + PRNG.normal(0, .1, (K,C,C))
        numIter = 10
        for i in xrange(numIter):
            tempLP = dict(resp=resp, aMean=aMean, aCov=aCov)
            SS = SuffStatBag(K=K, D=Data.dim, C=C)
            SS = obsModel.get_global_suff_stats(Data, SS, tempLP)
            obsModel.update_global_params(SS)
            aMean, aCov = obsModel.calcA_FromPost(Data)
            print "Initializing %d / %d" % (i, numIter-1)
    else:
        raise NotImplementedError('Unrecognized initname ' + initname)
    tempLP = dict(resp=resp, aMean=aMean, aCov=aCov)
    tmpData = XData(X=Data.X[permIDs[:K]])
    SS = SuffStatBag(K=K, D=Data.dim, C=C)
    SS = obsModel.get_global_suff_stats(tmpData, SS, tempLP)
    # SS = obsModel.get_global_suff_stats(Data, SS, tempLP)
    obsModel.update_global_params(SS)