'''
FromScratchBern.py

Initialize global params of Bernoulli data-generation model,
from scratch.
'''

import numpy as np
from bnpy.data import XData
from bnpy.suffstats import SuffStatBag
from scipy.cluster.vq import kmeans2


def init_global_params(obsModel, Data, K=0, seed=0,
                       initname='randexamples',
                       initBlockLen=20,
                       **kwargs):
    ''' Initialize parameters for Bernoulli obsModel, in place.

    Parameters
    -------
    obsModel : bnpy.obsModel subclass
        Observation model object to initialize.
    Data   : bnpy.data.DataObj
        Dataset to use to drive initialization.
        obsModel dimensions must match this dataset.
    initname : str
        name of routine used to do initialization
        Options: ['randexamples', 'randexamplesbydist', 'kmeans',
                  'randcontigblocks', 'randsoftpartition',
                 ]

    Post Condition
    -------
    obsModel has valid global parameters.
    Either its EstParams or Post attribute will be contain K components.
    '''
    PRNG = np.random.RandomState(seed)
    AdjMat = Data.toAdjacencyMatrix()

    K = np.minimum(K, Data.nNodes)
    if len(obsModel.CompDims) == 2:
        CompDims = (K, K,)
    else:
        CompDims = (K,)

    if initname == 'randexamples':
        # Choose K rows of AdjMat uniformly at random from the Data
        chosenNodes = PRNG.choice(AdjMat.shape[0], size=K, replace=False)

        # Build resp from chosenNodes 
        resp = np.zeros((Data.nEdges,) + CompDims)
        for k in xrange(K):
            srcnode_mask = np.flatnonzero(
                Data.edges[:,0] == chosenNodes[k])
            on_mask = srcnode_mask[
                np.sum(Data.X[srcnode_mask,:], axis=1) > 0]
            if len(CompDims) == 2:
                resp[on_mask, :, :] = 0.05 / (K**2-1) * PRNG.rand(K,K)
                resp[on_mask, k, k] = 0.95
            else:
                resp[on_mask, k] = 0.95
    
    elif initname == 'randexamplesbydist':
        # Choose K items from the Data,
        #  selecting the first at random,
        # then subsequently proportional to euclidean distance to the closest
        # item
        objID = PRNG.choice(AdjMat.shape[0])
        chosenNodes = list([objID])
        minDistVec = np.inf * np.ones(AdjMat.shape[0])
        for k in range(1, K):
            curDistVec = np.sum((AdjMat - AdjMat[objID])**2, axis=(1,2))
            minDistVec = np.minimum(minDistVec, curDistVec)
            sum_minDistVec = np.sum(minDistVec)
            if sum_minDistVec > 0:
                p = minDistVec / sum_minDistVec
            else:
                D = minDistVec.size
                p = 1.0 / D * np.ones(D)
            objID = PRNG.choice(Data.nNodes, p=p)
            chosenNodes.append(objID)

        # Build resp from chosenNodes        
        resp = np.zeros((Data.nEdges, K, K))
        for k in xrange(K):
            srcnode_mask = np.flatnonzero(
                Data.edges[:,0] == chosenNodes[k])
            on_mask = srcnode_mask[
                np.sum(Data.X[srcnode_mask,:], axis=1) > 0]
            if len(CompDims) == 2:
                resp[on_mask, :, :] = 0.05 / (K**2-1) * PRNG.rand(K,K)
                resp[on_mask, k, k] = 0.95
            else:
                resp[on_mask, k] = 0.95

    elif initname == 'kmeans':
        # Fill in resp matrix with hard-clustering from K-means
        # using an initialization with K randomly selected points from X
        np.random.seed(seed)
        if AdjMat.shape[2] != 1:
         raise NotImplementedError('Network data k-means initialization' +
                                   'only handles 1-D data')
        centroids, labels = kmeans2(data=AdjMat.squeeze(), k=K, minit='points')
        resp = np.zeros((Data.nEdges, K, K))
        for n in xrange(Data.nNodes):
            srcnode_mask = np.flatnonzero(Data.edges[:,0] == labels[n])
            on_mask = srcnode_mask[
                np.sum(Data.X[srcnode_mask,:], axis=1) > 0]
            k = labels[n]
            if len(CompDims) == 2:
                resp[on_mask, :, :] = 0.05 / (K**2-1) * PRNG.rand(K,K)
                resp[on_mask, k, k] = 0.95
            else:
                resp[on_mask, k] = 0.95

    else:
        raise NotImplementedError('Unrecognized initname ' + initname)

    tempLP = dict(resp=resp)

    # Use the temporary LP to perform one summary and global step
    SS = SuffStatBag(K=K, D=Data.dim)
    SS = obsModel.get_global_suff_stats(Data, SS, tempLP)
    obsModel.update_global_params(SS)
