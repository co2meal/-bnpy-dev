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
    obsModel : AbstractObsModel subclass
        Observation model object to initialize.
    Data   : bnpy.data.DataObj
        Dataset to use to drive initialization.
        obsModel dimensions must match this dataset.
    initname : str
        name of routine used to do initialization
        Options: 'randexamples', 'randexamplesbydist', 'kmeans'

    Returns
    -------
    initLP : dict
        Local parameters used for initialization

    Post Condition
    --------------
    obsModel has valid global parameters.
    Either its EstParams or Post attribute will be contain K components.
    '''
    PRNG = np.random.RandomState(seed)
    AdjMat = Data.toAdjacencyMatrix()
    if AdjMat.ndim == 3:
        AdjMat = AdjMat[:, :, 0]

    K = np.minimum(K, Data.nNodes)
    if len(obsModel.CompDims) == 2:
        CompDims = (K, K,)
    else:
        CompDims = (K,)

    if initname == 'randexamples':
        # Pick K nodes at random in provided graph,
        # and set all edges belonging to that node to one cluster.

        nNodes = AdjMat.shape[0]
        chosenNodes = PRNG.choice(nNodes, size=K, replace=False)

        # Build resp from chosenNodes 
        resp = np.zeros((Data.nEdges,) + CompDims)
        for k in xrange(K):
            src_mask = np.flatnonzero(
                Data.edges[:,0] == chosenNodes[k])
            rcv_mask = np.flatnonzero(
                Data.edges[:,1] == chosenNodes[k])
            src_on = np.sum(Data.X[src_mask,:], axis=1) > 0
            src_off = 1 - src_on
            rcv_on = np.sum(Data.X[rcv_mask,:], axis=1) > 0
            rcv_off = 1 - rcv_on
            if len(CompDims) == 2:
                resp[src_mask[src_on], k, k] = 1.0
                resp[src_mask[src_off], k, :] = 1.0 / (K-1)
                resp[src_mask[src_off], k, k] = 0.0                
                resp[rcv_mask[rcv_on], k, k] = 1.0
                resp[rcv_mask[rcv_off], :, k] = 1.0 / (K-1)
                resp[rcv_mask[rcv_off], k, k] = 0.0                
            else:
                resp[src_mask[src_on], k] = 1.0
                resp[rcv_mask[rcv_on], k] = 1.0
    
    elif initname == 'randexamplesbydist':
        # Choose K items from the Data,
        #  selecting the first at random,
        # then subsequently proportional to euclidean distance to the closest
        # item
        objID = PRNG.choice(AdjMat.shape[0])
        chosenNodes = list([objID])
        minDistVec = np.inf * np.ones(AdjMat.shape[0])
        for k in range(1, K):
            curDistVec = np.sum((AdjMat - AdjMat[objID])**2, axis=1)
            minDistVec = np.minimum(minDistVec, curDistVec)
            sum_minDistVec = np.sum(minDistVec)
            if sum_minDistVec > 0:
                p = minDistVec / sum_minDistVec
            else:
                D = minDistVec.size
                p = 1.0 / D * np.ones(D)
            objID = PRNG.choice(Data.nNodes, p=p)
            chosenNodes.append(objID)

    elif initname == 'kmeans':
        # Fill in resp matrix with hard-clustering from K-means
        # using an initialization with K randomly selected points from X
        np.random.seed(seed)
        centroids, labels = kmeans2(data=AdjMat, k=K, minit='points')
        chosenNodes = labels
    else:
        raise NotImplementedError('Unrecognized initname ' + initname)

    tempLP = chosenNodes_to_LP(chosenNodes, CompDims, Data, 
        PRNG=PRNG, K=K)
    # Use the temporary LP to perform one summary and global step
    SS = SuffStatBag(K=K, D=Data.dim)
    SS = obsModel.get_global_suff_stats(Data, SS, tempLP)
    obsModel.update_global_params(SS)
    return tempLP

def chosenNodes_to_LP(chosenNodes, CompDims, Data, K=0, PRNG=np.random):
    ''' Create resp local parameters for each of the chosen nodes.

    Returns
    -------
    LP : dict with fields
    * resp : 2D array, nEdges x K (or x K x K if full MMSB)
    * resp_bg : 1D array, size nEdges
    '''
    if len(chosenNodes) == Data.nNodes:
        kvGenerator = [(k,v) for (v,k) in enumerate(chosenNodes)]
    else:
        kvGenerator = enumerate(chosenNodes)

    if len(CompDims) == 1:
        resp = np.zeros((Data.nEdges,) + CompDims)
        resp_bg = np.zeros((Data.nEdges,))
        # collect all edges that touch the chosen nodes
        # and assign +1 edges to comp k,k
        # and assign +0 edges to bg comp
        for k, v in kvGenerator:
            v_mask = np.flatnonzero(
                np.logical_or(Data.edges[:, 0] == v,
                              Data.edges[:, 1] == v))
            v_mask_on = v_mask[np.sum(Data.X[v_mask,:], axis=1) > 0]
            v_mask_off = np.setdiff1d(v_mask, v_mask_on)
            resp[v_mask_on, k] = 0.95
            resp_bg[v_mask_off] = 0.95
        return dict(resp=resp, resp_bg=resp_bg)
    else:
        resp = np.zeros((Data.nEdges,) + CompDims)
        # collect all edges that touch the chosen nodes
        # and assign +1 edges to comp k,k
        # and assign +0 edges to some bg comps at random
        for k, v in kvGenerator:
            v_mask = np.flatnonzero(
                np.logical_or(Data.edges[:, 0] == v,
                              Data.edges[:, 1] == v))
            v_mask_on = v_mask[np.sum(Data.X[v_mask,:], axis=1) > 0]
            v_mask_off = np.setdiff1d(v_mask, v_mask_on)
            resp[v_mask_on, k, k] = 0.95
            resp[v_mask_off, :, :] = 0.95 / (K**2)
        return dict(resp=resp)