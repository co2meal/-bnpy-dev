import numpy as np

from bnpy.util import as1D


def calcHammingDistance(zTrue, zHat, excludeNegLabels=1, verbose=0,
        **kwargs):
    ''' Compute Hamming distance: sum of all timesteps with different labels.

    Args
    --------
    zHat : 1D array
        each entry is an integer label in {0, 1, ... Kest-1}
    zTrue : 1D array
        each entry is an integer label in {0, 1, ... Ktrue-1}

    Returns
    ------
    d : int
        Hamming distance from zTrue to zHat.

    Examples
    ------
    >>> calcHammingDistance([0, 0, 1, 1], [0, 0, 1, 1])
    0
    >>> calcHammingDistance([0, 0, 1, 1], [0, 0, 1, 2])
    1
    >>> calcHammingDistance([0, 0, 1, 1], [1, 1, 0, 0])
    4
    >>> calcHammingDistance([1, 1, 0, -1], [1, 1, 0, 0])
    0
    >>> calcHammingDistance([-1, -1, -2, 3], [1, 2, 3, 3])
    0
    >>> calcHammingDistance([-1, -1, 0, 1], [1, 2, 1, 0])
    2
    >>> calcHammingDistance([-1, -1, 0, 1], [1, 2, 1, 0], excludeNegLabels=0)
    4
    '''
    zHat = as1D(zHat)
    zTrue = as1D(zTrue)
    if excludeNegLabels:
        assert np.sum(zHat < 0) == 0
        good_tstep_mask = zTrue >= 0
        if verbose and np.sum(good_tstep_mask) < zTrue.size:
            print 'EXCLUDED %d/%d timesteps' % (np.sum(zTrue < 0), zTrue.size)
        dist = np.sum(zTrue[good_tstep_mask] != zHat[good_tstep_mask])
    else:
        dist = np.sum(zTrue != zHat)
    return dist


def buildCostMatrix(zHat, zTrue):
    ''' Construct cost matrix for alignment of estimated and true sequences

    Args
    --------
    zHat : 1D array
        each entry is an integer label in {0, 1, ... Kest-1}
    zTrue : 1D array
        each entry is an integer label in {0, 1, ... Ktrue-1}
        with optional negative state labels

    Returns
    --------
    CostMatrix : 2D array, size Ktrue x Kest
        CostMatrix[j,k] = count of events across all timesteps,
        where j is assigned, but k is not.
    '''
    zHat = as1D(zHat)
    zTrue = as1D(zTrue)
    Ktrue = int(np.max(zTrue)) + 1
    Kest = int(np.max(zHat)) + 1
    K = np.maximum(Ktrue, Kest)
    CostMatrix = np.zeros((K, K))
    for ktrue in xrange(K):
        for kest in xrange(K):
            CostMatrix[ktrue, kest] = np.sum(np.logical_and(zTrue == ktrue,
                                                            zHat != kest))
    return CostMatrix


def alignEstimatedStateSeqToTruth(zHat, zTrue, returnInfo=False):
    ''' Relabel the states in zHat to minimize the hamming-distance to zTrue

    Args
    --------
    zHat : 1D array
        each entry is an integer label in {0, 1, ... Kest-1}
    zTrue : 1D array
        each entry is an integer label in {0, 1, ... Ktrue-1}

    Returns
    --------
    zHatAligned : 1D array
        relabeled version of zHat that aligns to zTrue
    AInfo : dict
        information about the alignment
    '''
    try:
        import munkres
    except ImportError:
        raise ImportError('Required third-party module munkres not found.\n' +
                          'To fix, add $BNPYROOT/third-party/ to your path.')
    zHat = as1D(zHat)
    zTrue = as1D(zTrue)

    CostMatrix = buildCostMatrix(zHat, zTrue)
    MunkresAlg = munkres.Munkres()
    AlignedRowColPairs = MunkresAlg.compute(CostMatrix)
    zHatA = -1 * np.ones_like(zHat)
    for (ktrue, kest) in AlignedRowColPairs:
        mask = zHat == kest
        zHatA[mask] = ktrue
    if returnInfo:
        return zHatA, dict(CostMatrix=CostMatrix,
                           AlignedRowColPairs=AlignedRowColPairs)
    else:
        return zHatA


def convertStateSeq_flat2list(zFlat, Data):
    ''' Convert flat, 1D array representation of multiple sequences to list
    '''
    zListBySeq = list()
    for n in xrange(Data.nDoc):
        start = Data.doc_range[n]
        stop = Data.doc_range[n + 1]
        zListBySeq.append(zFlat[start:stop])
    return zListBySeq


def convertStateSeq_list2flat(zListBySeq, Data):
    ''' Convert nested list representation of multiple sequences to 1D array
    '''
    zFlat = np.zeros(Data.doc_range[-1])
    for n in xrange(Data.nDoc):
        start = Data.doc_range[n]
        stop = Data.doc_range[n + 1]
        zFlat[start:stop] = zListBySeq[n]
    return zFlat


def convertStateSeq_list2MAT(zListBySeq):
    ''' Convert nested list representation to MAT friendly format
    '''
    N = len(zListBySeq)
    zObjArr = np.zeros((N, 1), dtype=object)
    for n in xrange(N):
        zObjArr[n, 0] = np.asarray(
            zListBySeq[n][:, np.newaxis], dtype=np.int32)
    return zObjArr


def convertStateSeq_MAT2list(zObjArr):
    N = zObjArr.shape[0]
    zListBySeq = list()
    for n in xrange(N):
        zListBySeq.append(np.squeeze(zObjArr[n, 0]))
    return zListBySeq

def calcContigBlocksFromZ(Zvec):
    ''' Identify contig blocks assigned to one state in Zvec

    Examples
    --------
    >>> calcContigBlocksFromZ([0,0,0,0])
    (array([ 4.]), array([ 0.]))
    >>> calcContigBlocksFromZ([0,0,0,1,1])
    (array([ 3.,  2.]), array([ 0.,  3.]))
    >>> calcContigBlocksFromZ([0,1,0])
    (array([ 1.,  1.,  1.]), array([ 0.,  1.,  2.]))
    >>> calcContigBlocksFromZ([0,1,1])
    (array([ 1.,  2.]), array([ 0.,  1.]))
    >>> calcContigBlocksFromZ([6,6,5])
    (array([ 2.,  1.]), array([ 0.,  2.]))

    Returns
    -------
    blockStarts : 1D array of size B
    blockSizes : 1D array of size B
    '''
    changePts = np.asarray(np.hstack([0, 
        1+np.flatnonzero(np.diff(Zvec)),
        len(Zvec)]), dtype=np.float64)
    assert len(changePts) >= 2
    chPtA = changePts[1:]
    chPtB = changePts[:-1]
    blockSizes = chPtA - chPtB
    blockStarts = np.asarray(changePts[:-1], dtype=np.float64)
    return blockSizes, blockStarts
    


if __name__ == '__main__':
    import doctest
    doctest.run_docstring_examples(calcContigBlocksFromZ, globals())