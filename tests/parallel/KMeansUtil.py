""" Provides functions useful for different Kmeans implementations

List of functions
--------------

localStepForDataSlice
    perform kmeans local step (assignments and summary stats)
    for a "slice" of the provided dataset.

sliceGenerator
    generate disjoint slices for provided dataset, given number of workers

runBenchmarkAcrossProblemSizes
    Time execution of provided implementation, across range of N/D/K vals.
"""
import warnings
import itertools
import numpy as np
import bnpy

def localStepForDataSlice(X, Mu, start=None, stop=None):
    ''' K-means step

    Returns
    -----------
    SuffStatBag with fields
    * CountVec : 1D array, size K
    * DataStatVec : 2D array, K x D
    '''
    # If needed, convert input arrays from shared memory to numpy format
    if not isinstance(X, np.ndarray):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            Mu = np.ctypeslib.as_array(Mu)
            X = np.ctypeslib.as_array(X)
            # This does *not* allocate any new memory,
            # just allows using X and Mu as numpy arrays.

    # Unpack problem size variables
    K, D = Mu.shape

    # Grab current slice (subset) of X to work on    
    if start is not None:
        Xcur = X[start:stop]
    else:
        Xcur = X

    # Dist : 2D array, size N x K
    #     squared euclidean distance from X[n] to Mu[k]
    #     up to an additive constant independent of Mu[k]
    Dist = -2 * np.dot(Xcur, Mu.T)
    Dist += np.sum(np.square(Mu), axis=1)[np.newaxis,:]
    # Z : 1D array, size N
    #     Z[n] gives integer id k of closest cluster cntr Mu[k] to X[n,:]
    Z = Dist.argmin(axis=1)

    CountVec = np.zeros(K)
    DataStatVec = np.zeros((K, D))
    for k in xrange(K):
        mask_k = Z == k
        CountVec[k] = np.sum(mask_k)
        DataStatVec[k] = np.sum(Xcur[mask_k], axis=0)

    SS = bnpy.suffstats.SuffStatBag(K=K, D=D)
    SS.setField('CountVec', CountVec, dims=('K'))
    SS.setField('DataStatVec', DataStatVec, dims=('K', 'D'))
    return SS


def sliceGenerator(N=0, nWorkers=0):
    """ Iterate over slices given problem size and num workers

    Yields
    --------
    (start,stop) : tuple
    """
    batchSize = np.floor(N / nWorkers)
    for workerID in range(nWorkers):
        start = workerID * batchSize
        stop = (workerID + 1) * batchSize
        if workerID == nWorkers - 1:
            stop = N
        yield start, stop

def runBenchmarkAcrossProblemSizes(TestClass):
    """ Execute speed benchmark across several N/D/K values.

    Parameters
    --------
    TestClass : constructor for a TestCase instance
        Must offer a run_speed_benchmark method.

    Post Condition
    --------
    Speed tests are executed, and results are printed to std out.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--D', type=str, default='25')
    parser.add_argument('--N', type=str, default='10000')
    parser.add_argument('--K', type=str, default='10')
    parser.add_argument('--nWorkers', type=int, default=2)
    args = parser.parse_args()
    
    NKDiterator = itertools.product(
        rangeFromString(args.N),
        rangeFromString(args.K),
        rangeFromString(args.D))

    for (N, K, D) in NKDiterator:
        print '=============================== N=%d K=%d D=%d' % (
            N, K, D)
        kwargs = dict(**args.__dict__)
        kwargs['N'] = N
        kwargs['K'] = K
        kwargs['D'] = D

        # Create test instance with desired keyword args.
        # Required first arg is string name of test we'll execute
        myTest = TestClass('run_speed_benchmark', **kwargs) 
        myTest.setUp()
        TimeInfo = myTest.run_speed_benchmark()
        myTest.tearDown() # closes all processes


def getPtrForArray(X):
    """ Get int pointer to memory location of provided array

    This can be used to confirm that different processes are
    accessing a common resource, and not duplicating that resource,
    which is wasteful and slows down execution.

    Returns
    --------
    ptr : int
    """
    if isinstance(X, np.ndarray):
        ptr, read_only_flag = X.__array_interface__['data']
        return int(ptr)
    else:
        return id(X)

def rangeFromString(commaString):
    """ Convert a comma string like "1,5-7" into a list [1,5,6,7]

    Returns
    --------
    myList : list of integers

    Reference
    -------
    http://stackoverflow.com/questions/6405208/\
    how-to-convert-numeric-string-ranges-to-a-list-in-python
    """
    listOfLists = [rangeFromHyphen(r) for r in commaString.split(',')]
    flatList = itertools.chain(*listOfLists)
    return flatList

def rangeFromHyphen(hyphenString):
    """ Convert a hyphen string like "5-7" into a list [5,6,7]

    Returns
    --------
    myList : list of integers
    """
    x = [int(x) for x in hyphenString.split('-')]
    return range(x[0], x[-1]+1)

