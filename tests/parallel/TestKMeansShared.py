import os
import multiprocessing
from multiprocessing import sharedctypes
import warnings
import numpy as np
import unittest
import ctypes
import bnpy
import time

def localStep_Vectorized(Xsh, Msh, start=None, stop=None):
    ''' K-means step

    Returns
    -----------
    SuffStatBag with fields
    * N : 1D array, size K
    * x : 2D array, K x D
    '''
    # Unpack variables
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        Mu = np.ctypeslib.as_array(Msh)
        X = np.ctypeslib.as_array(Xsh)
    K, D = Mu.shape

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

class Worker(multiprocessing.Process):
    """ Single "worker" process that processes tasks delivered via queues
    """
    def __init__(self, uid, JobQueue, ResultQueue, 
                 Xsh=None,
                 Msh=None,
                 verbose=0):
        super(Worker, self).__init__()
        self.uid = uid
        self.JobQueue = JobQueue
        self.ResultQueue = ResultQueue
        self.Xsh = Xsh
        self.Msh = Msh
        self.verbose = verbose

    def printMsg(self, msg):
        if self.verbose:
            for line in msg.split("\n"):
                print "#%d: %s" % (self.uid, line)

    def run(self):
        self.printMsg("process SetUp! pid=%d" % (os.getpid()))

        # Construct iterator with sentinel value of None (for termination)
        jobIterator = iter(self.JobQueue.get, None)

        for jobArgs in jobIterator:
            start, stop = jobArgs
            msg = "X memory location: %d" % (getPtrForArray(self.Xsh))
            self.printMsg(msg)

            SS = localStep_Vectorized(self.Xsh, self.Msh,
                                      start=start, stop=stop)
            self.ResultQueue.put(SS)
            self.JobQueue.task_done()

        # Clean up
        self.printMsg("process CleanUp! pid=%d" % (os.getpid()))


class TestN1000K10(unittest.TestCase):

    def shortDescription(self):
        return None

    def setUp(self, N=1000, D=25, K=10, nWorkers=2, verbose=0):
        ''' Create a dataset X (2D array, N x D) and cluster means Mu (2D, KxD)
        '''
        self.N = N
        self.D = D
        self.K = K

        rng = np.random.RandomState((D * K) % 1000)
        self.X = rng.rand(N, D)
        self.Mu = rng.rand(K, D)
        Xsh = toSharedMemArray(self.X)
        Msh = toSharedMemArray(self.Mu)

        # Create a JobQ (to hold tasks to be done)
        # and a ResultsQ (to hold results of completed tasks)
        manager = multiprocessing.Manager()
        self.nWorkers = nWorkers
        self.JobQ = manager.Queue()
        self.ResultQ = manager.Queue()

        # Launch desired number of worker processes
        # We don't need to store references to these processes,
        # We can get everything we need from JobQ and ResultsQ
        for uid in range(self.nWorkers):
            Worker(uid, self.JobQ, self.ResultQ, 
                   Xsh=Xsh,
                   Msh=Msh,
                   verbose=verbose).start()

    def tearDown(self):
        """ Shut down all the workers.
        """
        self.shutdownWorkers()

    def shutdownWorkers(self):
        """ Shut down all worker processes.
        """
        for workerID in range(self.nWorkers):
            # Passing None to JobQ is shutdown signal
            self.JobQ.put(None)

    def run_baseline(self):
        """ Execute on entire matrix (no slices) in master process.
        """        
        SSall = localStep_Vectorized(self.X, self.Mu)
        return SSall

    def run_serial(self):
        """ Execute on slices processed in serial by master process.
        """        
        SSagg = None
        for start, stop in sliceGenerator(self.N, self.nWorkers):
            SSslice = localStep_Vectorized(self.X, self.Mu, start, stop)
            if start == 0:
                SSagg = SSslice
            else:
                SSagg += SSslice
        return SSagg

    def run_parallel(self):
        """ Execute on slices processed by workers in parallel.
        """
        # MAP!
        # Create several tasks (one per worker) and add to job queue
        for start, stop in sliceGenerator(self.N, self.nWorkers):
            self.JobQ.put((start, stop))

        # Pause at this line until all jobs are marked complete.
        self.JobQ.join()

        # REDUCE!
        # Aggregate results across across all workers
        SS = self.ResultQ.get()
        while not self.ResultQ.empty():
            SSchunk = self.ResultQ.get()
            SS += SSchunk
        return SS

    def run_with_timer(self, funcToCall, nRepeat=3):
        starttime = time.time()
        for r in xrange(nRepeat):
            getattr(self, funcToCall)()
        return (time.time() - starttime) / nRepeat

    def run_all_with_timer(self, nRepeat=3):

        serial_time = self.run_with_timer('run_serial')
        parallel_time = self.run_with_timer('run_parallel')
        base_time = self.run_with_timer('run_baseline')

        return dict(
            base_time=base_time,
            base_speedup=1.0,
            serial_time=serial_time,
            serial_speedup=base_time/serial_time,
            parallel_time=parallel_time,
            parallel_speedup=base_time/parallel_time,
            )

    def test_correctness_serial(self):
        ''' Verify that the local step worksas expected.

        No parallelization here. 
        Just verifying that we can split computation up into >1 slice,
        add up results from all slices and still get the same answer.
        '''
        print ''

        # Version A: summarize entire dataset
        SSall = localStep_Vectorized(self.X, self.Mu)

        # Version B: summarize each slice separately, then aggregate
        SSagg = None
        for start, stop in sliceGenerator(self.N, self.nWorkers):
            SSslice = localStep_Vectorized(self.X, self.Mu, start, stop)
            if start == 0:
                SSagg = SSslice
            else:
                SSagg += SSslice

        # Both A and B better give the same answer
        assert np.allclose(SSall.CountVec, SSagg.CountVec)
        assert np.allclose(SSall.DataStatVec, SSagg.DataStatVec)

    def test_correctness_parallel(self):
        """ Verify that we can execute local step across several processes

        Each process does the following:
        * grab its chunk of data from a shared jobQueue
        * performs computations on this chunk
        * load the resulting suff statistics object into resultsQueue      
        """
        print ''

        SS = self.run_parallel()

        # Baseline: compute desired answer in master process.
        SSall = localStep_Vectorized(self.X, self.Mu)

        print "Parallel Answer: CountVec = ", SS.CountVec[:3]
        print "   Naive Answer: CountVec = ", SSall.CountVec[:3]
        assert np.allclose(SSall.CountVec, SS.CountVec)
        assert np.allclose(SSall.DataStatVec, SS.DataStatVec)

    def test_speed(self, nRepeat=5):
        """ Compare speed of different algorithms.
        """
        print ''
        Results = self.run_all_with_timer(nRepeat=nRepeat)

        for key in ['base_time', 'serial_time', 'parallel_time']:
            print "%18s | %8.3f sec | %8.3f speedup" % (
                key, 
                Results[key], 
                Results[key.replace('time', 'speedup')],
                )

class TestN1e6K50(TestN1000K10):

    def setUp(self):
        super(type(self), self).setUp(
            N=1e6, K=50, D=25, verbose=0, nWorkers=2)


def toSharedMemArray(X):
    """ Get copy of X accessible from shared memory
    """
    Xtmp = np.ctypeslib.as_ctypes(X)
    Xsh = multiprocessing.sharedctypes.RawArray(Xtmp._type_, Xtmp)
    return Xsh

def getPtrForArray(X):
    """ Get int pointer to memory location of provided array

    Returns
    --------
    ptr : int
    """
    if isinstance(X, np.ndarray):
        ptr, read_only_flag = X.__array_interface__['data']
        return int(ptr)
    else:
        return id(X)