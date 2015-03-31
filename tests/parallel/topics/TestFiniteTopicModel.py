import os
import multiprocessing
from multiprocessing import sharedctypes
import warnings
import numpy as np
import unittest
import ctypes
import time

import bnpy

def calcLocalParamsAndSummarize(Data, hmodel, start=None, stop=None):
    '''
    Returns
    -----------
    SS : bnpy SuffStatBag
    '''
    print '>>', start, stop
    sliceArgs = dict(cslice=(start, stop))
    LP = hmodel.obsModel.calc_local_params(Data, dict(), **sliceArgs)
    LP = hmodel.allocModel.calc_local_params(Data, LP, **sliceArgs)

    SS = hmodel.allocModel.get_global_suff_stats(Data, LP, **sliceArgs)
    SS = hmodel.obsModel.get_global_suff_stats(Data, SS, LP, **sliceArgs)
    return SS


def sliceGenerator(nDoc=0, nWorkers=0):
    """ Iterate over slices given problem size and num workers

    Yields
    --------
    (start,stop) : tuple
    """
    batchSize = int(np.floor(nDoc / nWorkers))
    for workerID in range(nWorkers):
        start = workerID * batchSize
        stop = (workerID + 1) * batchSize
        if workerID == nWorkers - 1:
            stop = nDoc
        yield start, stop

class Worker(multiprocessing.Process):
    """ Single "worker" process that processes tasks delivered via queues
    """
    def __init__(self, uid, JobQueue, ResultQueue, 
                 Data=None,
                 hmodel=None,
                 verbose=0):
        super(Worker, self).__init__()
        self.uid = uid
        self.Data = Data
        self.hmodel = hmodel

        self.JobQueue = JobQueue
        self.ResultQueue = ResultQueue
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

            SS = calcLocalParamsAndSummarize(
                self.Data, self.hmodel, start=start, stop=stop)
            self.ResultQueue.put(SS)
            self.JobQueue.task_done()

        # Clean up
        self.printMsg("process CleanUp! pid=%d" % (os.getpid()))


class Test(unittest.TestCase):

    def shortDescription(self):
        return None

    def setUp(self, N=1000, nDoc=50, K=10, vocab_size=100,
              seed=0, nWorkers=2, 
              verbose=1):
        ''' 
        '''
        self.N = N
        self.nDoc = nDoc
        self.K = K

        PRNG = np.random.RandomState(seed)

        topics = PRNG.gamma(1.0, 1.0, size=(K, vocab_size))
        np.maximum(topics, 1e-30, out=topics)
        topics /= topics.sum(axis=1)[:,np.newaxis]
        topic_prior = 1.0/K * np.ones(K)
        self.Data = bnpy.data.WordsData.CreateToyDataFromLDAModel(
            nWordsPerDoc=N, nDocTotal=nDoc, K=K, topics=topics,
            seed=seed, topic_prior=topic_prior)

        self.hmodel = bnpy.HModel.CreateEntireModel(
             'VB', 'FiniteTopicModel', 'Mult',
             dict(alpha=0.1, gamma=5),
             dict(lam=0.1), 
             self.Data)

        self.hmodel.init_global_params(self.Data, initname='randexamples', K=K)

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
                   Data=self.Data,
                   hmodel=self.hmodel,
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
        SSall = calcLocalParamsAndSummarize(self.Data, self.hmodel)
        return SSall

    def run_serial(self):
        """ Execute on slices processed in serial by master process.
        """        
        SSagg = None
        for start, stop in sliceGenerator(self.nDoc, self.nWorkers):
            SSslice = calcLocalParamsAndSummarize(
                self.Data, self.hmodel, start, stop)
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
        for start, stop in sliceGenerator(self.nDoc, self.nWorkers):
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
        ''' Verify that the local step works as expected.

        No parallelization here. 
        Just verifying that we can split computation up into >1 slice,
        add up results from all slices and still get the same answer.
        '''
        print ''
        SSbase = self.run_baseline()
        SSserial = self.run_serial()
        allcloseSS(SSbase, SSserial)

    
    def test_correctness_parallel(self):
        """ Verify that we can execute local step across several processes

        Each process does the following:
        * grab its chunk of data from a shared jobQueue
        * performs computations on this chunk
        * load the resulting suff statistics object into resultsQueue      
        """
        print ''

        SSparallel = self.run_parallel()
        SSbase = self.run_baseline()
        allcloseSS(SSparallel, SSbase)

    
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
    

def allcloseSS(SS1, SS2):
    """ Verify that two suff stat bags have indistinguishable data.
    """
    # Both A and B better give the same answer
    for key in SS1._FieldDims.keys():
        arr1 = getattr(SS1, key)
        arr2 = getattr(SS2, key)
        print key
        if isinstance(arr1, float):
            print arr1
            print arr1
        elif arr1.ndim == 1:
            print arr1[:3]
            print arr2[:3]
        else:
            print arr1[:2, :3]
            print arr2[:2, :3]
        assert np.allclose(arr1, arr2)