import os
import multiprocessing
from multiprocessing import sharedctypes

import numpy as np
from scipy.spatial.distance import cdist
import unittest
import ctypes
import bnpy

def getPtrForArray(X):
    """ Get int pointer to memory location of provided array

    Returns
    --------
    ptr : int
    """
    ptr, read_only_flag = X.__array_interface__['data']
    return int(ptr)

def localStep_Vectorized(X, Mu, start=None, stop=None):
  ''' K-means step

      Returns
      -----------
      SuffStatBag with fields
      * N : 1D array, size K
      * x : 2D array, K x D
  '''
  # Unpack size variables
  K, D = Mu.shape

  if start is not None:
    Xcur = X[start:stop]
  else:
    Xcur = X

  # Assign each row of X to closest row of Mu
  Dist = cdist(Xcur, Mu,'sqeuclidean')
  Z = Dist.argmin(axis=1)

  CountVec = np.zeros(K)
  DataStatVec = np.zeros((K,D))
  for k in xrange(K):
    mask_k = Z == k
    CountVec[k] = np.sum(mask_k)
    DataStatVec[k] = np.sum(Xcur[mask_k], axis=0)

  SS = bnpy.suffstats.SuffStatBag(K=K, D=D)
  SS.setField('CountVec', CountVec, dims=('K'))
  SS.setField('DataStatVec', DataStatVec, dims=('K','D'))
  return SS

class Worker(multiprocessing.Process):
    def __init__(self, uid, JobQueue, ResultQueue):
        super(Worker, self).__init__()
        self.uid = uid
        self.JobQueue = JobQueue
        self.ResultQueue = ResultQueue

    def printMsg(self, msg):
        for line in msg.split("\n"):
            print "#%d: %s" % (self.uid, line)

    def run(self):
        self.printMsg("process SetUp! pid=%d" % (os.getpid()))

        # Construct iterator with sentinel value of None (for termination)
        jobIterator = iter(self.JobQueue.get, None)

        for jobArgs in jobIterator:
            X, Mu, start, stop = jobArgs
            self.printMsg("start=%d, stop=%d" % (start, stop))
            msg = "X memory location: %d" % (getPtrForArray(X))
            self.printMsg(msg)
            SS = localStep_Vectorized(X, Mu, start=start, stop=stop)
            self.ResultQueue.put(SS)
            self.JobQueue.task_done()

        self.printMsg("process CleanUp! pid=%d" % (os.getpid()))


class TestKMeans(unittest.TestCase):

  def setUp(self, N=1000, D=25, K=10):
    ''' Create a dataset X (2D array, N x D) and cluster means Mu (2D, KxD)
    '''
    rng = np.random.RandomState(N*D*K)
    self.X = rng.rand(N, D)
    self.Mu = rng.rand(K, D)

  def test_correctness_localStep(self):
    ''' Verify that the local step works as expected.

    No parallelization here. Just verifying that we can split computation up
    and still get the same answer.
    '''
    print ''
    N = self.X.shape[0]

    # Version A: summarize entire dataset
    SSall = localStep_Vectorized(self.X, self.Mu)  

    # Version B: summarize first and second halves separately, then add together
    SSpart1 = localStep_Vectorized(self.X, self.Mu, start=0, stop=N/2)
    SSpart2 = localStep_Vectorized(self.X, self.Mu, start=N/2, stop=N)
    SSbothparts = SSpart1 + SSpart2

    # Both A and B better give the same answer
    assert np.allclose(SSall.CountVec, SSbothparts.CountVec)
    assert np.allclose(SSall.DataStatVec, SSbothparts.DataStatVec)

  def test_parallel_localStep(self, nWorkers=5):
      """ Verify that we can execute local step across several processes

      Each process does the following:
      * grab its chunk of data from a shared jobQueue
      * performs computations on this chunk
      * load the resulting sufficient statistics object into resultsQueue      
      """
      print ''

      # Create a JobQ (to hold tasks to be done)
      # and a ResultsQ (to hold results of completed tasks)
      manager = multiprocessing.Manager()
      JobQ = manager.Queue()
      ResultQ = manager.Queue()

      # Launch desired number of worker processes
      for uid in range(nWorkers):
          Worker(uid, JobQ, ResultQ).start()

      # Create several tasks (one per worker) and add to job queue
      N = self.X.shape[0]
      batch_size = np.floor(N / float(nWorkers))
      for workerID in range(nWorkers):
          start = workerID * batch_size
          stop = (workerID + 1) * batch_size
          if workerID == nWorkers - 1:
              stop = N
          JobQ.put((self.X, self.Mu, start, stop))
          # TODO: provide shared memory version of X/Mu instead??
          # JobQ.put((X_shared, Mu_shared, start, stop))

      # Pause at this line until all jobs are marked complete.
      JobQ.join()

      # REDUCE STEP: Aggregate results across across all workers
      SS = ResultQ.get()
      while not ResultQ.empty():
          SSchunk = ResultQ.get()
          SS += SSchunk

      # Baseline: compute desired answer in master process.
      SSall = localStep_Vectorized(self.X, self.Mu)  

      print "Parallel Answer: CountVec.sum() = ", SS.CountVec.sum()
      print "   Naive Answer: CountVec.sum() = ", SSall.CountVec.sum()
      assert np.allclose(SSall.CountVec, SS.CountVec)

      # Shut down all the workers
      for workerID in range(nWorkers):
          # Handing off None to JobQ is shutdown signal
          JobQ.put( None )

