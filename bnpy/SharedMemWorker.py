""" 
Shared memory parallel implementation of LP local step

Classes
--------

SharedMemWorker : subclass of Process
    Defines work to be done by a single "worker" process
    which is created with references to shared read-only data
    We assign this process "jobs" via a queue, and read its results
    from a separate results queue.

"""

import sys
import os
import multiprocessing
from multiprocessing import sharedctypes
import itertools
import numpy as np
import ctypes
import time

import bnpy

class SharedMemWorker(multiprocessing.Process):
    """ Single "worker" process that processes tasks delivered via queues
    """
    def __init__(self, uid, JobQueue, ResultQueue, 
                 Data=None,
                 verbose=0):
        super(type(self), self).__init__() # Required super constructor call
        self.uid = uid
        self.JobQueue = JobQueue
        self.ResultQueue = ResultQueue
        self.Data=Data
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
            start, stop, LP_part = jobArgs #TODO: **kwargs!!
            if start is not None:
                self.printMsg("start=%d, stop=%d" % (start, stop))

            msg = "X memory location: %d" % (getPtrForArray(self.Data))
            self.printMsg(msg)

            # LP_chunk=LP_resp[start:stop]
            # LP_small=dict()
            # LP_small['resp']=LP_chunk
            # LP_small=self.allocModel.calc_local_params(self.Data, LP_small)

            #TODO: here we run into the problem of how to return this with the original shape, 
            #if we are calling this with a smaller version

            LP_resp=[[0 for i in range(len(LP_resp[0]))] for j in range(len(LP_resp))]#makes it entirely zeros and same size
            LP_resp[start:stop]=LP_small['resp']

            self.ResultQueue.put(LP_resp)
            self.JobQueue.task_done()

        # Clean up
        self.printMsg("process CleanUp! pid=%d" % (os.getpid()))


