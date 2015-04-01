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


#Two things for shared memory are: the data object and some pieces of the model...can serialize and not worry about 
#if have vector of size k (which is not shared memory can do topics)

#Go through 
#In the consturctor or run method, would need to go through and make the things needed


#Worker constructor
#hand off any shared memory arrays 


#Mike will write calcLocal(Data, slice **arrArgs)
#1) Write the static methods
#2) Create correct frameworks 
# Develop proper SharedMemWorker class and how do I pass in arbitrary shared memory that depends on Beta and model?


#Run some tests to do some of it

#Job Queue
#give it a start and stop, could hand off things related to alloc model like alpha or beta vector


class SharedMemWorker(multiprocessing.Process):
    """ Single "worker" process that processes tasks delivered via queues
    """
    def __init__(self, uid, JobQueue, ResultQueue, 
                 Data=None,
                 arrArgs=None,
                 LPkwargs=None,
                 verbose=0):
        super(Worker, self).__init__()
        self.uid = uid
        self.Data = Data
        if LPkwargs is None:
            LPkwargs = dict()
        self.LPkwargs = LPkwargs
        self.arrArgs = arrArgs #TODO: ask Mike about this unpacking!

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
            sliceArgs = dict(cslice=(start, stop))
            kwargs.update(sliceArgs)

            #TODO insert the static methods here
            #TODO: could take in array arguments
            LP = obsModelCalcLocalParams(self.Data, dict(), **kwargs)
            LP = allocModelCalcLocalParams(self.Data, self.LPKwargs, **kwargs)

            SS = allocModelGetGlobalSuffStats(self.Data, self.LPKwargs, **sliceArgs)
            SS = obsModelGetGlobalSuffStats(self.Data, SS, LP, **sliceArgs)

            self.ResultQueue.put(SS)
            self.JobQueue.task_done()

        # Clean up
        self.printMsg("process CleanUp! pid=%d" % (os.getpid()))

