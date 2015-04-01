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
                 makeDataSliceFromSharedMem,
                 o_calcLocalParams,
                 o_calcSummaryStats,
                 a_calcLocalParams,
                 a_calcSummaryStats,
                 dataSharedMem,
                 aSharedMem,
                 oSharedMem,
                 LPkwargs=None,
                 verbose=0):
        super(Worker, self).__init__()
        self.uid = uid
        self.JobQueue = JobQueue
        self.ResultQueue = ResultQueue

        #Function handles
        self.o_calcLocalParams=o_calcLocalParams
        self.o_calcSummaryStats=o_calcSummaryStats
        self.a_calcLocalParams=a_calcLocalParams
        self.a_calcSummaryStats=a_calcSummaryStats

        #Things to unpack
        self.dataSharedMem=dataSharedMem
        self.aSharedMem=aSharedMem
        self.oSharedMem=oSharedMem
        if LPkwargs is None:
            LPkwargs = dict()
        self.LPkwargs = LPkwargs

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
            sliceArgs, aArgs, oArgs = jobArgs
            Dslice = self.makeDataSliceFromSharedMem(self.dataSharedMem,sliceArgs)
            aArrDict = convertSharedMemToNumpyArrays(self.aSharedMem)
            aArgs.update(aArrDict)
            oArrDict = convertSharedMemToNumpyArrays(self.oSharedMem)
            oArgs.update(oArrDict)

            LP = self.o_calcLocalParams(Dslice, **oArgs)
            LP = self.a_calcLocalParams(Dslice, LP, **aArgs)

            SS = self.a_calcSummaryStats(Dslice, LP, **aArgs)
            SS = self.o_calcSummaryStats(Dslice, SS, LP, **oArgs)

            self.ResultQueue.put(SS)
            self.JobQueue.task_done() 

        # Clean up
        self.printMsg("process CleanUp! pid=%d" % (os.getpid()))

