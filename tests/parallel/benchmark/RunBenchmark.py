import numpy as np
import multiprocessing
import argparse
import os
import itertools
import time

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=str, default='10000')
    parser.add_argument('--D', type=str, default='25')
    parser.add_argument('--nWorker', type=str, default='1')
    parser.add_argument('--methods', type=str, default='parallel')
    parser.add_argument('--nRepeat', type=int, default=1)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--task', type=str, default='sleep')
    parser.add_argument('--durationPerSlice', type=float, default=1.0)
    parser.add_argument('--scaleFactor', type=float, default=0.0)
    parser.add_argument('--memoryType', type=str, default='shared')
    args = parser.parse_args()

    kwargs = args.__dict__
    kwargs['methods'] = args.methods.split()
    for probSizeArgs in problemGenerator(**kwargs):
        kwargs.update(probSizeArgs)

        X = makeData(**kwargs)
        if kwargs['scaleFactor'] == 0 and args.task != 'sleep':
            kwargs['scaleFactor'] = getScaleFactorForTask(X, **kwargs)

        JobQ, ResultQ = launchWorkers(X, **kwargs)

        time.sleep(.1)
        runAllBenchmarks(X, JobQ, ResultQ, **kwargs)

        closeWorkers(JobQ, **kwargs)
        time.sleep(.1)

def getScaleFactorForTask(X, maxWorker=None, **kwargs):
    if maxWorker is None:
        maxWorker = multiprocessing.cpu_count()
    durationPerSlice = kwargs['durationPerSlice']
    N = X.shape[0]
    sliceSize = np.floor(N/maxWorker)
    Xslice = X[:sliceSize]
    kwargs['scaleFactor'] = 1
    t = workOnSlice(Xslice, None, None, **kwargs)
    print 'FINDING PROBLEM SCALE.'
    print '  %d workers\n Min duration of slice: %.2f' \
        % (maxWorker, durationPerSlice)
    while t < durationPerSlice:
        kwargs['scaleFactor'] *= 2
        t = workOnSlice(Xslice, None, None, **kwargs)
    print 'SCALE: ', kwargs['scaleFactor'], 'telapsed=%.3f' % (t)
    return kwargs['scaleFactor']

def workOnSlice(X, start, stop,
                task='sleep',
                durationPerSlice=1.0,
                memoryType='shared',
                nWorker=1,
                scaleFactor=1.0,
                **kwargs):
    """ Perform work on a slice of data.
    """
    if start is None:
        start = 0
        stop = X.shape[0]
        Xslice = X
    else:
        start = int(start)
        stop = int(stop)
        Xslice = X[start:stop]

    if memoryType == 'local':
        Xslice = Xslice.copy()
    elif memoryType == 'random':
        Xslice = np.random.rand(Xslice.shape)
    nReps = int(np.ceil(scaleFactor))

    tstart = time.time()
    if task == 'sleep':
        time.sleep(durationPerSlice * nWorker)
    elif task == 'sumforloop':
        for rep in xrange(nReps):
            s = 0
            for n in xrange(stop-start):
                s = 2 * n
    elif task == 'colsumforloop':
        for rep in xrange(nReps):
            s = 0.0
            for n in xrange(stop-start):
                s += Xslice[n, 0]
    elif task == 'colsumvector':
        for rep in xrange(nReps):
            s = Xslice[:, 0].sum()

    telapsed = time.time() - tstart
    return telapsed


def runBenchmark(X, JobQ, ResultQ,
                 method='serial',
                 nWorker=1, nTaskPerWorker=1,
                 **kwargs):
    N = X.shape[0]
    ts = list()
    if method == 'monolithic':
        kwargs['nWorker'] = nWorker # scale work load by num workers
        t = workOnSlice(X, None, None, **kwargs)
        ts.append(t)
    elif method == 'serial':
        for start, stop in sliceGenerator(N, nWorker, nTaskPerWorker):
            t = workOnSlice(X, start, stop, **kwargs)
            ts.append(t)
    elif method == 'parallel':
        for start, stop in sliceGenerator(N, nWorker, nTaskPerWorker):
            JobQ.put((start, stop, kwargs))

        JobQ.join()
        while not ResultQ.empty():
            t = ResultQ.get()
            ts.append(t)
    return ts


def runAllBenchmarks(X, JobQ, ResultQ, 
                     nRepeat=1, methods='all', **kwargs):
    methodNames = ['monolithic', 'serial', 'parallel']
    print '======================= ', makeTitle(**kwargs)
    print '%16s %15s %15s %10s' % (' ', 
        'wallclock time', 'slice time', 'speedup')
    for method in methodNames:
        if 'all' not in methods and method not in methods:
            continue

        for rep in xrange(nRepeat):
            tstart = time.time()
            ts = runBenchmark(X, JobQ, ResultQ, method=method, **kwargs)
            telapsed = time.time() - tstart
            msg = "%16s" % (method)
            msg += " %11.3f sec" % (telapsed)
            msg += " %11.3f sec" % (np.median(ts))
            if method != 'monolithic' and 'all' in methods:
                msg += " %11.2f" % (telasped_monolithic/telapsed)
            print msg
        if method == 'monolithic':
            telasped_monolithic = telapsed

class SharedMemWorker(multiprocessing.Process):

    """ Single "worker" process that processes tasks delivered via queues
    """

    def __init__(self, uid, JobQ, ResultQ, X):
        super(type(self), self).__init__()  # Required super constructor call
        self.uid = uid
        self.JobQ = JobQ
        self.ResultQ = ResultQ
        self.X = X

    def run(self):
        # Construct iterator with sentinel value of None (for termination)
        jobIterator = iter(self.JobQ.get, None)

        # Loop over tasks in the job queue
        for sliceArgs in jobIterator:
            start, stop, kwargs = sliceArgs
            t = workOnSlice(self.X, start, stop, **kwargs)
            self.ResultQ.put(t)
            self.JobQ.task_done()


def launchWorkers(X, nWorker=1, **kwargs):

    manager = multiprocessing.Manager()
    JobQ = manager.Queue()
    ResultQ = manager.Queue()

    for wID in xrange(nWorker):
        worker = SharedMemWorker(wID, JobQ, ResultQ, X)
        worker.start()
    return JobQ, ResultQ


def closeWorkers(JobQ, nWorker=1, **kwargs):
    for wID in xrange(nWorker):
        JobQ.put(None)  # this is shutdown signal


def makeData(N=10, D=10, **kwargs):
    PRNG = np.random.RandomState(0)
    X = PRNG.rand(N, D)
    return X


def problemGenerator(N=None, D=None, nWorker=None, **kwargs):
    iterator = itertools.product(
        rangeFromString(N),
        rangeFromString(D),
        rangeFromString(nWorker),
    )
    for N, D, nWorker in iterator:
        yield dict(N=N, D=D, nWorker=nWorker)


def makeTitle(N=0, D=0, nWorker=0, 
              task='', memoryType='', scaleFactor=1.0, **kwargs):
    title = "N=%d D=%d nWorker=%d\n" \
        + "task %s\n" \
        + "memoryType %s\n"\
        + "scaleFactor %s\n"
    return title % (N, D, nWorker, task, memoryType, scaleFactor)


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
    return range(x[0], x[-1] + 1)



def sliceGenerator(N=0, nWorker=0, nTaskPerWorker=1):
    """ Iterate over slices given problem size and num workers

    Yields
    --------
    (start,stop) : tuple
    """
    sliceSize = np.floor(N / nWorker)
    for sliceID in range(nWorker * nTaskPerWorker):
        start = sliceID * sliceSize
        stop = (sliceID + 1) * sliceSize
        if sliceID == (nWorker * nTaskPerWorker) - 1:
            stop = N
        yield start, stop


if __name__ == "__main__":
    main()
