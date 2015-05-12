'''
Generic tests for using merge moves during model training with bnpy.
'''
import sys
import numpy as np
import unittest
from nose.plugins.attrib import attr

import bnpy

def arg2name(aArg):
    if isinstance(aArg, dict):
        aName = aArg['name']
    elif isinstance(aArg, str):
        aName = aArg
    return aName

def pprintResult(model, Info, Ktrue=0):
    """ Pretty print the result of a learning algorithm.
    """
    print " %25s after %4.1f sec and %4d laps.  ELBO=% 7.5f  K=%d  Ktrue=%d"\
     % (Info['status'][:25],
        Info['elapsedTimeInSec'],
        Info['lapTrace'][-1],
        Info['evBound'],
        model.allocModel.K,
        Ktrue,
        )
    
def pprint(val):
    """ Pretty print the provided value.
    """
    if isinstance(val, str):
        print '  %s' % (val[:40])
    elif hasattr(val, 'items'):
        firstMsg = ''
        msg = ''
        for (k, v) in val.items():
            if k.count('name'):
                firstMsg = str(v)
            else:
                msg += " %s=%s" % (k, str(v))
        print '  ' + firstMsg + ' ' + msg

def pprintCommandToReproduceError(dataArg, aArg, oArg, algName, **kwargs):
    for key, val in dataArg.items():
        if key == 'name':
            continue
        kwargs[key] = val
    del kwargs['doWriteStdOut']
    del kwargs['doSaveToDisk']
    kwargs['printEvery'] = 1
    kwstr = ' '.join(['--%s %s' % (key, kwargs[key]) for key in kwargs])
    print "python -m bnpy.Run %s %s %s %s %s" % (
        dataArg['name'],
        aArg['name'], 
        oArg['name'],
        algName, 
        kwstr,
        )

def is_monotonic(ELBOvec, aArg=None, atol=1e-5, verbose=True):
    ''' Returns True if provided vector monotonically increases, False o.w. 

    Returns
    -------
    result : boolean (True or False)
    '''
    if aArg is not None:
        if 'name' in aArg:
            if aArg['name'] == 'HDPTopicModel':
                # ELBO can fluctuate more due to no caching at localstep
                atol = 1e-3

    ELBOvec = np.asarray(ELBOvec, dtype=np.float64)
    assert ELBOvec.ndim == 1
    diff = ELBOvec[1:] - ELBOvec[:-1]
    maskIncrease = diff > 0
    maskWithinTol = np.abs(diff) < atol
    maskOK = np.logical_or(maskIncrease, maskWithinTol)
    isMonotonic = np.all(maskOK)
    if not isMonotonic and verbose:
        print "NOT MONOTONIC!"
        print '  %d violations in vector of size %d. Biggest drop %.8f' \
            % (np.sum(1 - maskOK), ELBOvec.size, diff[diff < 0].max())
    return isMonotonic

class MergeMoveEndToEndTest(unittest.TestCase):

    """ Defines test exercises for executing bnpy.run on provided dataset.

        Attributes
        ----
        Data : bnpy.data.DataObj
            dataset under testing
    """

    __test__ = False  # Do not execute this abstract module!

    def shortDescription(self):
        return None

    def makeAllKwArgs(self, aArg, obsArg, initArg=dict(), 
        **kwargs):

        allKwargs = dict(
            doSaveToDisk=False,
            doWriteStdOut=False,
            saveEvery=-1,
            printEvery=-1,
            traceEvery=1,
            convergeThr=0.0001,
            doFullPassBeforeMstep=1,
            nLap=300,
            nBatch=2,
            mergeStartLap=2,
            deleteStartLap=2,
            nCoordAscentItersLP=50,
            convThrLP=0.001,
        )
        allKwargs.update(kwargs)
        allKwargs.update(aArg)
        allKwargs.update(obsArg)
        allKwargs.update(initArg)

        if allKwargs['moves'].count('delete'):
            try:
                MaxSize = 0.5 * int(self.datasetArg['nDocTotal'])
            except KeyError:
                MaxSize = 0.5 * int(self.datasetArg['nObsTotal'])
            allKwargs['dtargetMaxSize'] = int(MaxSize)

        if aArg['name'] == 'HDPTopicModel':
            allKwargs['mergePairSelection'] = 'corrlimitdegree'            
        else:
            allKwargs['mergePairSelection'] = 'wholeELBObetter'
        return allKwargs

    def run_MOVBWithMoves(self, aArg, oArg,
            moves='merge',
            algName='moVB',
            nWorkers=0,
            **kwargs):
        """ Execute single run with merge moves enabled.

        Post Condition
        --------------
        Will raise AssertionError if any bad results detected.
        """
        Ktrue = self.Data.TrueParams['K']
        pprint(aArg)
        pprint(oArg)
        initArg = dict(**kwargs)
        pprint(initArg)
        kwargs = self.makeAllKwArgs(aArg, oArg, initArg, 
            moves=moves, nWorkers=nWorkers,
            **kwargs)
        model, Info = bnpy.run(self.Data, 
            arg2name(aArg), arg2name(oArg), algName, **kwargs)
        pprintResult(model, Info, Ktrue=Ktrue)

        afterFirstLapMask = Info['lapTrace'] >= 1.0
        evTraceAfterFirstLap = Info['evTrace'][afterFirstLapMask]
        isMonotonic = is_monotonic(evTraceAfterFirstLap,
            aArg=aArg)

        try:
            assert isMonotonic
            assert model.allocModel.K == model.obsModel.K
            assert model.allocModel.K == Ktrue

        except AssertionError as e:
            pprintCommandToReproduceError(
                self.datasetArg, aArg, oArg, algName, **kwargs)
            assert isMonotonic
            assert model.allocModel.K == model.obsModel.K
            if not model.allocModel.K == Ktrue:
                print '>>>>>> WHOA! Kfinal != Ktrue <<<<<<'
        return Info


    def runMany_MOVBWithMoves(self, 
            initnames=['truelabels', 
                       'repeattruelabels', 
                       'truelabelsandempty'],
            algName='moVB',
            nWorkers=0,
            moves='merge,delete,shuffle'):
        print ''
        for aKwArgs in self.nextAllocKwArgsForVB():
            for oKwArgs in self.nextObsKwArgsForVB():
                Info = dict()
                for iname in initnames:
                    if iname.count('junk') or iname.count('empty'):
                        initKextra = 1
                    else:
                        initKextra = 0
                    Info[iname] = self.run_MOVBWithMoves(
                        aKwArgs, oKwArgs, 
                        moves=moves,
                        algName=algName,
                        nWorkers=nWorkers,
                        initKextra=initKextra,
                        initname=iname)

    def test_MOVBWithMerges(self):
        self.runMany_MOVBWithMoves(moves='merge')

    def test_MOVBWithDeletes(self):
        self.runMany_MOVBWithMoves(moves='delete')

    def test_MOVBWithMergeDeletes(self):
        self.runMany_MOVBWithMoves(moves='merge,delete')

    def test_MOVBWithShuffleMergeDeletes(self):
        self.runMany_MOVBWithMoves(moves='shuffle,merge,delete')

    def test_MOVBWithMerges_0ParallelWorkers(self):
        self.runMany_MOVBWithMoves(moves='merge', algName='pmoVB',
            nWorkers=0)

    def test_MOVBWithMerges_2ParallelWorkers(self):
        self.runMany_MOVBWithMoves(moves='merge', algName='pmoVB',
            nWorkers=2)