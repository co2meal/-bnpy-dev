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
    print " %25s after %4.1f sec  ELBO=% 7.3f  nLap=%5d  Kfinal=%d  Ktrue=%d"\
     % (Info['status'][:25],
        Info['elapsedTimeInSec'],
        Info['evBound'],
        Info['lapTrace'][-1],
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
    kwargs['doWriteStdOut'] = True
    kwargs['printEvery'] = 1
    kwstr = ' '.join(['--%s %s' % (key, kwargs[key]) for key in kwargs])
    print "python -m bnpy.Run %s %s %s %s %s" % (
        dataArg['name'],
        aArg['name'], 
        oArg['name'],
        algName, 
        kwstr,
        )

def is_monotonic(ELBOvec, atol=1e-6, verbose=True):
    ''' Returns True if provided vector monotonically increases, False o.w. 

    Returns
    -------
    result : boolean (True or False)
    '''
    ELBOvec = np.asarray(ELBOvec, dtype=np.float64)
    assert ELBOvec.ndim == 1
    diff = ELBOvec[1:] - ELBOvec[:-1]
    maskIncrease = diff > 0
    maskWithinTol = np.abs(diff) < atol
    maskOK = np.logical_or(maskIncrease, maskWithinTol)
    isMonotonic = np.all(maskOK)
    if not isMonotonic and verbose:
        print "NOT MONOTONIC!"
        print '  %d violations found in vector of size %d.' % (
            np.sum(1 - maskOK), ELBOvec.size)
        from IPython import embed; embed()
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

    def makeAllKwArgs(self, aArg, obsArg, initArg=dict(), **kwargs):
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
        )
        allKwargs.update(kwargs)
        allKwargs.update(aArg)
        allKwargs.update(obsArg)
        allKwargs.update(initArg)

        if aArg['name'] == 'HDPTopicModel':
            allKwargs['mergePairSelection'] = 'corrlimitdegree'            
        else:
            allKwargs['mergePairSelection'] = 'wholeELBObetter'
        return allKwargs

    def run_MOVBWithMerges_truelabels(self, aArg, oArg,
            **kwargs):
        """ Execute single run with merge moves enabled.

        Post Condition
        --------------
        Will raise AssertionError if any bad results detected.
        """
        Ktrue = self.Data.TrueParams['K']
        algName = 'moVB'
        pprint(aArg)
        pprint(oArg)

        initArg = dict(**kwargs)
        kwargs = self.makeAllKwArgs(aArg, oArg, initArg, 
            mergeStartLap=2, moves='merge')
        model, Info = bnpy.run(self.Data, 
            arg2name(aArg), arg2name(oArg), algName, **kwargs)
        pprintResult(model, Info, Ktrue=Ktrue)

        afterFirstLapMask = Info['lapTrace'] >= 1.0
        evTraceAfterFirstLap = Info['evTrace'][afterFirstLapMask]
        isMonotonic = is_monotonic(evTraceAfterFirstLap)

        try:
            assert isMonotonic
            assert model.allocModel.K == Ktrue
            assert model.allocModel.K == model.obsModel.K
        except AssertionError as e:
            pprintCommandToReproduceError(self.datasetArg, aArg, oArg, algName, **kwargs)
            raise(e)


    def test_MOVBWithMerges_truelabelsandempty(self):
        print ''
        for aKwArgs in self.nextAllocKwArgsForVB():
            for oKwArgs in self.nextObsKwArgsForVB():
                self.run_MOVBWithMerges_truelabels(
                    aKwArgs, oKwArgs, 
                    initname='truelabelsandempty',
                    initKextra=2)

    def test_MOVBWithMerges_truelabels(self):
        print ''
        for aKwArgs in self.nextAllocKwArgsForVB():
            for oKwArgs in self.nextObsKwArgsForVB():
                self.run_MOVBWithMerges_truelabels(
                    aKwArgs, oKwArgs, initname='truelabels')

    def test_MOVBWithMerges_repeattruelabels(self):
        print ''
        for aKwArgs in self.nextAllocKwArgsForVB():
            for oKwArgs in self.nextObsKwArgsForVB():
                self.run_MOVBWithMerges_truelabels(
                    aKwArgs, oKwArgs, initname='repeattruelabels')
