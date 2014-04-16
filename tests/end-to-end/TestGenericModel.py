'''
Unit tests for running Learn.py

Verification for each possible model and learn algorithm

'''
import numpy as np
import unittest
from unittest.case import SkipTest

from bnpy.util import closeAtMSigFigs
from bnpy.Run import run
from bnpy.ContinueRun import continueRun

class TestGenericModel(unittest.TestCase):
  __test__ = False # Do not execute this abstract module!

  def get_kwargs_do_save_disk(self):
    kwargs = dict(saveEvery=1, printEvery=1, traceEvery=1)
    kwargs['doSaveToDisk'] = True
    kwargs['doWriteStdOut'] = False
    if hasattr(self, 'kwargs'):
      kwargs.update(self.kwargs)
    return kwargs

  def get_kwargs(self):
    kwargs = dict(saveEvery=-1, printEvery=-1, traceEvery=1)
    kwargs['doSaveToDisk'] = False
    kwargs['doWriteStdOut'] = False
    kwargs['convergeSigFig'] = 12
    if hasattr(self, 'kwargs'):
      kwargs.update(self.kwargs)
    return kwargs

  def verify_monotonic(self, ELBOvec):
    ''' Returns True if monotonically increasing, False otherwise.
    '''
    ELBOvec = np.asarray(ELBOvec, dtype=np.float64)
    assert ELBOvec.ndim == 1
    diff = ELBOvec[1:] - ELBOvec[:-1]
    maskIncrease = diff > 0
    maskWithinPercDiff = np.abs(diff)/np.abs(ELBOvec[:-1]) < 0.0000001
    mask = np.logical_or(maskIncrease, maskWithinPercDiff)
    mask = np.asarray(mask, dtype=np.float64)
    print "%.5e" % (np.abs(np.sum(mask) - float(diff.size)))
    return np.abs(np.sum(mask) - float(diff.size)) < 0.000001

  def test__verify_monotonic_catches_bad(self):
    assert self.verify_monotonic( [502.3, 503.1, 504.01, 504.00999999])
    assert not self.verify_monotonic( [502.3, 503.1, 504.01, 504.00989999])
    assert not self.verify_monotonic( [401.3, 400.99, 405.12])

  ######################################################### EM tests
  def test_EM__evidence_repeatable_and_monotonic(self):
    if 'EM' not in self.learnAlgs:
      raise SkipTest
    kwargs = self.get_kwargs()
    hmodel1, LP1, Info1 = run(self.Data, self.allocModelName,
                          self.obsModelName, 'EM', **kwargs)
    hmodel2, LP2, Info2 = run(self.Data, self.allocModelName,
                          self.obsModelName, 'EM', **kwargs)
    assert len(Info1['evTrace']) == len(Info2['evTrace'])
    assert np.allclose( Info1['evTrace'], Info2['evTrace'])

    self.verify_monotonic(Info1['evTrace'])

  ######################################################### VB tests

  def test_vb_repeatable(self):
    if 'VB' not in self.learnAlgs:
      raise SkipTest
    kwargs = self.get_kwargs()
    hmodel1, LP1, Info1 = run(self.Data, self.allocModelName,
                          self.obsModelName, 'VB', **kwargs)
    hmodel2, LP2, Info2 = run(self.Data, self.allocModelName,
                          self.obsModelName, 'VB', **kwargs)
    assert len(Info1['evTrace']) == len(Info2['evTrace'])
    assert np.allclose( Info1['evTrace'], Info2['evTrace'])


  def test_vb_repeatable_when_continued(self):
    if 'VB' not in self.learnAlgs:
      raise SkipTest
    kwargs = self.get_kwargs_do_save_disk()
    kwargs['nLap'] = 10
    hmodel1, LP1, Info1 = run(self.Data, self.allocModelName,
                          self.obsModelName, 'VB', **kwargs)
    kwargs['nLap'] = 5
    kwargs['startLap'] = 5
    hmodel2, LP2, Info2 = continueRun(self.Data, self.allocModelName,
                          self.obsModelName, 'VB', **kwargs)
    if hasattr(self, 'mustRetainLPAcrossLapsForGuarantees'):
      print Info1['evTrace'][-1]
      print Info2['evTrace'][-1]
      assert closeAtMSigFigs(Info1['evTrace'][-1], Info2['evTrace'][-1], M=2)

    else:
      assert Info1['evTrace'][-1] == Info2['evTrace'][-1]

  ######################################################### soVB tests
  def test_sovb_repeatable_across_diff_num_batches(self):
    if 'soVB' not in self.learnAlgs:
      raise SkipTest
    for nBatch in [1, 2, 3]:
      kwargs = self.get_kwargs()
      kwargs['nBatch'] = nBatch
      hmodel1, LP1, Info1 = run(self.Data, self.allocModelName,
                          self.obsModelName, 'soVB', **kwargs)
      hmodel2, LP2, Info2 = run(self.Data, self.allocModelName,
                          self.obsModelName, 'soVB', **kwargs)
      assert len(Info1['evTrace']) == len(Info2['evTrace'])
      assert np.allclose( Info1['evTrace'], Info2['evTrace'])

  def test_sovb_with_one_batch_equivalent_to_vb(self):
    if 'soVB' not in self.learnAlgs:
      raise SkipTest
    kwargs = self.get_kwargs()
    kwargs['nBatch'] = 1
    kwargs['rhoexp'] = 0
    __, __, vbInfo = run(self.Data, self.allocModelName,
                          self.obsModelName, 'VB', **kwargs)
    __, __, sovbInfo = run(self.Data, self.allocModelName,
                          self.obsModelName, 'soVB', **kwargs)
    vbEv = vbInfo['evTrace'][:-1]
    sovbEv = sovbInfo['evTrace']
    for ii in range(len(vbEv)):
      if hasattr(self, 'mustRetainLPAcrossLapsForGuarantees'):
        print vbEv[ii], sovbEv[ii]
        assert closeAtMSigFigs(vbEv[ii], sovbEv[ii], M=2)
      else:
        assert closeAtMSigFigs(vbEv[ii], sovbEv[ii], M=8)

  ######################################################### moVB tests
  def test_movb_repeatable_across_diff_num_batches(self):
    if 'moVB' not in self.learnAlgs:
      raise SkipTest
    for nBatch in [1, 2, 3]:
      kwargs = self.get_kwargs()
      kwargs['nBatch'] = nBatch
      hmodel1, LP1, Info1 = run(self.Data, self.allocModelName,
                          self.obsModelName, 'moVB', **kwargs)
      hmodel2, LP2, Info2 = run(self.Data, self.allocModelName,
                          self.obsModelName, 'moVB', **kwargs)
      assert len(Info1['evTrace']) == len(Info2['evTrace'])
      assert np.allclose( Info1['evTrace'], Info2['evTrace'])

  def test_movb_with_one_batch_equivalent_to_vb(self):
    if 'moVB' not in self.learnAlgs:
      raise SkipTest
    kwargs = self.get_kwargs()
    kwargs['nBatch'] = 1
    __, __, vbInfo = run(self.Data, self.allocModelName,
                          self.obsModelName, 'VB', **kwargs)
    __, __, movbInfo = run(self.Data, self.allocModelName,
                          self.obsModelName, 'moVB', **kwargs)
    vbEv = vbInfo['evTrace'][:-1]
    movbEv = movbInfo['evTrace']
    print vbEv
    print movbEv
    assert len(vbEv) == len(movbEv)
    for ii in range(len(vbEv)):
      assert closeAtMSigFigs(vbEv[ii], movbEv[ii], M=8)

  ######################################################### ELBO tests
  def test_vb_sovb_and_movb_all_estimate_evBound_in_same_ballpark(self):
    kwargs = self.get_kwargs()
    kwargs['nBatch'] = 5
    __, __, vbInfo = run(self.Data, self.allocModelName,
                          self.obsModelName, 'VB', **kwargs)
    __, __, movbInfo = run(self.Data, self.allocModelName,
                          self.obsModelName, 'moVB', **kwargs)
    __, __, sovbInfo = run(self.Data, self.allocModelName,
                          self.obsModelName, 'soVB', **kwargs)
    vbEv = vbInfo['evBound']
    movbEv = movbInfo['evBound']
    sovbEv = np.mean(sovbInfo['evTrace'][-10:])

    print vbEv
    print movbEv
    print sovbEv
    assert closeAtMSigFigs(vbEv, movbEv, M=2)
    assert closeAtMSigFigs(vbEv, sovbEv, M=2)




