'''
Unit tests for running Learn.py

Verification for each possible model and learn algorithm that
*  

'''
import bnpy
import Learn
import unittest

import numpy as np

class TestGenericModel(unittest.TestCase):
  def setUp(self):
    self.__test__ = False

  def get_kwargs(self):
    kwargs = dict(saveEvery=-1, printEvery=-1, traceEvery=1)
    kwargs['doSaveToDisk'] = False
    kwargs['doWriteStdOut'] = False
    if hasattr(self, 'kwargs'):
      kwargs.update(self.kwargs)
    return kwargs

  def test_sovb_repeatable_across_diff_num_batches(self):
    for nBatch in [1, 2, 3]:
      kwargs = self.get_kwargs()
      kwargs['nBatch'] = nBatch
      hmodel1, LP1, Info1 = Learn.run(self.Data, self.allocModelName,
                          self.obsModelName, 'soVB', **kwargs)
      hmodel2, LP2, Info2 = Learn.run(self.Data, self.allocModelName,
                          self.obsModelName, 'soVB', **kwargs)
      assert len(Info1['evTrace']) == len(Info2['evTrace'])
      assert np.allclose( Info1['evTrace'], Info2['evTrace'])

  def test_movb_repeatable_across_diff_num_batches(self):
    for nBatch in [1, 2, 3]:
      kwargs = self.get_kwargs()
      kwargs['nBatch'] = nBatch
      hmodel1, LP1, Info1 = Learn.run(self.Data, self.allocModelName,
                          self.obsModelName, 'moVB', **kwargs)
      hmodel2, LP2, Info2 = Learn.run(self.Data, self.allocModelName,
                          self.obsModelName, 'moVB', **kwargs)
      assert len(Info1['evTrace']) == len(Info2['evTrace'])
      assert np.allclose( Info1['evTrace'], Info2['evTrace'])

  def test_vb_repeatable(self):
    kwargs = self.get_kwargs()
    hmodel1, LP1, Info1 = Learn.run(self.Data, self.allocModelName,
                          self.obsModelName, 'VB', **kwargs)
    hmodel2, LP2, Info2 = Learn.run(self.Data, self.allocModelName,
                          self.obsModelName, 'VB', **kwargs)
    assert len(Info1['evTrace']) == len(Info2['evTrace'])
    assert np.allclose( Info1['evTrace'], Info2['evTrace'])

  def test_vb_sovb_and_movb_all_estimate_evBound_in_same_ballpark(self):
    kwargs = self.get_kwargs()
    kwargs['nBatch'] = 5
    __, __, vbInfo = Learn.run(self.Data, self.allocModelName,
                          self.obsModelName, 'VB', **kwargs)
    __, __, movbInfo = Learn.run(self.Data, self.allocModelName,
                          self.obsModelName, 'moVB', **kwargs)
    __, __, sovbInfo = Learn.run(self.Data, self.allocModelName,
                          self.obsModelName, 'soVB', **kwargs)
    vbEv = vbInfo['evBound']
    movbEv = movbInfo['evBound']
    sovbEv = np.mean(sovbInfo['evTrace'][-10:])

    print vbEv
    print movbEv
    print sovbEv
    assert bnpy.util.closeAtMSigFigs(vbEv, movbEv, M=2)
    assert bnpy.util.closeAtMSigFigs(vbEv, sovbEv, M=2)

  def test_movb_with_one_batch_equivalent_to_vb(self):
    kwargs = self.get_kwargs()
    kwargs['nBatch'] = 1
    __, __, vbInfo = Learn.run(self.Data, self.allocModelName,
                          self.obsModelName, 'VB', **kwargs)
    __, __, movbInfo = Learn.run(self.Data, self.allocModelName,
                          self.obsModelName, 'moVB', **kwargs)
    vbEv = vbInfo['evTrace'][:-1]
    movbEv = movbInfo['evTrace']
    assert len(vbEv) == len(movbEv)
    for ii in range(len(vbEv)):
      assert bnpy.util.closeAtMSigFigs(vbEv[ii], movbEv[ii], M=8)


  def test_sovb_with_one_batch_equivalent_to_vb(self):
    kwargs = self.get_kwargs()
    kwargs['nBatch'] = 1
    kwargs['rhoexp'] = 0
    __, __, vbInfo = Learn.run(self.Data, self.allocModelName,
                          self.obsModelName, 'VB', **kwargs)
    __, __, sovbInfo = Learn.run(self.Data, self.allocModelName,
                          self.obsModelName, 'soVB', **kwargs)
    vbEv = vbInfo['evTrace'][:-1]
    sovbEv = sovbInfo['evTrace']
    for ii in range(len(vbEv)):
      assert bnpy.util.closeAtMSigFigs(vbEv[ii], sovbEv[ii], M=8)