'''
Unit tests for SuffStatDict
'''
from bnpy.suffstats import SuffStatDict
import numpy as np
import unittest

class TestSSK2(unittest.TestCase):
  def shortDescription(self):
    return None

  def setUp(self):
    self.SS = SuffStatDict(K=2, N=[10,20], xxT=np.random.rand(2, 4, 4))
     
  def test_additivity(self):
    Sboth = self.SS + self.SS
    assert np.allclose(Sboth.N, self.SS.N + self.SS.N)
    assert np.allclose(Sboth.xxT, self.SS.xxT + self.SS.xxT)
    assert Sboth.K == self.SS.K

  def test_merge_component_without_entropy_raises_error(self):
    ''' Verify that calling mergeComponents without defining entropy
        raises an exception
    '''
    with self.assertRaises(ValueError) as context:
      self.SS.mergeComponents(0,1)

  def test_merge_component(self):
    MSS = self.SS.copy()
    C0 = self.SS.getComp(0)
    C1 = self.SS.getComp(1)
    K = MSS.K
    MSS.addPrecompEntropy( np.ones(K))
    MSS.addPrecompMergeEntropy( np.ones((K,K)))
    MSS.mergeComponents(0, 1)
    assert np.allclose(MSS.N, C0.N + C1.N)
    assert np.allclose(MSS.xxT, C0.xxT + C1.xxT)
    assert MSS.K == self.SS.K - 1

  def test_remove_component(self):
    ''' Verify that removing a component leaves expected remaining attribs intact.
    '''
    SS = self.SS
    SS1 = SS.getComp(1)    
    SS.removeComponent(0)
    assert SS.K == 1
    assert np.allclose(SS.N, SS1.N)
    assert np.allclose(SS.xxT, SS1.xxT)

class TestSSK1(unittest.TestCase):
  def shortDescription(self):
    return None

  def setUp(self):
    self.SS = SuffStatDict(K=1, N=[10], xxT=np.random.rand(5,5))
    
  def test_getComp(self):
    ''' Verify getComp(0) on a K=1 SuffStatDict yields same exact object
    '''
    Scomp = self.SS.getComp(0)
    keysA = np.unique(Scomp.__dict__.keys())
    keysB = np.unique(self.SS.__dict__.keys())
    print keysA
    print keysB
    assert len(keysA) == len(keysB)
    for a in range(len(keysA)):
      assert keysA[a] == keysB[a]
    for key in keysA:
      print Scomp[key], self.SS[key]
      if type(Scomp[key]) == np.float32:
        assert np.allclose(Scomp[key], self.SS[key])
        self.SS[key] *= 10
        assert not np.allclose(Scomp[key], self.SS[key])
    
  def test_additivity(self):
    Sadd = self.SS + self.SS
    assert np.allclose(Sadd.N, self.SS.N * 2)
    assert np.allclose(Sadd.xxT, self.SS.xxT * 2)
    assert Sadd.K == self.SS.K

