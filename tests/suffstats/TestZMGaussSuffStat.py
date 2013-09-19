'''
'''
from bnpy.suffstats import SuffStatDict
import numpy as np

class TestZMGaussSuffStat(object):
  def setUp(self):
    self.SS = SuffStatDict(K=2, N=[10,20], xxT=np.random.rand(2, 4, 4))
     
  def test_additivity(self):
    Sboth = self.SS + self.SS
    assert np.allclose(Sboth.N, self.SS.N + self.SS.N)
    assert np.allclose(Sboth.xxT, self.SS.xxT + self.SS.xxT)
    assert Sboth.K == self.SS.K

class TestZMGaussSuffStatK1(object):
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