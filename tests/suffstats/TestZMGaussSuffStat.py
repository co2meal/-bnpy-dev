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
