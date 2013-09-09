'''
'''
from bnpy.suffstats import ZMGaussSuffStat
import numpy as np

class TestZMGaussSuffStat(object):
  def setUp(self):
    self.SS = ZMGaussSuffStat(N=10, xxT=np.eye(3))
    
  def test_additivity(self):
    Sboth = self.SS + self.SS
    assert Sboth.N == self.SS.N + self.SS.N
    assert np.allclose(Sboth.xxT, self.SS.xxT + self.SS.xxT)