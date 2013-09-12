'''
ZMGaussSuffStat.py
'''
import numpy as np
from SuffStat import SuffStat

class ZMGaussSuffStat(SuffStat):
  @classmethod
  def MakeEmpty(cls,D):
    return cls( N=0, xxT=np.zeros((D,D)))
  
  def __init__(self, N=0, xxT=0):
    self.N=N
    self.xxT = np.asarray(xxT)
    if self.xxT.ndim < 2:
      assert self.xxT.size==1
      self.xxT = np.asarray([[np.squeeze(xxT)]])
    self.D = self.xxT.shape[0]
    
  def toDict(self):
    return dict(N=self.N, xxT=self.xxT)