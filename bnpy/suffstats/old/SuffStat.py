'''
'''
import numpy as np
import copy

class SuffStat(object):

  @classmethod
  def MakeSuffStat(cls, *args, **kwargs):
    return cls(*args, **kwargs)

  def __init__(self, N=0):
    self.N = N
    
  def __add__(self, SS):
    dictA = self.toDict()
    dictB = SS.toDict()
    keyvalList = [ (k,dictA[k]+dictB[k]) for k in dictA if k in dictB]
    return self.MakeSuffStat( **dict(keyvalList) )
  
  def toDict(self):
    return self.__dict__
    
  def __str__(self):
    s = ''
    for (k,v) in self.toDict().items():
      s += '%s:%s\n' % (k, str(v) )
    return s
    
  def copy(self):
    return copy.deepcopy(self)
    