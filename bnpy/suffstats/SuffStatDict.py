'''
SuffStatDict.py
'''

import numpy as np
import copy

class SuffStatDict(object):

  def __init__(self, K=None, D=None, **kwArrArgs):
    self.__dict__ = dict(K=K, D=D)
    self.__compkeys__ = set()
    for key, arr in kwArrArgs.items():
      print "IN ARG:", key, arr
      self.__setattr__(key, arr)
        
  def getComp(self, compID):
    compSS = SS(K=1, D=self.D)
    for key in self.__compkeys__:
      compSS[key] = self.__dict__[key][compID]
    return compSS
        
  def applyAmpFactor(self, ampF):
    for key in self.__compkeys__:
      self.__dict__[key] *= ampF      
   
  def __add__(self, SSobj):
    sumSS = SS(K=self.K, D=self.D)
    for key in self.__compkeys__:
      sumSS[key] = self.__dict__[key] + SSobj[key]    
    return sumSS    
  
  def __sub__(self, SSobj):
    sumSS = SS(K=self.K, D=self.D)
    for key in self.__compkeys__:
      sumSS[key] = self.__dict__[key] - SSobj[key]    
    return sumSS    
        
        
  def __getitem__(self, key):
    if type(key) == int:
      return self.getComp(key)
    else:
      return self.__getattr__(key) 
       
  def __setitem__(self, key, arr):
    self.__setattr__(key,arr)           
        
  def __setattr__(self, key, arr):
    if key == '__dict__':
      super(SS, self).__setattr__(key, arr)
      return
    elif key.startswith('__'):
      self.__dict__[key] = arr
      return
    if key.startswith('K'):
      self.__dict__['K'] = arr
      return
    if key.startswith('D'):
      self.__dict__['D'] = arr
      return
    arr = np.asarray(arr)
    if arr.ndim == 0:
      arr = arr[np.newaxis]
    if self.K is None:
      self.K = arr.shape[0]
    if arr.ndim == 2 and self.K == 1 and arr.size > self.K:
      arr = arr[np.newaxis, :, :]
    if arr.ndim == 1 and self.K == 1 and arr.size > self.K:
      arr = arr[np.newaxis, :]

    if arr.shape[0] != self.K and arr.size > 1:
      raise ValueError('Dimension mismatch. K=%d, Kfound=%d' % (self.K, arr.shape[0]))
    if arr.shape[0] == self.K:
      print 'adding key!', key
      self.__compkeys__.add(key)
    self.__dict__[key] = arr
    
  def __getattr__(self, key):
    return self.__dict__[key]
    
  def __repr__(self):
    return self.__dict__.__repr__()