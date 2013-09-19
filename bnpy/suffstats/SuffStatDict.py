'''
SuffStatDict.py

Container object for sufficient statistics for bnpy models.
Each field should be an *additive* sufficient statistic,
  represented by a numpy array
  where first dimension is the number of components.
  
For example, to create suff stats for a 2-component mixture model with 10 and 50 members in each component
>> S = SuffStatDict(N=[10, 50])  
Next, add the covariance suff stats xxT
>> SS.xxT = np.dstack([10*np.eye(2), 50*np.eye(2)])
To get the second component (index = 2-1 = 1)
>> S1 = SS.getComp(1)
>> print S1
dict(N=50, xxT=[[50,0],[0,50]]]  
  
Acts like a dictionary, so 
  can add new fields with SS['N'] = [1,2,3,4]
To access the suff stats for a single component,
 getComp(compID) returns a SuffStatDict object associated with component compID 
'''

import numpy as np
import copy

class SuffStatDict(object):

  def __init__(self, K=None, D=None, **kwArrArgs):
    self.__dict__ = dict(K=K, D=D)
    self.__compkeys__ = set()
    for key, arr in kwArrArgs.items():
      self.__setattr__(key, arr)
        
  def getComp(self, compID):
    if compID < 0 or compID >= self.K:
      raise IndexError('Bad compID. Valid range [0, %d] but provided %d' % (self.K-1, compID))
    compSS = SuffStatDict(K=1, D=self.D)
    for key in self.__compkeys__:
      if self.K == 1:
        compSS[key] = self.__dict__[key]
      else:
        compSS[key] = self.__dict__[key][compID]
    compSS.__compkeys__ = self.__compkeys__
    return compSS

  def hasAmpFactor(self):
    return 'ampF' in self.__dict__
        
  def applyAmpFactor(self, ampF):
    self.__dict__['ampF'] = ampF
    for key in self.__compkeys__:
      self.__dict__[key] *= ampF      
    
  def hasPrecompEntropy(self):
    return '__precompEntropy__' in self.__dict__  
    
  def addPrecompEntropy(self, Hvec):
    self.__dict__['__precompEntropy__'] = Hvec
  
  def getPrecompEntropy(self):
    return self.__dict__['__precompEntropy__']
       
  def copy(self):
    newSS = SuffStatDict()
    newSS.__dict__ = copy.deepcopy(self.__dict__)     
    return newSS
       
  def __add__(self, SSobj):
    sumSS = SuffStatDict(K=self.K, D=self.D)
    for key in self.__compkeys__:
      sumSS[key] = self.__dict__[key] + SSobj[key]
    if self.hasPrecompEntropy():
      sumSS.addPrecompEntropy(self.getPrecompEntropy() + SSobj.getPrecompEntropy())   
    return sumSS    
  
  def __sub__(self, SSobj):
    sumSS = SuffStatDict(K=self.K, D=self.D)
    for key in self.__compkeys__:
      sumSS[key] = self.__dict__[key] - SSobj[key]
      
    if self.hasPrecompEntropy():
      sumSS.addPrecompEntropy(self.getPrecompEntropy() - SSobj.getPrecompEntropy()) 
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
      super(SuffStatDict, self).__setattr__(key, arr)
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

    if self.K > 1 and arr.shape[0] != self.K and arr.size > 1:
      raise ValueError('Dimension mismatch. K=%d, Kfound=%d' % (self.K, arr.shape[0]))
    self.__compkeys__.add(key)
    self.__dict__[key] = arr
    
  def __getattr__(self, key):
    return self.__dict__[key]
    
  def __repr__(self):
    return self.__dict__.__repr__()