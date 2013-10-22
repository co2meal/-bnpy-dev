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

FieldNamesThatDoNotExpand = set(['N'])

class SuffStatDict(object):

  def __init__(self, K=None, doCheck=False, D=None, **kwArrArgs):
    self.__dict__ = dict(K=K, D=D)
    self.__compkeys__ = set()
    self.__scalars__ = dict()
    self.__precompELBOTerms__ = dict()
    self.__precompMerge__ = dict()
    self.__doCheck__ = doCheck
    for key, arr in kwArrArgs.items():
      self.__setattr__(key, arr)
        
  def copy(self):
    newSS = SuffStatDict()
    newSS.__dict__ = copy.deepcopy(self.__dict__)     
    return newSS
    
  def getComp(self, compID):
    if compID < 0 or compID >= self.K:
      raise IndexError('Bad compID. Valid range [0, %d] but provided %d' % (self.K-1, compID))
    compSS = SuffStatDict(K=1, D=self.D)
    for key in self.__compkeys__:
      if self.K == 1 or self.__dict__[key].size == 1:
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

  def subtractSpecificComponents(self, SSobj, compIDs):
    ''' Subtract (in-place) from specific components "compIDs" of this object
        the entire SuffStatDict object SSobj
    '''
    assert len(compIDs) == SSobj.K
    for key in self.__compkeys__:
      self.__dict__[key][compIDs] -= SSobj.__dict__[key]

  ######################################################### Insert comps
  #########################################################
  def insertComponents(self, SSextra):
    ''' Insert (in-place) all components from SSextra into this object
    '''
    if self.K == 1:
      self.expandSingletonDims()

    for key in self.__compkeys__:
      arrA = self.__dict__[key]
      arrB = SSextra.__dict__[key]
      arrC = np.insert(arrA, arrA.shape[0], arrB, axis=0)
      self.__dict__[key] = arrC
      # TODO: what about compkeys that are defined as KxK
    if self.hasPrecompEntropy():      
      key = '__precompEntropy__'
      arrA = self.__dict__[key]
      arrZ = np.zeros(SSextra.K, dtype=arrA.dtype)
      arrC = np.insert(arrA, arrA.shape[0], arrZ, axis=0)
      self.__dict__[key] = arrC
    if self.hasPrecompMergeEntropy():  
      key = '__mergeEntropy__'
      arrA = self.__dict__[key]
      bottomZ = np.zeros((SSextra.K, self.K), dtype=arrA.dtype)
      arrC = np.vstack( [arrA, bottomZ])
      rightZ = np.zeros((self.K + SSextra.K, SSextra.K), dtype=arrA.dtype)
      arrC = np.hstack( [arrC, rightZ])
      self.__dict__[key] = arrC  
    self.K = self.K + SSextra.K

  def insertEmptyComponents(self, Kextra):
    ''' Insert (in-place) Kextra empty components into this object
    '''
    if self.K == 1:
      self.expandSingletonDims()

    for key in self.__compkeys__:
      arrA = self.__dict__[key]
      if arrA.ndim == 3:
        myShape = (Kextra, arrA.shape[1], arrA.shape[2])
        zeroFill = np.zeros( myShape, dtype=arrA.dtype)
      elif arrA.ndim == 2:
        zeroFill = np.zeros( (Kextra, arrA.shape[1]), dtype=arrA.dtype)
      else:
        zeroFill = np.zeros(Kextra, dtype=arrA.dtype)
      arrC = np.insert(arrA, arrA.shape[0], zeroFill, axis=0)
      self.__dict__[key] = arrC
      # TODO: what about compkeys that are defined as KxK
    if self.hasPrecompEntropy():      
      key = '__precompEntropy__'
      arrA = self.__dict__[key]
      zeroFill = np.zeros(Kextra, dtype=arrA.dtype)
      arrC = np.insert(arrA, arrA.shape[0], zeroFill, axis=0)
      self.__dict__[key] = arrC
    if self.hasPrecompMergeEntropy():  
      key = '__mergeEntropy__'
      arrA = self.__dict__[key]
      zeroFillBottom = np.zeros((Kextra, self.K), dtype=arrA.dtype)
      arrC = np.vstack( [arrA, zeroFillBottom])
      zeroFillRight = np.zeros((self.K + Kextra, Kextra), dtype=arrA.dtype)
      arrC = np.hstack( [arrC, zeroFillRight])
      self.__dict__[key] = arrC  
    self.K = self.K + Kextra

  ######################################################### Insert comps
  #########################################################
  def addScalar(self, name, val):
    ''' Add scalar unattached to any component
    '''
    self.__dict__['__scalars__'][name] = val

  ######################################################### Remove comp
  #########################################################
  def removeComponent(self, kB):
    ''' Remove (in-place) the component kB from this SuffStatDict object.
    '''
    # Remove component from each suff stat field
    for key in self.__compkeys__:
      self.__dict__[key] = np.delete(self.__dict__[key], kB, axis=0)
      # TODO: what about compkeys that are defined as KxK
      # how to separate these from when field is KxD, but "coincidentally D=K"?
    # Remove component from precomputed entropy (Kx1 vector)
    if self.hasPrecompEntropy():
      key = '__precompEntropy__'
      self.__dict__[key] = np.delete(self.__dict__[key], kB, axis=0)
    # Remove component from precomputed merge entropy (KxK matrix)
    if self.hasPrecompMergeEntropy():
      key = '__mergeEntropy__'
      self.__dict__[key] = np.delete(self.__dict__[key], kB, axis=0)
      self.__dict__[key] = np.delete(self.__dict__[key], kB, axis=1)
    self.K = self.K - 1

    if self.K == 1:
      self.contractSingletonDims()

  def expandSingletonDims(self):
    for key in self.__compkeys__:
      if not key in FieldNamesThatDoNotExpand:
        newArr = self.__dict__[key][np.newaxis,:]
        self.__dict__[key] = newArr

  def contractSingletonDims(self):
    for key in self.__compkeys__:
      if not key in FieldNamesThatDoNotExpand:
        newArr = np.squeeze(self.__dict__[key])
        self.__dict__[key] = newArr


  ######################################################### Precomp ELBO terms
  #########################################################
  def hasPrecompELBO(self):
    keyList = self.__dict__['__precompELBOTerms__'].keys() 
    return len(keyList) > 0
    
  def hasPrecompELBOTerm(self, name):
    return name in self.__dict__['__precompELBOTerms__']

  def addPrecompELBOTerm(self, name, value):
    ''' Add a named term to precomputed ELBO terms
    '''
    self.__dict__['__precompELBOTerms__'][name] = np.asarray(value)

  def getPrecompELBOTerm(self, name):
    return self.__dict__['__precompELBOTerms__'][name]

  ######################################################### Precomp Merge terms
  #########################################################
  def hasPrecompMerge(self):
    keyList = self.__dict__['__precompMerge__'].keys() 
    return len(keyList) > 0
    
  def hasPrecompMergeTerm(self, name):
    return name in self.__dict__['__precompMerge__']

  def addPrecompMergeTerm(self, name, value):
    ''' Add a named term to precomputed merge terms
    '''
    self.__dict__['__precompMerge__'][name] = np.asarray(value)

  def getPrecompMergeTerm(self, name):
    return self.__dict__['__precompMerge__'][name]

  def setToZeroAllPrecompMergeTerms(self):
    ''' Zero out precomputed merge terms if not needed anymore
    '''
    for key in self.__dict__['__precompMerge__']:
      self.__dict__['__precompMerge__'][key].fill(0)

  ######################################################### Precomp Entropy
  #########################################################
  def hasPrecompEntropy(self):
    return '__precompEntropy__' in self.__dict__  
    
  def addPrecompEntropy(self, precompEntropyVec):
    self.__dict__['__precompEntropy__'] = np.asarray(precompEntropyVec)
  
  def getPrecompEntropy(self):
    return self.__dict__['__precompEntropy__']

  ######################################################### Precompute
  #########################################################  Merge Entropy
  def mergeComponents(self, kA, kB):
    ''' Merge (in-place) all additive fields for components kA, kB
        into field index kA, and remove component kB entirely.
    '''
    if not self.hasPrecompMergeEntropy():
      raise ValueError("Attribute merge entropy not defined, required for merge")
    if not self.hasPrecompEntropy():
      raise ValueError("Attribute precomp entropy not defined, required for merge")
    assert np.maximum(kA,kB) < self.K
    for key in self.__compkeys__:
      self.__dict__[key][kA] += self.__dict__[key][kB] 
    
    # Fix the precomputed entropy for new "merged" component kA    
    self.__dict__['__precompEntropy__'][kA] = self.__dict__['__mergeEntropy__'][kA,kB]

    # Remove kB entirely from this object
    #  this call automatically updates self.K to be one less    
    self.removeComponent(kB)

    # New "merged" component kA's entries in mergeEntropy
    # no longer represent the correct computation.
    key = '__mergeEntropy__'
    self.__dict__[key][kA,kA+1:] = np.nan
    self.__dict__[key][:kA,kA] = np.nan

  def setToZeroPrecompMergeEntropy(self):
    ''' Remove precomputed merge entropy if not needed anymore
    '''
    self.__dict__['__mergeEntropy__'].fill(0)

  def addPrecompMergeEntropy(self, Hmerge):
    ''' Add precomputed entropy for all possible merge pairs
        Args
        --------
        Hmerge : KxK matrix where 
                  Hmerge[i,j] = entropy if comps i,j were merged into one comp
    '''
    self.__dict__['__mergeEntropy__'] = np.asarray(Hmerge)

  def hasPrecompMergeEntropy(self):
    return  '__mergeEntropy__' in self.__dict__  

  def getPrecompMergeEntropy(self):
    return self.__dict__['__mergeEntropy__']

  ######################################################### Override
  #########################################################  add / subtract
  def __add__(self, SSobj):
    sumSS = SuffStatDict(K=self.K, D=self.D)
    for key in self.__compkeys__:
      sumSS[key] = self.__dict__[key] + SSobj[key]
    if self.hasPrecompELBO():
      for key in self.__dict__['__precompELBOTerms__']:
        arr1 = self.getPrecompELBOTerm(key)
        arr2 = SSobj.getPrecompELBOTerm(key)
        sumSS.addPrecompELBOTerm(key, arr1 + arr2)
    if self.hasPrecompEntropy():
      H1 = self.getPrecompEntropy()
      H2 = SSobj.getPrecompEntropy()
      sumSS.addPrecompEntropy(H1 + H2)
    if self.hasPrecompMergeEntropy() or SSobj.hasPrecompMergeEntropy():
      mergeH1 = self.getPrecompMergeEntropy()
      mergeH2 = SSobj.getPrecompMergeEntropy()
      sumSS.addPrecompMergeEntropy(mergeH1 + mergeH2)      
    return sumSS    
  
  def __sub__(self, SSobj):
    sumSS = SuffStatDict(K=self.K, D=self.D)
    for key in self.__compkeys__:
      sumSS[key] = self.__dict__[key] - SSobj[key]
    if self.hasPrecompELBO():
      for key in self.__dict__['__precompELBOTerms__']:
        arr1 = self.getPrecompELBOTerm(key)
        arr2 = SSobj.getPrecompELBOTerm(key)
        sumSS.addPrecompELBOTerm(key, arr1 - arr2)
    if self.hasPrecompEntropy():
      H1 = self.getPrecompEntropy()
      H2 = SSobj.getPrecompEntropy()
      sumSS.addPrecompEntropy(H1 - H2)
    if self.hasPrecompMergeEntropy() or SSobj.hasPrecompMergeEntropy():
      mergeH1 = self.getPrecompMergeEntropy()
      mergeH2 = SSobj.getPrecompMergeEntropy()
      sumSS.addPrecompMergeEntropy(mergeH1 - mergeH2)
    return sumSS    
       
       
  ######################################################### Override 
  #########################################################  getattr / setattr
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

    if self.K > 1 and arr.ndim == 0:
      self.addScalar(key, float(arr))    
      return
    if arr.ndim == 0:
      arr = arr[np.newaxis]
    if self.K is None:
      self.K = arr.shape[0]

    if self.__doCheck__ and self.K > 1 and arr.shape[0] != self.K and arr.size > 1:
      raise ValueError('Dimension mismatch. K=%d, Kfound=%d' % (self.K, arr.shape[0]))
    self.__compkeys__.add(key)
    self.__dict__[key] = arr
    
  def __getattr__(self, key):
    if key in self.__dict__['__scalars__']:
      return self.__dict__['__scalars__'][key]
    return self.__dict__[key]
    
  def __repr__(self):
    return self.__dict__.__repr__()
