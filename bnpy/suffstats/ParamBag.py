'''
ParamBag.py

Container object for related groups of parameters.
For example, can group "means" and "variances" together.

Create a new bag with
  PB = ParamBag(K=5, D=3)

Add new fields with
  PB.setField('name', value)
  PB.setField('name', value, dims=('K','D'))

Access fields using just attribute notation
  PB.N
  PB.x

Access all fields related to single component with
  PB.getComp(compID)
'''
import numpy as np
import copy

class ParamBag(object):
  def __init__(self, K=0, D=0):
    self.K = K
    self.D = D
    self._FieldDims = dict()

  def copy(self):
    return copy.deepcopy(self)

  def setField(self, key, value, dims=None):
    ''' Set a named field to particular array value.
    '''
    # Parse dims tuple
    if dims is None and key in self._FieldDims:
      dims = self._FieldDims[key]
    else:
      self._FieldDims[key] = dims
    # Parse value as numpy array
    setattr(self, key, self.parseArr(value, dims))

  def parseArr(self, arr, dims=None):
    ''' Parse provided array-like variable into a standard numpy array
        with provided dimensions "dims", as a tuple

        Returns
        --------
        numpy array with expected dimensions
    '''
    K = self.K
    D = self.D
    if dims is not None:
      if type(dims) == str:
        dims = (dims) # force to tuple
    arr = np.asarray(arr, dtype=np.float64)
    # Handle converted arr, case by case
    if arr.ndim == 0:
      if dims is not None and (K > 1):
          raise ValueError('Field has dim 0 but needs dim %d' % (K))
      else:
        arr = np.float64(arr)
    # ----------------------------------------------------- Dim 1
    elif arr.ndim == 1:
      if dims is None or dims[0] != 'K':
        raise ValueError('Bad dim %s %s' % (arr.ndim, arr.size))
      if len(dims) == 1 and arr.size != K:
          raise ValueError('Bad dim %s %s' % (arr.ndim, arr.size))
      if len(dims) == 2 and arr.size != K * D:
          raise ValueError('Bad dim %s %s' % (arr.ndim, arr.size))
      if len(dims) == 2 and (K > 1 and D > 1):
          raise ValueError('Bad dim %s %s. Expected 2-dim.' % (arr.ndim, arr.size))
      if K == 1 or D == 1:
        arr = np.squeeze(arr)
    # ----------------------------------------------------- Dim 2
    elif arr.ndim == 2:
      if dims is None or dims[0] != 'K' or len(dims) < 2 or len(dims) > 3:
        raise ValueError('Bad dim: %s %s' % (arr.ndim, arr.size))
      if len(dims) == 3 and not K == 1:
        raise ValueError('Bad dim: %s %s' % (arr.ndim, arr.size))
      expectedSize = getattr(self, dims[0])*getattr(self, dims[1])
      if len(dims) == 2 and arr.size != expectedSize:
          raise ValueError('Bad dim: %s %s' % (arr.ndim, arr.size))
      if K == 1 or D == 1:
        arr = np.squeeze(arr)
    # ----------------------------------------------------- Dim 3
    elif arr.ndim == 3:
      if dims is None or dims[0] != 'K' or len(dims) != 3:
        raise ValueError('Bad dim: %s %s' % (arr.ndim, arr.size))
      expectedSize = getattr(self, dims[0]) * getattr(self, dims[1]) * getattr(self,dims[2])
      if arr.size != expectedSize:
        raise ValueError('Bad dim: %s %s' % (arr.ndim, arr.size))
      if K == 1 or D == 1:
        arr = np.squeeze(arr)
    return arr

  # ===================================================== Insert / Remove / Get
  def insertComps(self, PB):
    ''' Insert ParamBag fields to self in-place
    '''
    assert PB.D == self.D
    self.K += PB.K
    for key in self._FieldDims:
      dims = self._FieldDims[key]
      if dims is not None and dims[0] == 'K':
        arrA = getattr(self, key)
        arrB = getattr(PB, key)
        if arrA.ndim == arrB.ndim:
          arrC = np.append(arrA, arrB, axis=0)
        elif self.K == 1 and PB.K > 1:
          arrC = np.insert(arrB, 0, arrA, axis=0)
        elif PB.K == 1 and self.K > 1:
          arrC = np.insert(arrA, arrA.shape[0], arrB, axis=0)
        else:
          arrC = np.insert(arrA[np.newaxis,:], 1, arrB[np.newaxis,:], axis=0)
        self.setField(key, arrC, dims)
        

  def removeComp(self, k):
    ''' Updates self in-place to remove component "k"
    '''
    if k < 0 or k >= self.K:
      raise IndexError('Bad compID. Expected [0, %d] but got %d' % (self.K-1, k))
    self.K -= 1
    for key in self._FieldDims:
      arr = getattr(self,key)
      dims = self._FieldDims[key]
      if dims is not None:
        for dimID,name in enumerate(dims):
          if name == 'K':
            arr = np.delete(arr, k, axis=dimID)
        self.setField(key, arr, dims)


  def getComp(self, k):
    ''' Returns ParamBag object for component "k"
    '''
    if k < 0 or k >= self.K:
      raise IndexError('Bad compID. Expected [0, %d] but provided %d' % (self.K-1, k))
    cPB = ParamBag(K=1, D=self.D)
    for key in self._FieldDims:
      arr = getattr(self,key)
      dims = self._FieldDims[key]
      if dims is not None:
        if self.K == 1:
          cPB.setField(key, arr, dims=dims)
        else:
          cPB.setField(key, arr[k], dims=dims)
      else:
        cPB.setField(key, arr)
    return cPB

  # ======================================================= Override add / subtract
  def __add__(self, PB):
    if self.K != PB.K or self.D != PB.D:
      raise ValueError('Dimension mismatch')
    PBsum = ParamBag(K=self.K, D=self.D)
    for key in self._FieldDims:
      arrA = getattr(self, key)
      arrB = getattr(PB, key)
      PBsum.setField(key, arrA + arrB, dims=self._FieldDims[key])
    return PBsum

  def __iadd__(self, PB):
    if self.K != PB.K or self.D != PB.D:
      raise ValueError('Dimension mismatch')
    for key in self._FieldDims:
      arrA = getattr(self, key)
      arrB = getattr(PB, key)
      self.setField(key, arrA + arrB)
    return self

  def __sub__(self, PB):
    if self.K != PB.K or self.D != PB.D:
      raise ValueError('Dimension mismatch')
    PBdiff = ParamBag(K=self.K, D=self.D)
    for key in self._FieldDims:
      arrA = getattr(self, key)
      arrB = getattr(PB, key)
      PBdiff.setField(key, arrA - arrB, dims=self._FieldDims[key])
    return PBdiff

  def __isub__(self, PB):
    if self.K != PB.K or self.D != PB.D:
      raise ValueError('Dimension mismatch')
    for key in self._FieldDims:
      arrA = getattr(self, key)
      arrB = getattr(PB, key)
      self.setField(key, arrA - arrB)
    return self
