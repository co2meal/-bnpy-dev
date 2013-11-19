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
    setattr(self, key, self.parseArr(value, dims=dims, key=key))

  ######################################################### 
  #########################################################
  def parseArr(self, arr, dims=None, key=None):
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
    expectedShape = self._getExpectedShape(dims=dims)
    print arr.shape
    print expectedShape
    if arr.shape not in self._getAllowedShapes(expectedShape):
      self._raiseDimError(dims, arr, key)
    if arr.ndim == 0:
      arr = np.float64(arr)
    '''
    # Handle converted arr, case by case
    if arr.ndim == 0:
      if dims is not None and (K > 1):
        self._raiseDimError(dims, arr, key)
      else:
        arr = np.float64(arr)
    # ----------------------------------------------------- Dim 1
    elif arr.ndim == 1:
      if dims is None or dims[0] != 'K':
        self._raiseDimError(dims, arr, key)
      if len(dims) == 1 and arr.size != getattr(self, dims[0]):
        self._raiseDimError(dims, arr, key)
      if len(dims) == 2:
        expectedSize = getattr(self, dims[0])*getattr(self, dims[1])
        if (K > 1 and D> 1) or (arr.size != expectedSize):
          self._raiseDimError(dims, arr, key)
    # ----------------------------------------------------- Dim 2
    elif arr.ndim == 2:
      if dims is None or dims[0] != 'K' or len(dims) < 2 or len(dims) > 3:
        self._raiseDimError(dims, arr, key)
      if len(dims) == 3 and not K == 1:
        self._raiseDimError(dims, arr, key)
      expectedSize = getattr(self, dims[0])*getattr(self, dims[1])
      if len(dims) == 2 and arr.size != expectedSize:
        self._raiseDimError(dims, arr, key)
    # ----------------------------------------------------- Dim 3
    elif arr.ndim == 3:
      if dims is None or dims[0] != 'K' or len(dims) != 3:
        self._raiseDimError(dims, arr, key)
      expectedSize = getattr(self, dims[0]) * getattr(self, dims[1]) * getattr(self,dims[2])
      if arr.size != expectedSize:
        self._raiseDimError(dims, arr, key)
    else:
      raise NotImplementedError('Cannot handle more than 3 dims')
    '''
    if K == 1 or D == 1:
      arr = np.squeeze(arr)
    return arr

  def _getExpectedShape(self, key=None, dims=None):
    if key is not None:
      dims = self._FieldDims[key]
    if dims is None:
      expectShape = ()
    else:
      shapeList = list()
      for dim in dims:
        shapeList.append(getattr(self,dim))
      expectShape = tuple(shapeList)
    return expectShape    

  def _getAllowedShapes(self, shape):
    ''' Return set of allowed shapes that can be squeezed into given shape
        Examples
        --------
        () --> ()
        (1) --> (), (1)
        (2) --> (2)
        (3,1) -->  (3) or (3,1)
        (1,1) --> () or (1) or (1,1)
        (3,2) --> (3,2)
    '''
    allowedShapes = set()
    if len(shape) == 0:
      allowedShapes.add(tuple())
      return allowedShapes
    shapeVec = np.asarray(shape, dtype=np.int32)
    onesMask = shapeVec == 1
    keepMask = np.logical_not(onesMask)
    nOnes = sum(onesMask)
    for b in range(2**nOnes):
      bStr = np.binary_repr(b)
      bStr = '0'*(nOnes - len(bStr)) + bStr
      keepMask[onesMask] = np.asarray([int(x) > 0 for x in bStr])
      curShape = shapeVec[keepMask]
      allowedShapes.add(tuple(curShape))
    return allowedShapes

  def _raiseDimError(self, dims, badArr, key=None):
    expectShape = self._getExpectedShape(dims=dims)
    if key is None:
      msg = 'Bad Dims. Expected %s, got %s' % (expectShape, badArr.shape)
    else:
      msg = 'Bad Dims for field %s. Expected %s, got %s' % (key, expectShape, badArr.shape)
    raise ValueError(msg)

  def setAllFieldsToZero(self):
    for key, dims in self._FieldDims.items():
      curShape = getattr(self,key).shape
      self.setField(key, np.zeros(curShape), dims=dims)

  # ===================================================== Insert / Remove / Get
  def _getExpandedField(self, key, dims, K=None):
      if K is None:
        K = self.K
      arr = getattr(self, key)
      if arr.ndim < len(dims):
        for dimID, dimName in enumerate(dims):      
          if dimName == 'K' and K == 1:
            arr = np.expand_dims(arr, axis=dimID)
          elif getattr(self, dimName) == 1:
            arr = np.expand_dims(arr, axis=dimID)
      return arr
  
  def insertEmptyComps(self, Kextra):
    '''  Insert Kextra additional components to self in-place
    '''
    origK = self.K
    self.K += Kextra
    for key in self._FieldDims:
      dims = self._FieldDims[key]
      if dims is None:
        continue
      arr = self._getExpandedField(key, dims, K=origK)
      for dimID, dimName in enumerate(dims):      
        if dimName == 'K':
          curShape = list(arr.shape)
          curShape[dimID] = Kextra
          zeroFill = np.zeros(curShape)
          arr = np.append(arr, zeroFill, axis=dimID)
      self.setField(key, arr, dims=dims)

  def insertComps(self, PB):
    ''' Insert additional components (as ParamBag) to self in-place
    '''
    assert PB.D == self.D
    origK = self.K
    self.K += PB.K
    for key in self._FieldDims:
      dims = self._FieldDims[key]
      if dims is not None and dims[0] == 'K':
        arrA = self._getExpandedField(key, dims, K=origK)
        arrB = PB._getExpandedField(key, dims)
        arrC = np.append(arrA, arrB, axis=0)
        self.setField(key, arrC, dims=dims)
      elif dims is None:
        self.setField(key, getattr(self,key) + getattr(PB,key), dims=None)
 

  def removeComp(self, k):
    ''' Updates self in-place to remove component "k"
    '''
    if k < 0 or k >= self.K:
      msg = 'Bad compID. Expected [0, %d], got %d' % (self.K-1, k)
      raise IndexError(msg)
    if self.K <= 1:
      raise ValueError('Cannot remove final component.')
    self.K -= 1
    for key in self._FieldDims:
      arr = getattr(self,key)
      dims = self._FieldDims[key]
      if dims is not None:
        for dimID,name in enumerate(dims):
          if name == 'K':
            arr = np.delete(arr, k, axis=dimID)
        self.setField(key, arr, dims)


  def setComp(self, k, compPB):
    ''' Set (in-place) component k to the parameters in compPB object
    '''
    if k < 0 or k >= self.K:
      raise IndexError('Bad compID. Expected [0, %d] but provided %d' % (self.K-1, k))
    if compPB.K != 1:
      raise ValueError('Expected compPB to have K=1')
    for key, dims in self._FieldDims.items():
      if dims is None:
        self.setField(key, getattr(compPB,key), dims=None)
      elif self.K == 1: 
        self.setField(key, getattr(compPB,key), dims=dims)
      else:
        bigArr = getattr(self, key)
        bigArr[k] = getattr(compPB, key)

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

  # ======================================================= Override add
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


  # ======================================================= Override subtract
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
