'''
GroupXData.py

Data object holding a collection of dense real observations, organized by a 
provided group structure. 

Attributes
--------
X : 2D array, size nObs x D
doc_range : 1D array, size nDoc + 1
nDoc  : int number of documents (groups) in current dataset (in-memory)
nDocTotal : int total number of documents, including current in-memory batch
nObs : int total number of unique observations in the current, in-memory batch
TrueParams : (optional) dict
summary : (optional) string providing human-readable description of this data

Example
--------
>> import numpy as np
>> from bnpy.data import GroupData
>> X = np.random.randn(1000, 3) # Create 1000x3 matrix
>> doc_range = [0, 500, 1000] # assign obs 0-499 to doc 1, 500-1000 to doc 2
>> myData = GroupXData(X, doc_range)
>> print myData.nObs
1000
>> print myData.D
3
>> print myData.X.shape
(1000,3)
>> print myData.nDoc
2
'''

import numpy as np
from XData import XData
from bnpy.util import as1D

def _toStdArray(X):
  if X.dtype.byteorder != '=':
    X = X.newbyteorder('=').copy()
  X = np.asarray_chkfinite(X, dtype=np.float64, order='C')
  return X

def _toStd1DArray(X):
  if X.ndim > 1:
    X = np.squeeze(X)
  assert X.ndim == 1
  return _toStdArray(X)

def _toStd2DArray(X):
  if X.ndim < 2:
    X = X[np.newaxis,:]
  assert X.ndim == 2
  return _toStdArray(X)

class GroupXData(XData):
  
  @classmethod
  def LoadFromFile(cls, filepath, nDocTotal=None, **kwargs):
    ''' Static constructor for loading data from disk into XData instance
    '''
    if filepath.endswith('.mat'):
      return cls.read_from_mat(filepath, nDocTotal=nDocTotal, **kwargs)
    raise NotImplemented('Only .mat file supported.')

  def save_to_mat(self, matfilepath):
    ''' Save contents of current object to disk
    '''
    import scipy.io
    SaveVars = dict(X=self.X, nDoc=self.nDoc, doc_range=self.doc_range)
    if hasattr(self, 'Xprev'):
      SaveVars['Xprev'] = self.Xprev
    if hasattr(self, 'TrueParams') and 'Z' in self.TrueParams:
      SaveVars['TrueZ'] = self.TrueParams['Z']
    scipy.io.savemat(matfilepath, SaveVars, oned_as='row')

  @classmethod
  def read_from_mat(cls, matfilepath, nDocTotal=None, **kwargs):
    ''' Static Constructor for building an instance of GroupXData from disk
    '''
    import scipy.io
    InDict = scipy.io.loadmat(matfilepath, **kwargs)
    if 'X' not in InDict:
      raise KeyError('Stored matfile needs to have data in field named X')
    if 'doc_range' not in InDict:
      raise KeyError('Stored matfile needs to have field named doc_range')
    if nDocTotal is not None:
      InDict['nDocTotal'] = nDocTotal
    return cls(**InDict)
  
  def __init__(self, X=None, doc_range=None, nDocTotal=None, 
                     Xprev=None, TrueZ=None, 
                     TrueParams=None, fileNames=None, summary=None, **kwargs):
    ''' Create an instance of GroupXData for provided array X

        Reallocation of memory may occur, to ensure that X is a 2D numpy array
        with standardized data-type, byteorder, contiguity, and ownership.
    '''
    self.X = _toStd2DArray(X)
    self.doc_range = as1D(np.asarray(doc_range, dtype=np.int32, order='C'))
    if summary is not None:
      self.summary = summary
    if Xprev is not None:
      self.Xprev = _toStd2DArray(Xprev)

    ## Verify attributes are consistent
    self._set_dependent_params(doc_range, nDocTotal)
    self._check_dims()

    ## Add optional true parameters / true hard labels
    if TrueParams is not None:
      self.TrueParams = dict()
      for key, arr in TrueParams:
        self.TrueParams[key] = _toStdArray(arr)

    if TrueZ is not None:
      if not hasattr(self, 'TrueParams'):
        self.TrueParams = dict()
      self.TrueParams['Z'] = _toStd1DArray(TrueZ)

    ## Add optional source files for each group/sequence
    if fileNames is not None:
      self.fileNames = [str(x).strip() for x in np.squeeze(fileNames)]

  def _set_dependent_params(self, doc_range, nDocTotal=None): 
    self.nObs = self.X.shape[0]
    self.dim = self.X.shape[1]
    self.nDoc = self.doc_range.size - 1
    if nDocTotal is None:
      self.nDocTotal = self.nDoc
    else:
      self.nDocTotal = nDocTotal

  def _check_dims( self ):
    assert self.X.ndim == 2
    assert self.X.flags.c_contiguous
    assert self.X.flags.owndata
    assert self.X.flags.aligned
    assert self.X.flags.writeable

    assert self.doc_range.ndim == 1
    assert self.doc_range.size == self.nDoc + 1    
    assert self.doc_range[0] == 0
    assert self.doc_range[-1] == self.nObs
    assert np.all(self.doc_range[1:] - self.doc_range[:-1] >= 1)

  def get_size(self):
    return self.nDoc

  def get_total_size(self):
    return self.nDocTotal

  def get_dim(self):
    return self.dim

  def get_text_summary(self):
    ''' Returns human-readable description of this dataset
    '''
    if hasattr(self, 'summary'):
      s = self.summary
    else:
      s = 'GroupXData'
    return s

  def get_stats_summary(self):
    ''' Returns human-readable summary of this dataset's basic properties
    '''
    s = '  size: %d units (documents)\n' % (self.get_size())
    s += '  dimension: %d' % (self.get_dim())
    return s

  def toXData(self):
    ''' Return simplified XData instance, losing group structure
    '''
    if hasattr(self, 'Xprev'):
      return XData(self.X, Xprev=self.Xprev)
    else:
      return XData(self.X)

  ######################################################### Create Subset
  ######################################################### 
  def select_subset_by_mask(self, docMask=None, 
                                  atomMask=None,
                                  doTrackFullSize=True):
    ''' Creates new GroupXData object by selecting certain rows (observations)

        If doTrackFullSize is True, 
          ensure nDocTotal attribute is the same as the full dataset.
    '''
    if atomMask is not None:
      return self.toXData().select_subset_by_mask(atomMask)

    if len(docMask) < 1:
      raise ValueError('Cannot select empty subset')

    newXList = list()
    newXPrevList = list()
    newDocRange = np.zeros(len(docMask)+1)
    newPos = 1

    for d in xrange(len(docMask)):
      start = self.doc_range[docMask[d]]
      stop = self.doc_range[docMask[d]+1]
      newXList.append(self.X[start:stop])
      if hasattr(self, 'Xprev'):
        newXPrevList.append(self.Xprev[start:stop])
      newDocRange[newPos] = newDocRange[newPos-1] + stop - start
      newPos += 1

    newX = np.vstack(newXList)
    if hasattr(self, 'Xprev'):
      newXprev = np.vstack(newXPrevList)
    else:
      newXprev = None

    nDocTotal=None
    if doTrackFullSize:
      nDocTotal = self.nDocTotal

    return GroupXData(newX, newDocRange, Xprev=newXprev, nDocTotal=nDocTotal)

  ######################################################### Add Data
  ######################################################### 
  def add_data(self, XDataObj):
    ''' Updates (in-place) this object by adding new data
    '''
    if not self.dim == XDataObj.dim:
      raise ValueError("Dimensions must match!")
    self.nObs += XDataObj.nObs
    self.nDocTotal += XDataObj.nDocTotal
    self.nDoc += XDataObj.nDoc
    self.X = np.vstack([self.X, XDataObj.X])
    self.doc_range = np.hstack([self.doc_range, 
                                XDataObj.doc_range[1:] + self.doc_range[-1]])
    if hasattr(self, 'Xprev'):
      self.Xprev = np.vstack([self.Xprev, XDataObj.Xprev])
    self._check_dims()

  def get_random_sample(self, nDoc, randstate=np.random):
    nDoc = np.minimum(nDoc, self.nDoc)
    mask = randstate.permutation(self.nDoc)[:nDoc]
    Data = self.select_subset_by_mask(mask, doTrackFullSize=False)
    return Data

  #########################################################  I/O methods
  ######################################################### 
  def __str__(self):
    return self.X.__str__()
