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

def _toStd2DArray(X):
  X = np.asarray_chkfinite(X, dtype=np.float64, order='C')
  if X.ndim < 2:
    X = X[np.newaxis,:]
  assert X.ndim == 2
  return X

class GroupXData(XData):
  
  @classmethod
  def read_from_mat(cls, matfilepath, nDocTotal=None, **kwargs):
    ''' Static Constructor for building an instance of GroupXData from disk
    '''
    import scipy.io
    InDict = scipy.io.loadmat( matfilepath, **kwargs)
    if 'X' not in InDict:
      raise KeyError('Stored matfile needs to have data in field named X')
    return cls(InDict['X'], InDict['doc_range'],
                            nDocTotal=nDocTotal)
  
  def __init__(self, X, doc_range, nDocTotal=None, 
                        Xprev=None, TrueZ=None, TrueParams=None, summary=None):
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
      self.TrueParams = TrueParams
    if TrueZ is not None:
      if not hasattr(self, 'TrueParams'):
        self.TrueParams = dict()
      self.TrueParams['Z'] = TrueZ

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
    s = '  %d observations, each of dimension %d' % (self.nObs, self.dim)
    s += '\n  %d groups' % (self.nDoc) 
    return s

  ######################################################### Create Subset
  ######################################################### 
  def select_subset_by_mask(self, docMask, doTrackFullSize=True):
    ''' Creates new GroupXData object by selecting certain rows (observations)
        If doTrackFullSize is True, 
          ensure nDocTotal attribute is the same as the full dataset.
    '''
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

    if doTrackFullSize:
      nDocTotal = self.nDocTotal
    else:
      nDocTotal = len(newDocRange)
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
    self.X = np.vstack([self.X, XDataObj.X])
    doc_range = np.zeros(self.nDoc + XDataObj.nDoc + 1)
    doc_range[:self.nDoc+1] = self.doc_range
    doc_range[self.nDoc+1:] = XDataObj.doc_range + self.doc_range[-1]
    self.doc_range = doc_range
    self.nDoc += XDataObj.nDoc
    if hasattr(self, 'Xprev'):
      self.Xprev = np.vstack([self.Xprev, XDataObj.Xprev])


  def get_random_sample(self, nDoc, randstate=np.random):
    nDoc = np.minimum(nDoc, self.nDoc)
    mask = randstate.permutation(self.nDoc)[:nDoc]
    Data = self.select_subset_by_mask(mask, doTrackFullSize=False)
    return Data

  #########################################################  I/O methods
  ######################################################### 
  def __str__(self):
    return self.X.__str__()
