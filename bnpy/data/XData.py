'''
XData.py

Data object for holding a dense matrix X of real 64-bit floating point numbers,
Each row of X represents a single observation.

Attributes
--------
X : 2D array, size N x D
TrueParams : (optional) dict
summary : (optional) string providing human-readable description of this data

Example
--------
>> import numpy as np
>> from bnpy.data import XData
>> X = np.random.randn(1000, 3) # Create 1000x3 matrix
>> myData = XData(X)
>> print myData.nObs
1000
>> print myData.D
3
>> print myData.X.shape
(1000,3)
'''

import numpy as np
from .DataObj import DataObj


class XData(DataObj):
  
  @classmethod
  def read_from_mat(cls, matfilepath, nObsTotal=None, **kwargs):
    ''' Static Constructor for building an instance of XData from disk
    '''
    import scipy.io
    InDict = scipy.io.loadmat( matfilepath, **kwargs)
    if 'X' not in InDict:
      raise KeyError('Stored matfile needs to have data in field named X')
    return cls(InDict['X'], nObsTotal)
  
  def __init__(self, X, nObsTotal=None, TrueZ=None, Xprev=None,
                        TrueParams=None, summary=None):
    ''' Create an instance of XData for provided array data X

        Reallocation of memory may occur, to ensure that X is a 2D numpy array
        with standardized data-type, byteorder, contiguity, and ownership.
    '''
    X = np.asarray(X)
    if X.ndim < 2:
      X = X[np.newaxis,:]
    self.X = np.float64(X.newbyteorder('=').copy())
    ## Verify attributes are consistent
    self._set_dependent_params(nObsTotal=nObsTotal)
    self._check_dims()
    ## Add optional true parameters / true hard labels
    if TrueParams is not None:
      self.TrueParams = TrueParams
    if TrueZ is not None:
      if not hasattr(self, 'TrueParams'):
        self.TrueParams = dict()
      self.TrueParams['Z'] = TrueZ

    if summary is not None:
      self.summary = summary
    if Xprev is not None:
      self.Xprev = np.float64(Xprev.newbyteorder('=').copy())

  def _set_dependent_params( self, nObsTotal=None): 
    self.nObs = self.X.shape[0]
    self.dim = self.X.shape[1]
    if nObsTotal is None:
      self.nObsTotal = self.nObs
    else:
      self.nObsTotal = nObsTotal

  def _check_dims( self ):
    assert self.X.ndim == 2
    assert self.X.flags.c_contiguous
    assert self.X.flags.owndata
    assert self.X.flags.aligned
    assert self.X.flags.writeable
    
  def get_size(self):
    return self.nObs

  def get_total_size(self):
    return self.nObs

  def get_text_summary(self):
    ''' Returns human-readable description of this dataset
    '''
    if hasattr(self, 'summary'):
      s = self.summary
    else:
      s = 'X Data'
    return s

  def get_stats_summary(self):
    ''' Returns human-readable summary of this dataset's basic properties
    '''
    s = '  %d observations, each of dimension %d' % (self.nObs, self.dim)
    return s

  ######################################################### Create Subset
  ######################################################### 
  def select_subset_by_mask(self, mask, doTrackFullSize=True):
    ''' Creates new XData object by selecting certain rows (observations)
        If doTrackFullSize is True, 
          ensure nObsTotal attribute is the same as the full dataset.
    '''
    if hasattr(self, 'Xprev'):
      newXprev = self.Xprev[mask]
    else:
      newXprev = None
    if doTrackFullSize:
        return XData(self.X[mask], nObsTotal=self.nObsTotal, Xprev=newXprev)
    return XData(self.X[mask], Xprev=newXprev)

  ######################################################### Add Data
  ######################################################### 
  def add_data(self, XDataObj):
    ''' Updates (in-place) this object by adding new data
    '''
    if not self.dim == XDataObj.dim:
      raise ValueError("Dimensions must match!")
    self.nObs += XDataObj.nObs
    self.nObsTotal += XDataObj.nObsTotal
    self.X = np.vstack([self.X, XDataObj.X])
    if hasattr(self, 'Xprev'):
      self.Xprev = np.vstack([self.Xprev, XDataObj.Xprev])

  def get_random_sample(self, nObs, randstate=np.random):
    nObs = np.minimum(nObs, self.nObs)
    mask = randstate.permutation(self.nObs)[:nObs]
    Data = self.select_subset_by_mask(mask, doTrackFullSize=False)
    return Data

  def generate_batch_ids(self, nBatch, dataorderseed, sizeBatch):
    PRNG = np.random.RandomState(dataorderseed)
    obsIDs = PRNG.permutation(self.nObsTotal).tolist()
    obsIDByBatch = dict()
    for batchID in range(nBatch-1):
      obsIDByBatch[batchID] = obsIDs[:sizeBatch]
      del obsIDs[:sizeBatch]
    # Last batch gets leftovers, may be bigger
    obsIDByBatch[nBatch-1] = obsIDs 
    return obsIDByBatch


  #########################################################  I/O methods
  ######################################################### 
  def __str__(self):
    return self.X.__str__()
