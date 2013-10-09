'''
XData.py

Data object for holding a dense matrix X of real numbers,
where each row of X is a single observation.

This object guarantees underlying numpy array representation is best for math ops.
This means byteorder and C-contiguity are standardized.
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
    return cls( InDict['X'], nObsTotal )
  
  def __init__(self, X, nObsTotal=None, TrueZ=None):
    ''' Constructor for building an instance of XData given an array
        Ensures array is 2-dimensional with proper byteorder, contiguity, and ownership
    '''
    X = np.asarray(X)
    if X.ndim < 2:
      X = X[np.newaxis,:]
    self.X = X.newbyteorder('=').copy()
    self.set_dependent_params(nObsTotal=nObsTotal)
    self.check_dims()
    if TrueZ is not None:
      self.addTrueLabels(TrueZ)
    
  def addTrueLabels(self, TrueZ):
    ''' Adds a "true" discrete segmentation of this data,
        so that each of the nObs items have a single label
    '''
    self.TrueLabels = TrueZ
    assert self.nObs == TrueZ.size

  #########################################################  internal methods
  #########################################################   
  def set_dependent_params( self, nObsTotal=None): 
    self.nObs = self.X.shape[0]
    self.dim = self.X.shape[1]
    if nObsTotal is None:
      self.nObsTotal = self.nObs
    else:
      self.nObsTotal = nObsTotal
    
  def check_dims( self ):
    assert self.X.ndim == 2
    assert self.X.flags.c_contiguous
    assert self.X.flags.owndata
    assert self.X.flags.aligned
    assert self.X.flags.writeable
    
  #########################################################  DataObj operations
  ######################################################### 
  def select_subset_by_mask(self, mask):
    ''' Creates new XData object by selecting certain rows (observations)
        Ensures the nObsTotal attribute is the same.
    '''
    return XData(self.X[mask], nObsTotal=self.nObsTotal)

  #########################################################  I/O methods
  ######################################################### 
  def __str__(self):
    np.set_printoptions(precision=5)
    return self.X.__str__()
    
  def summarize_num_observations(self):
    return '  num obs: %d' % (self.nObsTotal)
