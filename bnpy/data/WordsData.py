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