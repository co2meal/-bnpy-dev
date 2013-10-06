'''
XData.py

Data object for holding a dense matrix X of real numbers,
where each row of X is a single observation.

This object guarantees underlying numpy array representation is best for math ops.
This means byteorder and C-contiguity are standardized.
'''
import numpy as np
import scipy.io
from .DataObj import DataObj

class GraphData(DataObj):
    
    @classmethod
    def read_from_mat(cls, matfilepath, nObsTotal=None, **kwargs):
        ''' Static Constructor for building an instance of XData from disk'''
        InDict = scipy.io.loadmat( matfilepath, **kwargs)
        if 'X' not in InDict:
            raise KeyError('Stored matfile needs to have data in field named X')
        return cls( InDict['X'], nObsTotal )
  
    def __init__(self, X, nObsTotal=None):
        ''' Constructor for building an instance of XData given an array
        Ensures array is 2-dimensional with proper byteorder, contiguity, and ownership'''
        X = np.asarray(X)
        self.X = X
        self.set_dependent_params(nObsTotal=nObsTotal)
        self.check_dims()
        
    #########################################################  internal methods #########################################################   
    def set_dependent_params( self, nObsTotal=None): 
        self.nObs = self.X.shape[0]
        self.dim = self.X.shape[1]
        self.N = int(np.max([self.X[:,0].max(),self.X[:,1].max()])) + 1
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
    