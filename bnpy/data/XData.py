'''
XData.py

General template for all data objects.
'''
import numpy as np
import scipy.io
np.set_printoptions( precision=5)

class XData(object):
  
  @classmethod
  def read_from_mat( cls, matfilepath, nObsTotal=None, **kwargs):
    InDict = scipy.io.loadmat( matfilepath, **kwargs)
    if 'X' not in InDict:
      raise KeyError('Stored matfile needs to have data in field named X')
    return cls( InDict['X'], nObsTotal )
  
  def __init__(self, X, nObsTotal=None):
    X = np.asarray(X)
    if X.ndim < 2:
      X = X[np.newaxis,:]
    self.X = X.newbyteorder('=').copy()
    self.set_dependent_params(nObsTotal=nObsTotal)
  
  def __str__(self):
    return self.X.__str__()
    
  def summarize_num_observations(self):
    return '  num obs: %d' % (self.nObsTotal)
    
  def check_dims( self ):
    assert self.X.ndim == 2
    assert self.X.flags.c_contiguous
    assert self.X.flags.owndata
    assert self.X.flags.aligned
    assert self.X.flags.writeable
    
  def set_dependent_params( self, nObsTotal=None): 
    self.nObs = self.X.shape[0]
    self.dim = self.X.shape[1]
    if nObsTotal is None:
      self.nObsTotal = self.nObs
    else:
      self.nObsTotal = nObsTotal
  
  def subset_as_XData(self, mask):
    return XData( self.X[mask], nObsTotal=self.nObsTotal )
  
  def add_obs( self, X):
    if self.nObs != self.nObsTotal:
      raise ValueError('Adding observations violates consistency of larger dataset!')
    if type(X) == type( self):
      self.add_obs_from_mat(X.X)
    else:
      self.add_obs_from_mat(X)
      
  def add_obs_from_mat(self,X):
    ''' add numpy matrix X
    '''
    Xnew = np.asarray(X).copy().newbyteorder('=')
    self.X = np.vstack( [self.X, Xnew] )
    self.check_dims()
    self.set_dependent_params( )