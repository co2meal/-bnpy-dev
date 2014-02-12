'''
DiverseData.py

Data object for holding different data formats. For instance it can 
hold both dense matrix X of real numbers,
where each row of X is a single observation and word counts.

TO DO: write better doc.

This object guarantees underlying numpy array representation is best for math ops.
This means byteorder and C-contiguity are standardized.
'''
import numpy as np
from .DataObj import DataObj
from .XData import XData
from .WordsData import WordsData
from .MinibatchIterator import MinibatchIterator

SupportedDataTypes = {'XData':XData,'WordsData':WordsData}

class DiverseData(DataObj):
  @classmethod
  def read_from_mat(cls, matfilepath, nObsTotal=None, **kwargs):
    ''' Static Constructor for building an instance of XData from disk
    '''
   #TO DO
    raise NotImplementedError
  
  def __init__(self, Xlist, XtypeList, nObsTotal=None, TrueZ=None):
    ''' Constructor for building an instance of DiverseData given a list of 
        data arrays and types. Ensures array is 2-dimensional with proper byteorder, 
        contiguity, and ownership.
    '''
    assert type(Xlist) == list, 'DiverseData requires the data to be encapsulated in a list'
    self.DataList = []
    for X,Xtype in zip(Xlist,XtypeList):
        assert Xtype in SupportedDataTypes
        self.DataList.append(SupportedDataTypes[Xtype](X))
        self.set_dependent_params(nObsTotal=nObsTotal)
        self.check_dims()
        if TrueZ is not None:
            self.addTrueLabels(TrueZ)
    
  def addTrueLabels(self, TrueZ):
    ''' Adds a "true" discrete segmentation of this data,
        so that each of the nObs items have a single label
    '''
    assert self.nObs == TrueZ.size
    self.TrueLabels = TrueZ
  
  def to_minibatch_iterator(self, **kwargs):
    return MinibatchIterator(self, **kwargs)

  #########################################################  internal methods
  #########################################################   
  def set_dependent_params( self, nObsTotal=None): 
    # mainly for backward compatibility TODO -- maybe remove these fields?  
    self.nObs = self.DataList[0].nObs
    #self.dim = [X.dim for X in self.DataList]
    if nObsTotal is None:
      self.nObsTotal = self.nObs
    else:
      self.nObsTotal = nObsTotal
    
  def check_dims( self ):
      nObs = self.nObs
      for data in self.DataList:
          assert data.nObs==nObs, 'The number of observations must be constant across the data list provided to DiverseData' 
   
  #########################################################  DataObj operations
  ######################################################### 
  def select_subset_by_mask(self, mask, doTrackFullSize=True):
    ''' Creates new XData object by selecting certain rows (observations)
        If doTrackFullSize is True, 
          ensure nObsTotal attribute is the same as the full dataset.
    '''
    # if doTrackFullSize:
    #    return XData(self.X[mask], nObsTotal=self.nObsTotal)
    #return XData(self.X[mask])
    raise NotImplementedError
    

  def add_data(self, XDataObj):
    ''' Updates (in-place) this object by adding new data
    '''
    raise NotImplementedError
    # TODO: if not self.dim == XDataObj.dim:
    #  raise ValueError("Dimensions must match!")
    #self.nObs += XDataObj.nObs
    #self.nObsTotal += XDataObj.nObsTotal
    #self.X = np.vstack([self.X, XDataObj.X])

  #########################################################  I/O methods
  ######################################################### 
  def __str__(self):
    np.set_printoptions(precision=5)
    return [X.__str__() for X in self.X]

  def summarize_num_observations(self):
    return '  num obs: %d' % (self.nObsTotal)
