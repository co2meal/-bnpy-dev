'''
GraphXData.py

Data object for holding dense real observations that sit on the edges of a graph

Attributes
-------
X : 2D array, size nObs x D
sourceID : 1D array, size nObs, holding the ID of the source node of edge i
destID : 1D array, size nObs, holding the ID of the destination node of edge i
nObs : int total number of unique observations in the current, in-memory batch
TrueParams : (optional) dict
summary : (optional) string providing human-readable description of this data

Example
--------
'''

import numpy as np
import scipy.io

from .XData import XData

class GraphXData(XData):

  @classmethod
  def LoadFromFile(cls, filepath, nObsTotal=None, **kwargs):
    ''' Static constructor for loading data from disk into XData instance
    '''
    if filepath.endswith('.mat'):
      return cls.read_from_mat(filepath, nObsTotal, **kwargs)
    raise NotImplemented('Only .mat file supported')

  @classmethod
  def read_from_mat(cls, matfilepath, nObsTotal=None, **kwargs):
    ''' Static constructor for loading data from .mat file into GraphXData
        instance.
        If no sourceID/destID field is given, it's assumed that 'X' is of size
          N^2, and gives edges in order 0->1, ..., 0->N-1, 1->0, etc.
    '''
    InDict = scipy.io.loadmat( matfilepath, **kwargs)
    if 'X' not in InDict:
      raise KeyError('Stored matfile needs to have data in field named X')

    if ('sourceID' not in InDict) or ('destID' not in InDict):
      N = np.sqrt(InDict['X'].shape[0])
      #This check *might* numerically fail for big N
      if int(N) != N:
        raise ValueError('Either specify sourceID and destID or give a full adjacency matrix')

      adjList = np.tile(np.arange(N), (N, 1))
      InDict['sourceID'] = adjList.T.ravel()
      InDict['destID'] = adjList.ravel()
      
    return cls(**InDict)


  def __init__(self, X, sourceID, destID, nNodesTotal=None, nObsTotal=None,
               TrueZ=None, TrueParams=None, summary=None, **kwargs):
    sourceID = np.asarray(sourceID)
    destID = np.asarray(destID)
    self.sourceID = np.uint32(sourceID.newbyteorder('=').copy())
    self.destID = np.uint32(destID.newbyteorder('=').copy())
    
    super(GraphXData, self).__init__(X, nObsTotal=nObsTotal, TrueZ=TrueZ,
                                     Xprev=None, TrueParams=TrueParams,
                                     summary=summary)
    #self._check_dims()
    #self._set_dependent_params(nNodesTotal=nNodesTotal,
    #                           nObsTotal=nObsTotal)


  def _check_dims(self):
    assert self.sourceID.ndim == 1
    assert self.sourceID.flags.c_contiguous
    assert self.sourceID.flags.owndata
    assert self.sourceID.flags.aligned
    assert self.sourceID.flags.writeable

    assert self.destID.ndim == 1
    assert self.destID.flags.c_contiguous
    assert self.destID.flags.owndata
    assert self.destID.flags.aligned
    assert self.destID.flags.writeable

    assert self.destID.shape[0] == self.sourceID.shape[0]
    assert self.sourceID.shape[0] == self.X.shape[0]

    super(GraphXData,self)._check_dims()


  def _set_dependent_params(self, nNodesTotal=None, nObsTotal=None):
    self.nNodes = np.max([np.max(self.sourceID), np.max(self.destID)]) + 1
    if nNodesTotal is None:
      self.nNodesTotal = self.nNodes
    else:
      self.nNodesTotal = nNodesTotal
    super(GraphXData,self)._set_dependent_params(nObsTotal=nObsTotal)

  def get_stats_summary(self):
    ''' Returns human-readable summary of this dataset's basic properties
    '''
    s = 'Graph with N = %d nodes and %d edges\n' % (self.nNodes, self.nObs)
    s+= ' dimension: %d' % (self.get_dim())
    return s
    
  ######################################################### Create Subset
  ######################################################### 
  def select_subset_by_mask(self, mask, doTrackFullSize=True):
    ''' Creates new XData object by selecting certain rows (observations)
        If doTrackFullSize is True, 
          ensure nObsTotal and nNodesTotal attributes are the same as the full
          dataset.
    '''
    if doTrackFullSize:
      return GraphXData(self.X[mask], self.sourceID[mask], self.destID[mask],
                        nNodesTotal=self.nNodesTotal, nObsTotal=self.nObsTotal)
    return GraphXData(self.X[mask], self.source[mask], self.destID[mask])
                       
  #def get_random_sample(self, nObs, randstate=np.random):
  #  nObs = np.minimum(nObs, self.nObs)
  #  mask = randstate.permutation(self.nObs)[:nObs]
  #  Data = self.select_subset_by_mask(mask, doTrackFullSize=False)
  #  return Data
    
  ######################################################### Add Data
  #########################################################
  def add_data(self, GraphXDataObj):
    ''' Updates (in-place) this object by adding new data
    '''
    super(GraphXData, self).add_data(GraphXDataObj)

    self.nNodes += GraphXDataObj.nNodes
    self.nNodesTotal += GraphXDataObj.nNodesTotal
    self.sourceID = np.append(self.sourceID, GraphXDataObj.sourceID)
    self.destID = np.append(self.destID, GraphXDataObj.destID)
    

  #########################################################  I/O methods
  #########################################################
  def __str__(self):
    return super(GraphXData,self).__str__() + '\n sourceID: ' + \
      self.sourceID.__str__() + '\n destID: ' + self.destID.__str__()
      

    