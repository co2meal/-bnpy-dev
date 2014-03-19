'''
GraphData.py

Data object that represents word counts across a collection of documents.

Terminology
-------
* Vocab : The finite collection of possible words.  
    {apple, berry, cardamom, fruit, pear, walnut}
  We assume this set has a fixed ordering, so each word is associated 
  with a particular integer in the set 0, 1, ... vocab_size-1
     0: apple        3: fruit
     1: berry        4: pear
     2: cardamom     5: walnut
* Document : a collection of words, observed together from the same source
  For example: 
      "apple, berry, berry, pear, pear, pear, walnut"

* nDoc : number of documents in the current, in-memory dataset
* nDocTotal : total number of docs, in entire dataset (for online applications)
'''

from .AdmixMinibatchIterator import AdmixMinibatchIterator
from .DataObj import DataObj
import numpy as np
import scipy.sparse
from ..util import RandUtil

class GraphData(DataObj):

  ######################################################### Constructor
  #########################################################
  def __init__(self, edge_id=None, edge_weight=None, nNodeTotal=None, TrueParams=None, **kwargs):
    ''' Constructor for WordsData object

        Args
        -------
        edge_id : the source and receiver node ids
        edge_value : this might be a vector of all ones if binary graph
        nNodeTotal : total number of nodes
        TrueParams : None [default], or dict of attributes
    '''
    self.edge_id = np.asarray(np.squeeze(edge_id), dtype=np.uint32)
    self.edge_weight = np.asarray(np.squeeze(edge_weight), dtype=np.float64)
    self.nNodeTotal = int(nNodeTotal)
    self.nEdgeTotal = len(edge_id)
  
    # Save "true" parameters that generated toy-data, if provided
    if TrueParams is not None:
      self.TrueParams = TrueParams

  ######################################################### Create Toy Data
  def to_sparse_matrix(self):
    ''' Make sparse matrix counting vocab usage across all words in dataset

        Returns
        --------
        C : sparse (CSC-format) matrix, of shape nObs-x-vocab_size, where
             C[n,v] = word_count[n] iff word_id[n] = v
                      0 otherwise
             That is, each word token n is represented by one entire row
                      with only one non-zero entry: at column word_id[n]

    '''
    if hasattr(self, "__sparseMat__"):
      return self.__sparseMat__
    edge_values = np.ones(self.nEdgeTotal)
    self.__sparseMat__ = scipy.sparse.csc_matrix(
                        (edge_values, ( np.int64(self.edge_id[:,0]), np.int64(self.edge_id[:,1]) ) ),
                        shape=(self.nNodeTotal, self.nNodeTotal))
    return self.__sparseMat__

  ######################################################### Create from MAT
  #########################################################  (class method)
  @classmethod
  def read_from_mat(cls, matfilepath, **kwargs):
    ''' Creates an instance of WordsData from Matlab matfile
    '''
    import scipy.io
    InDict = scipy.io.loadmat(matfilepath, **kwargs)
    return cls(**InDict)

  ######################################################### Create from DB
  #########################################################  (class method)
  @classmethod
  def read_from_db(cls, dbpath, sqlquery, vocab_size=None, nDocTotal=None):
    pass