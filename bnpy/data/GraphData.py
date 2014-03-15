'''
WordsData.py

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

  ######################################################### Create Toy Data

