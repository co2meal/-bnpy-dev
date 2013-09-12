'''
MinibatchIteratorFromDisk.py

Extension of MinibatchIterator 
   reads in data that has been pre-split into batches on disk.
   
Usage:
  construct by providing a list of valid filepaths to .mat files (see XData for format)
  then call has_next_batch()
            get_next_batch()
  
  Traversal order of the files is randomized every lap through the full dataset
  Set the "dataseed" parameter to get repeatable orders.
'''

import numpy as np
import scipy.io

from MinibatchIterator import MinibatchIterator
from XData import XData

class MinibatchIteratorFromDisk( MinibatchIterator):
  def __init__( self, fList, nObsTotal, nBatch=None, nRep=10, dataseed=42):
    if nBatch is None or nBatch > len(fList):
      nBatch = len(fList)
    self.fList = [fList[k] for k in range(nBatch)]
    self.configBatchTraversalOrder( nBatch, nRep, dataseed)
    self.nObsTotal = nObsTotal
    
  def get_batch_as_XData( self, bID):
    return XData.read_from_mat( self.fList[bID], nObsTotal=self.nObsTotal )
