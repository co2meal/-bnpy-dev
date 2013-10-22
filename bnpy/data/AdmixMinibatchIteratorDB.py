'''
AdmixMinibatchIteratorDB.py

Generic object for iterating over a single bnpy Data set
by pulling a minibatch from an sqlite type database. 
Support for more generic databases will come in the next version.

Usage
--------
Construct by providing the underlying full-dataset
>> MB = AdmixMinibatchIterator(Data, nBatch=10, nObsBatch=100)
Then call
     has_next_batch()  to test if more data is available
     get_next_batch()  pull dataset from database in this function
     
Batches are defined via a random partition of all data items
   e.g. for 100 items split into 20 batches
      batch 1 : items 5, 22, 44, 30, 92
      batch 2 : items 93, 33, 46, 12, 78,
      etc.
      
Supports multiple laps through the data.  Specify # of laps with parameter nLap.
Traversal order of the batch is randomized every lap through the full dataset
Set the "dataorderseed" parameter to get repeatable orders.

Attributes
-------
nBatch : number of batches to divide full dataset into
nObsBatch : number of observations in each batch (on average)
nObsTotal : number of observations in entire full dataset (in terms of documents)
nLap : number of times to pass thru all batches in dataset during iteration

batchID : exact integer ID of the current batch. range=[0, nBatch-1]
curLapPos : integer count of current position in batch order. incremented 1 at a time.
lapID : integer ID of the current lap
'''
from IPython import embed
import numpy as np
import sqlite3
MAXSEED = 1000000
  
class AdmixMinibatchIteratorDB(object):
  def __init__(self, Data, dbpath=None, nDocTotal=None, nBatch=None, nObsBatch=None, nLap=20, dataorderseed=42, allocModelName=None):
    ''' Constructor for creating an iterator over the batches of data
    '''
    self.Data = Data
    self.nBatch = nBatch
    self.nLap = nLap
    self.dbpath = dbpath
    # specify self.nObsTotal to be the total number of documents in the admixture case
    # used only really to have stochastic online not break
    self.nDocTotal = nDocTotal
    self.nObsTotal = nDocTotal #nDoc is the total number of documents in the given mini-batch
    self.vocab_size = Data.vocab_size
    
    # number of observations in batch
    if nObsBatch is None:
        self.nObsBatch = nDocTotal/nBatch
    else:
        self.nObsBatch = nObsBatch
    
    # Config order in which batches are traversed
    self.curLapPos = -1
    self.lapID  = 0
    self.dataorderseed = int(int(dataorderseed) % MAXSEED)
    # Make list with entry for every distinct batch
    #   where each entry is itself a list of obsIDs in the full dataset
    self.obsIDByBatch = self.configObsIDsForEachBatch()
          
  #########################################################  accessor methods
  #########################################################   
  def has_next_batch( self ):
    if self.lapID >= self.nLap:
      return False
    if self.lapID == self.nLap - 1:
      if self.curLapPos == self.nBatch - 1:
        return False
    return True
 
  def get_next_batch( self ):
    ''' Returns DataObj of the next batch
    Since this is a DB thing, we should connect to the DB and return appropriate documents
    '''
    if not self.has_next_batch():
      raise StopIteration()
      
    self.curLapPos += 1
    if self.curLapPos >= self.nBatch:
      self.curLapPos = 0
      self.lapID += 1
    
    # Create the DataObj for the current batch
    self.batchOrderCurLap = self.get_rand_order_for_batchIDs_current_lap()
    self.batchID = self.batchOrderCurLap[self.curLapPos]
    obsIDsCurBatch = self.obsIDByBatch[self.batchID]
    
    query = 'select * from data where rowid in (' + ','.join(map(str, obsIDsCurBatch)) + ')'
    bData = self.Data.read_from_db( self.dbpath, query, nDoc=len(obsIDsCurBatch), nDocTotal = self.nDocTotal, vocab_size = self.vocab_size ) 
    #bData = self.Data.select_subset_by_mask(obsIDsCurBatch)    
    return bData 
    
  def getObsIDsForCurrentBatch(self):
    return self.obsIDByBatch[self.batchOrderCurLap[self.curLapPos]]

  def get_text_summary(self):
    ''' Returns string with human-readable description of this dataset 
        e.g. source, author/creator, etc.
    '''
    if hasattr(self, 'summary'):
      return self.summary
    return 'Generic %s Dataset' % (self.__class__.__name__)

  #########################################################  internal methods
  #########################################################           
  def configObsIDsForEachBatch(self):
    ''' Assign each observation in dataset to a batch by random permutation

        Returns
        --------
        obsIDByBatch : list of length self.nBatch,
                       where obsIDByBatch[bID] : list of all obsIDs in batch bID 
    '''
    PRNG = np.random.RandomState(self.dataorderseed)
    #Note that we're using nDocTotal to permute document ids
    obsIDs = PRNG.permutation(self.nDocTotal).tolist()
    # need to add 1 to list of document indices since sql indexes rows starting with 1
    obsIDs = [x+1 for x in obsIDs]
    
    obsIDByBatch = dict()
    for batchID in range(self.nBatch):
      obsIDByBatch[batchID] = obsIDs[:self.nObsBatch]
      del obsIDs[:self.nObsBatch]
    return obsIDByBatch

  def get_rand_order_for_batchIDs_current_lap(self):
    ''' Returns array of batchIDs, permuted in random order
        Order changes each time we traverse all items (each lap)
    '''
    curseed = self.dataorderseed + self.lapID
    PRNG = np.random.RandomState(curseed)
    return PRNG.permutation( self.nBatch )

  #########################################################  I/O methods
  #########################################################    
  def summarize_num_observations(self):
    s = '  num batch %d, num obs per batch %d\n' % (self.nBatch, self.nObsBatch)
    s += '  num documents (total across all batches): %d' % (self.nDocTotal)
    return s