'''
BagOfWordsMinibatchIterator.py

Object that manages iterating over many minibatches of a large document collection.

Usage
--------
Construct by providing the underlying full-dataset
>> I = BagOfWordsMinibatchIterator(Data, nBatch=10, nLap=3)

To determine if more data remains, call *has_next_batch*
>> I.has_next_batch()

To access the next batch, call the *get_next_batch* method.
This returns a BagOfWordsData object
>> DataChunk = I.get_next_batch()

Batches are defined in advance via a random partition of all documents
For example, given 10 documents, a possible set of 3 batches is
   batch 1 : docs 1, 3, 9, 4,
   batch 2 : docs 5, 7, 0
   batch 3 : docs 8, 2, 6

Each lap (pass through the data) iterates through these same fixed batches.

The traversal order of the batch is randomized at each lap.
For example, during the first 3 laps, we may see the following orders
   lap 0 : batches 0, 2, 1
   lap 1 : batches 2, 1, 0
   lap 2 : batches 0, 1, 2
Set the "dataorderseed" parameter to get repeatable orders.

Attributes
-------
nBatch : number of batches to divide full dataset into
nLap : number of times to pass thru all batches in dataset 
       before raising a StopIteration exception

batchID : integer ID of the most recent batch returned by get_next_batch() 
          Range is [0, nBatch-1]
curLapPos : integer count of current position in batch order. 
            Range is [0, nBatch-1]
            Always incremented by 1 after every call to get_next_batch()
lapID : integer ID of the current lap
        Range is [0, nLap-1]. 
'''

import numpy as np
MAXSEED = 1000000
  
class BagOfWordsMinibatchIterator(object):

  def __init__(self, Data, nBatch=10, nLap=10, 
                           dataorderseed=42, startLap=0, **kwargs):
    ''' Constructor for iterator over subsets of document collection.

        Each batch/subset is represented as a BagOfWordsData object

        Args
        --------
        Data : WordsData object
        nBatch : integer number of batches to divide dataset into
        nLap : integer number of laps to complete
        dataorderseed : int seed for division of data into fixed set of batches
                                     and random traversal of elements in this set
    '''
    assert Data.nDocTotal == Data.nDoc
    self.Data = Data
    self.nBatch = int(nBatch)
    self.nLap = nLap + int(startLap)
    
    # Config order in which batches are traversed
    self.curLapPos = -1
    self.lapID  = int(startLap)
    self.dataorderseed = int(int(dataorderseed) % MAXSEED)

    ## Decide how many docs will be in each batch
    ## nDocPerBatch : 1D array, size nBatch
    ## nDocPerBatch[b] gives total number of docs in batch b
    nDocPerBatch = Data.nDoc // nBatch * np.ones(nBatch, dtype=np.int32)
    nRem = Data.nDoc - nDocPerBatch.sum()
    nDocPerBatch[:nRem] += 1

    ## Randomly divide the documents across all nBatch batches
    PRNG = np.random.RandomState(self.dataorderseed)
    randDocIDs = PRNG.permutation(self.Data.nDoc).tolist()
    self.DataPerBatch = list()
    self.docIDsPerBatch = list()
    for b in xrange(nBatch):
      curBatchDocMask = randDocIDs[:nDocPerBatch[b]]
      Dchunk = Data.select_subset_by_mask(curBatchDocMask)
      self.DataPerBatch.append(Dchunk)
      self.docIDsPerBatch.append(curBatchDocMask)
      # Remove all docs assigned to this batch from consideration,
      # so future batches get assigned new docs 
      del randDocIDs[:nDocPerBatch[b]]

    ## Decide which order the batches will be traversed in the first lap
    self.batchOrderCurLap = self.get_rand_order_for_batchIDs_current_lap()

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
    ''' Get the WordsData object for the next batch

        Raises
        --------
        StopIteration if we have completed all specified laps

        Updates (in-place)
        --------
        batchID reflects index of batch returned.
  `     lapID reflects how many laps have been *completed*. in [0, 1, ... nLap-1]
        curLapPos is units of progress through current lap. in [0, ... nBatch-1]

        Returns
        --------
        Data : WordsData object for the current batch
    '''
    if not self.has_next_batch():
      raise StopIteration()
      
    self.curLapPos += 1
    if self.curLapPos >= self.nBatch:
      # Starting a new lap!
      self.curLapPos = 0
      self.lapID += 1
      self.batchOrderCurLap = self.get_rand_order_for_batchIDs_current_lap()

    # Create the DataObj for the current batch
    self.batchID = self.batchOrderCurLap[self.curLapPos]
    return self.DataPerBatch[self.batchID]


  def get_rand_order_for_batchIDs_current_lap(self):
    ''' Returns array of batchIDs, permuted in random order
        Order changes each lap (each full traversal of all items)
    '''
    curseed = int(self.dataorderseed + self.lapID)
    PRNG = np.random.RandomState(curseed)
    return PRNG.permutation(self.nBatch)

  #########################################################  I/O methods
  ######################################################### 
  def get_text_summary(self):
    ''' Returns string with human-readable description of this dataset 
        e.g. source, author/creator, etc.
    '''
    if hasattr(self, 'summary'):
      return self.summary
    return 'Minibatch Iterator: %d batches' % (self.nBatch)
     
  def summarize_num_observations(self):
    s = '  num batch %d, num obs per batch %d\n' % (self.nBatch, self.nObsBatch)
    s += '  num obs (total across all batches): %d' % (self.Data.nObsTotal)
    return s