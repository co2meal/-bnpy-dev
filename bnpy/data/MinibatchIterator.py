'''
MinibatchIterator.py

Generic object for iterating over a single bnpy Data set
  by considering one subset minibatch (often just called a batch) at a time

Usage
  construct it with a Data object
  then call
     has_next_batch()  to test if more data is available
     get_next_batch()  to get the next batch (as a Data object)
     
Batches are defined via a random partition of all data items
   e.g. for 100 items split into 20 batches
      batch 1 : items 5, 22, 44, 30, 92
      batch 2 : items 93, 33, 46, 12, 78,
      etc.
      
Supports multiple laps through the data.  Specify # of laps with parameter nRep.
  Traversal order of the batch is randomized every lap through the full dataset
  Set the "dataseed" parameter to get repeatable orders.
'''
import numpy as np
MAXSEED = 1000000
  
class MinibatchIterator( object ):
  def __init__( self, Data, nBatch=10, batch_size=None, nRep=10, dataseed=42):
    self.Data = Data
    if batch_size is None:
      self.batch_size = Data.nObs/nBatch
    else:
      self.batch_size = batch_size
    assert abs( self.batch_size*nBatch - Data.nObs ) < nBatch
    self.configBatchTraversalOrder( nBatch, nRep, dataseed)
    # Make list with entry for every distinct batch
    #   where each entry is itself a list of obsIDs in the full dataset
    self.obsIDByBatch = self.get_obs_id_list_per_batch()
          
  def configBatchTraversalOrder(self, nBatch, nRep, dataseed):
    self.nBatch = nBatch
    self.nRep = nRep
    self.passID = 0
    self.repID  = 0
    self.dataseed = 10* int( int(dataseed) % MAXSEED)
    # Prepare current order for traversing batches
    self.CurBatchIDs = self.get_random_order_for_batchIDs()
    
  def get_obs_id_list_per_batch(self):
    PRNG = np.random.RandomState( self.dataseed )
    obsIDs = PRNG.permutation( self.Data.nObs ).tolist()
    obsIDByBatch = dict()
    for batchID in range( self.nBatch):
      obsIDByBatch[batchID] = obsIDs[:self.batch_size]
      del obsIDs[:self.batch_size]
    return obsIDByBatch  

  def get_random_order_for_batchIDs(self):
    curseed = self.dataseed + self.repID
    PRNG = np.random.RandomState( curseed )
    return PRNG.permutation( self.nBatch )

  def get_batch_as_XData( self, bID):
    return self.Data.subset_as_XData( self.obsIDByBatch[bID] )
    
  def has_next_batch( self ):
    if self.repID >= self.nRep:
      return False
    return True
 
  def get_next_batch( self ):
    if not self.has_next_batch():
      raise StopIteration()
    if self.passID == 0:
      print '---------------------------------  batchIDs=', self.CurBatchIDs[:4], '...'

    bID = self.CurBatchIDs[ self.passID]
    bData = self.get_batch_as_XData( bID )
    
    # Increment stuff for next time
    self.passID += 1
    if self.passID >= self.nBatch:
      self.passID = 0
      self.repID += 1
      self.CurBatchIDs = self.get_random_order_for_batchIDs()
    return bData 
