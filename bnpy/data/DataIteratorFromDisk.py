'''
DataIteratorFromDisk.py

Object that manages iterating over minibatches stored to disk. 

Usage
--------
Construct by providing the underlying full-dataset
>> I = DataIterator('/path/to/folder/', nBatch=10, nLap=3)

To determine if more data remains, call *has_next_batch*
>> I.has_next_batch()

To access the next batch, call the *get_next_batch* method.
This returns a BagOfWordsData object
>> DataChunk = I.get_next_batch()

Batches are defined in advance based on what is saved to disk.
Each file in the provided directory defines a single batch.

Each lap (pass through the data) iterates through these same fixed batches.

The traversal order of the batch is randomized at each lap.
For example, during the first 3 laps, we may see the following orders
   lap 0 : batches 0, 2, 1
   lap 1 : batches 2, 1, 0
   lap 2 : batches 0, 1, 2
Set the "dataorderseed" parameter to get repeatable orders.

Attributes
-------
nBatch : number of batches to retrieve from disk
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
import os
import sys
import glob
import numpy as np
import scipy.io

MAXSEED = 1000000
  
from bnpy.data import WordsData, XData, GroupXData

class DataIteratorFromDisk(object):

  def __init__(self, datapath, nBatch=0, nLap=1, 
                     dataorderseed=42, startLap=0, **kwargs):
    ''' Create an iterator over batches saved to disk.

        Each batch/subset is represented as an instance of a bnpy Data object

        Args
        --------
        Data : bnpy Data object
        nBatch : integer number of batches to divide dataset into
        nLap : integer number of laps to complete
        dataorderseed : int seed for random number generator that determines
                        division of data into fixed set of batches
                        and random order for traversing batches during each lap
    '''
    self.datapath = datapath
    self.nLap = nLap + int(startLap)
    
    # Config order in which batches are traversed
    self.curLapPos = -1
    self.lapID  = int(startLap)
    self.dataorderseed = int(int(dataorderseed) % MAXSEED)

    for extPattern in ['*.ldac', '*.dat', '*.mat']:
      datafileList = glob.glob(os.path.join(datapath, extPattern))
      if len(datafileList) > 0:
        break

    if len(datafileList) == 0:
      raise ValueError('Bad Data Error: No data files found in path.')

    ## Sort file list, in place, so we always have same order
    datafileList.sort()

    initfpath = None
    if datafileList[0].count('InitData') > 0:
      initfpath = datafileList.pop(0)
    elif datafileList[-1].count('InitData') > 0:
      initfpath = datafileList.pop(-1)
    self.initfpath = initfpath

    if nBatch < 1:
      self.nBatch = len(datafileList)
    else:
      self.nBatch = np.minimum(nBatch, len(datafileList))
    self.datafileList = datafileList[:self.nBatch]

    ## Decide which order the batches will be traversed in the first lap
    self.batchOrderCurLap = self.get_rand_order_for_batchIDs_current_lap()

    ## Load entire dataset, like vocab_size and total number of docs
    self.loadWholeDatasetInfo()

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
    ''' Get the Data object for the next batch

        Raises
        --------
        StopIteration if we have completed all specified laps

        Updates (in-place)
        --------
        batchID gives index of batch returned.
  `     lapID gives how many laps have been *completed*.
        curLapPos indicates progress through current lap.

        Returns
        --------
        Data : bnpy Data object for the current batch
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
    return self.loadDataForBatch(self.batchID)

  def loadWholeDatasetInfo(self):
    ''' Load information about entire dataset from disk
    '''
    self.totalSize, self.batchSize = get_total_size(self.datafileList)
    self.DataInfo = dict()
    if self.datafileList[0].endswith('.ldac'):
      vfilepath = os.path.join(self.datapath, 'vocab_size.conf')
      self.DataInfo['vocab_size'] = int(np.loadtxt(vfilepath))
      self.DataInfo['nDocTotal'] = self.totalSize

    dtype = 'XData'
    dtypepath = os.path.join(self.datapath, 'data_type.conf')
    if os.path.exists(dtypepath):
      with open(dtypepath, 'r') as f:
        dtype = f.readline().strip()
    self.dtype = dtype
    if dtype == 'GroupXData':
      self.DataInfo['nDocTotal'] = self.totalSize
    else:
      self.DataInfo['nObsTotal'] = self.totalSize

  def loadDataForBatch(self, batchID):
    ''' Load the data assigned to a particular batch
    '''
    dpath = self.datafileList[batchID]
    if dpath.endswith('.ldac'):
      return WordsData.LoadFromFile_ldac(dpath, **self.DataInfo)
    else:
      if self.dtype == 'GroupXData':
        return GroupXData.LoadFromFile(dpath)
      else:
        return XData.LoadFromFile(dpath)

  def get_rand_order_for_batchIDs_current_lap(self):
    ''' Returns array of batchIDs, permuted in random order
        Order changes each lap (each full traversal of all items)
    '''
    curseed = int(self.dataorderseed + self.lapID)
    PRNG = np.random.RandomState(curseed)
    return PRNG.permutation(self.nBatch)

  #########################################################  I/O methods
  ######################################################### 
  def get_stats_summary(self):
    ''' Returns human-readable summary of this dataset's basic properties
    '''
    if not hasattr(self, 'totalSize'):
      self.totalSize, self.batchSize = get_total_size(self.dataFileList)

    s = '  total size: %d units\n' % (self.totalSize)
    s += '  median batch size: %d units\n' % (self.batchSize)
    s += '  num. batches: %d' % (self.nBatch)
    return s

  def get_text_summary(self):
    ''' Returns human-readable one-line description of this dataset
    '''
    if self.datapath.endswith(os.path.sep):
      dataName = self.datapath.split(os.path.sep)[-2]
    else:
      dataName = self.datapath.split(os.path.sep)[-1]
    return dataName

  ########################################################## Initial data
  ##########################################################
  def loadInitData(self):
    return self.loadDataForBatch(0)


def get_total_size(datafileList):
  totalSize = 0
  curSizes = list()
  for dfile in datafileList:
    curSize = get_size_of_batch_from_file(dfile)
    totalSize += curSize
    curSizes.append(curSize)
  return totalSize, np.median(curSizes) 

def get_size_of_batch_from_file(filepath):
  if filepath.endswith('.ldac'):
    with open(filepath, 'r') as f:
      return len(f.readlines())
  elif filepath.endswith('.mat'):
    try:
      MDict = scipy.io.loadmat(filepath, variable_names=['nObs'])
      return MDict['nObs']
    except:
      MDict = scipy.io.loadmat(filepath, variable_names=['nDoc'])
      return MDict['nDoc']
  else:
    with open(filepath, 'r') as f:
      return len(f.readlines())

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('path', default='')
  parser.add_argument('--nBatch', default=0, type=int)
  parser.add_argument('--nLap', default=1, type=int)
  args = parser.parse_args()
  path = args.path

  if os.path.exists(path):
    DI = DataIteratorFromDisk(path, nLap=args.nLap, nBatch=args.nBatch)
    print DI.get_stats_summary()

    while DI.has_next_batch():
      Dchunk = DI.get_next_batch()
      try:
        print DI.batchID, Dchunk.nDoc, Dchunk.X[0] 
      except:
        print DI.batchID, Dchunk.nObs, Dchunk.X[0]