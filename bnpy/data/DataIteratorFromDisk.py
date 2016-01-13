'''
DataIteratorFromDisk.py

Object that manages iterating over minibatches stored to disk.

See Also
--------
DataIterator : iterator for in-memory datasets

Usage
--------
Construct by providing the file system path to
the underlying files that define the full-dataset.
>> I = DataIterator('/path/to/folder/', nBatch=10, nLap=3)

To determine if more data remains, call *has_next_batch*
>> I.has_next_batch()

To access the next batch, call the *get_next_batch* method.
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
'''
import os
import sys
import glob
import numpy as np
import scipy.io

from bnpy.data import WordsData, XData, GroupXData

MAXSEED = 1000000

Words_AllocToDataTypeMap = dict(
    FinitMixtureModel='WordsData',
    DPMixtureModel='WordsData',
    FiniteTopicModel='WordsData',
    HDPTopicModel='WordsData',
)

X_AllocToDataTypeMap = dict(
    FinitMixtureModel='XData',
    DPMixtureModel='XData',
    FiniteTopicModel='GroupXData',
    HDPTopicModel='GroupXData',
    FiniteHMM='GroupXData',
    HDPHMM='GroupXData',
)


def decideDataTypeFromModel(aModelType, oModelType):
    """ Decide which dataset format to use for given allocModel/obsModel

    Returns
    -------
    s : string name of type ['XData', 'GroupXData', 'WordsData']
    """
    if oModelType.count('Gauss') or oModelType.count('Bern'):
        try:
            return X_AllocToDataTypeMap[aModelType]
        except KeyError:
            return 'XData'
    elif oModelType.count('Mult'):
        try:
            return Words_AllocToDataTypeMap[aModelType]
        except KeyError:
            return 'WordsData'
    elif len(aModelType) == 0:
        return 'XData'
    else:
        raise ValueError(
            'Unrecognized model combo: ' + aModelType + ' ' + oModelType)


class DataIteratorFromDisk(object):

    """ Object that manages iterating over minibatches of a dataset.

    Methods
    ------
    get_next_batch() : get next minibatch of this dataset.

    Attributes
    ------
    datafileList : list
        each entry is string filepath to a single batch
    nBatch : int
        total number of batches provided dataset is divided into
    nLap : int
        number of laps (passes thru whole dataset) to complete
    batchID : int
        integer ID of the most recent batch returned by get_next_batch()
        batchID has range [0, nBatch-1]
    curLapPos : int
        integer ID of current position in batch order.
        Range is [0, nBatch-1].
        curLapPos is always incremented by 1 after every
        call to get_next_batch()
    lapID : int
        integer ID of the current lap.
        Range is [0, nLap-1].
        lapID is always incremented by one after each lap.
    """

    def __init__(self, dataPath='', aModelType='', oModelType='',
                 nBatch=0, nLap=1, dataorderseed=42, startLap=0, **kwargs):
        ''' Create an iterator over batches saved to disk.

        Each batch/subset is represented by an instance of
        a bnpy.data.DataObj object. Each such batch-specific object is
        configured so that it is aware of the total size of the
        whole dataset.


        Parameters
        ------
        datapath : string
            valid file-system path to directory containing data files
        nBatch : int
            total number of batches provided dataset is divided into
        nLap : int
            number of laps (passes thru whole dataset) to complete
        dataorderseed : int
            seed for random number generator that determines
            random division of data into fixed set of batches
            and random order for visiting batches during each lap
        '''
        self.datapath = dataPath
        self.nLap = nLap + int(startLap)

        # Config order in which batches are traversed
        self.curLapPos = -1
        self.lapID = int(startLap)
        self.dataorderseed = int(int(dataorderseed) % MAXSEED)

        for extPattern in ['*.ldac', '*.dat', '*.mat']:
            datafileList = glob.glob(os.path.join(dataPath, extPattern))
            if len(datafileList) > 0:
                break
        if len(datafileList) == 0:
            raise ValueError('No data files found in path.')
        # Sort file list, in place, so we always have same order
        datafileList.sort()

        # Look for a file in the directory named "InitData.*"
        # If found, we only use that file for initialization!
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

        self.dtype = decideDataTypeFromModel(aModelType, oModelType)
        self.DataInfo = self.loadWholeDatasetInfo()
        if not hasattr(self, 'name') and 'datasetName' in self.DataInfo:
            self.name = self.DataInfo['datasetName']
        # Decide which order the batches will be traversed in the first lap
        self.batchOrderCurLap = self.getRandPermOfBatchIDsForCurLap()

    def has_next_batch(self):
        if self.lapID >= self.nLap:
            return False
        if self.lapID == self.nLap - 1:
            if self.curLapPos == self.nBatch - 1:
                return False
        return True

    def get_next_batch(self, batchIDOnly=False):
        ''' Get the Data object for the next batch

        Keyword args
        ------------
        batchIDOnly : boolean
            If true, return only batch information, not a data object.

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
            self.batchOrderCurLap = self.getRandPermOfBatchIDsForCurLap()

        # Create the DataObj for the current batch
        self.batchID = self.batchOrderCurLap[self.curLapPos]
        if batchIDOnly:
            return self.batchID
        return self.loadDataForBatch(self.batchID)

    def getRandPermOfBatchIDsForCurLap(self):
        ''' Returns array of batchIDs, permuted in random order.

        Random seed used for permutation is determined by:
            seed = dataorderseed + lapID

        This allows us to always jump to lap L
        and get reproduce its order exactly.

        Returns
        -------
        curBatchIDs : 1D array, size nBatch
            random permutation of integers [0, 1, ... nBatch-1]
        '''
        curseed = int(self.dataorderseed + self.lapID)
        PRNG = np.random.RandomState(curseed)
        return PRNG.permutation(self.nBatch)

    def get_stats_summary(self):
        ''' Returns human-readable summary of this dataset's basic properties
        '''
        if not hasattr(self, 'totalSize'):
            self.totalSize, self.batchSize = self.get_total_size(
                self.dataFileList)

        s = '  total size: %d units\n' % (self.totalSize)
        s += '  median batch size: %d units\n' % (self.batchSize)
        s += '  num. batches: %d' % (self.nBatch)
        return s

    def get_text_summary(self):
        ''' Returns human-readable one-line description of this dataset
        '''
        if 'datasetName' in self.DataInfo:
            return self.DataInfo['datasetName']
        elif self.datapath.endswith(os.path.sep):
            dataName = self.datapath.split(os.path.sep)[-2]
        else:
            dataName = self.datapath.split(os.path.sep)[-1]
        return dataName

    def get_total_size(self, datafileList):
        totalSize = 0
        curSizes = list()
        for dfile in datafileList:
            curSize = self.get_size_of_batch_from_file(dfile)
            totalSize += curSize
            curSizes.append(curSize)
        return totalSize, np.median(curSizes)

    def get_size_of_batch_from_file(self, filepath):
        if filepath.endswith('.ldac'):
            with open(filepath, 'r') as f:
                return len(f.readlines())
        elif filepath.endswith('.mat'):
            if self.dtype == 'XData':
                MDict = scipy.io.loadmat(
                    filepath, variable_names=['doc_range'])
                return int(MDict['doc_range'][0, -1])
            else:
                MDict = scipy.io.loadmat(
                    filepath, variable_names=['doc_range'])
                return MDict['doc_range'].size - 1
        else:
            raise ValueError('Unrecognized file type: ' + filepath)

    def loadWholeDatasetInfo(self):
        ''' Load information about entire dataset from disk

        Returns
        -------
        DataInfo : dict
            contains important format-specific fields defining total size
            * nDocTotal [for GroupXData]
            * nObsTotal [for XData]
        '''
        self.totalSize, self.batchSize = self.get_total_size(self.datafileList)
        conffilepath = os.path.join(self.datapath, 'Info.conf')
        if os.path.exists(conffilepath):
            DataInfo = loadDictFromConfFile(conffilepath)
        else:
            DataInfo = dict()
        if self.datafileList[0].endswith('.ldac'):
            if 'vocab_size' not in DataInfo:
                vocab_size = int(os.environ['W'])
                DataInfo['vocab_size'] = vocab_size
            DataInfo['nDocTotal'] = self.totalSize
        elif self.dtype == 'GroupXData':
            DataInfo['nDocTotal'] = self.totalSize
        else:
            DataInfo['nObsTotal'] = self.totalSize
        DataInfo['dtype'] = self.dtype
        return DataInfo

    def loadDataForBatch(self, batchID):
        ''' Load the data assigned to a particular batch

        Returns
        -------
        Dchunk : bnpy.data.DataObj subclass
        '''
        dpath = self.datafileList[batchID]
        if dpath.endswith('.ldac'):
            return WordsData.LoadFromFile_ldac(dpath, **self.DataInfo)
        else:
            if self.dtype == 'GroupXData':
                return GroupXData.LoadFromFile(dpath, **self.DataInfo)
            else:
                return XData.LoadFromFile(dpath, **self.DataInfo)

    def loadInitData(self):
        return self.loadDataForBatch(0)

    def getBatch(self, batchID):
        ''' Returns Data object for requested batch

        Returns
        -------
        Dbatch : bnpy DataObj
        '''
        return self.loadDataForBatch(batchID)

    def getDataSliceFunctionHandle(self):
        """ Return function handle that can make data slice objects.

        Useful with parallelized algorithms,
        when we need to use shared memory.

        Returns
        -------
        f : function handle
        """
        return self.loadDataForBatch(0).getDataSliceFunctionHandle()

    def calcSliceArgs(self, batchID, workerID, nWorkers, lapFrac=0):
        SliceInfo = dict(**self.DataInfo)
        SliceInfo['filepath'] = self.datafileList[batchID]
        SliceInfo['sliceID'] = workerID
        SliceInfo['nSlice'] = nWorkers
        SliceInfo['lapFrac'] = lapFrac
        SliceInfo['batchID'] = batchID
        return SliceInfo


def loadDataForSlice(filepath='', dtype='', **kwargs):
    """ Return data object loaded from specific file.

    Keyword args
    ------------
    workerID
    nWorkers
    """
    if filepath.endswith('.ldac'):
        return WordsData.LoadFromFile_ldac(filepath, **kwargs)
    else:
        if dtype == 'GroupXData':
            return GroupXData.LoadFromFile(filepath, **kwargs)
        else:
            return XData.LoadFromFile(filepath, **kwargs)


def loadDictFromConfFile(filepath):
    confDict = dict()
    with open(filepath, 'r') as f:
        for line in f.readlines():
            fields = [s.strip() for s in line.strip().split('=')]
            key = fields[0]
            try:
                val = int(fields[1])
            except ValueError:
                val = fields[1]
            confDict[key] = val
    return confDict

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', default='')
    parser.add_argument('--aModelType', default='HDP')
    parser.add_argument('--oModelType', default='Mult')
    parser.add_argument('--nBatch', default=0, type=int)
    parser.add_argument('--nLap', default=1, type=int)
    args = parser.parse_args()
    path = args.path

    if os.path.exists(path):
        DI = DataIteratorFromDisk(path, aModelType=args.aModelType,
                                  oModelType=args.oModelType,
                                  nLap=args.nLap, nBatch=args.nBatch)
        print DI.get_stats_summary()

        while DI.has_next_batch():
            Dchunk = DI.get_next_batch()
            try:
                print DI.batchID, Dchunk.nDoc, Dchunk.X[0].shape
            except:
                print DI.batchID, Dchunk.nObs, Dchunk.X[0].shape
