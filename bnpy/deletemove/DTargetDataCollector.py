'''
DTargetDataCollector.py

Provides methods that sample target dataset

Sample selection criteria
---------
* targetMinNumWords (for bag-of-words data only)

'''
import numpy as np

import DeleteLogger

def addDataFromBatchToPlan(Plan, Dchunk, hmodel, LPchunk, batchID,
                    lapFrac=None, isFirstBatch=0,
                    dtargetMaxSize=1000,
                    dtargetMinCount=0.01,
                    uIDs=None,
                    **kwargs): 
  ''' Add relevant data from the provided Dchunk to the Plan
  '''
  if isFirstBatch:
    DeleteLogger.log('<<<<<<<<<<<<<<<<<<<<<<<<< DTargetCollector')
  
  relData, relIDs = selectRelevantSubset(Dchunk, hmodel, LPchunk, Plan,
                                         dtargetMinCount=dtargetMinCount)
  if relData is None or relData.get_size() < 1:
    return

  if lapFrac is not None:
    DeleteLogger.log(' lap %6.3f | batch %d | relevant size %5d' 
                     % (lapFrac, batchID, relData.get_size()))

  ## Add all these docs to the Plan
  batchIDs = [batchID for n in xrange(relData.get_size())]

  if hasValidKey(Plan, 'DTargetData'):
    Plan['DTargetData'].add_data(relData)
    Plan['batchIDs'].extend(batchIDs)
  else:
    Plan['DTargetData'] = relData
    Plan['batchIDs'] = batchIDs

  if not hasValidKey(Plan, 'targetSSByBatch'):
    Plan['targetSSByBatch'] = dict()

  curSize = getSize(Plan['DTargetData'])
  if curSize > dtargetMaxSize * 3:
    print 'WARNING: max size exceeded. Target Size=%d' % (dtargetMaxSize)
    
  ## Track a summary of the selected docs
  targetLPchunk = hmodel.allocModel.selectSubsetLP(Dchunk, LPchunk, relIDs)
  targetSSchunk = hmodel.get_global_suff_stats(relData, targetLPchunk)

  Plan['targetSSByBatch'][batchID] = targetSSchunk

  if not hasValidKey(Plan, 'targetSS'):
    Plan['targetSS'] = targetSSchunk.copy()
    Plan['targetSS'].uIDs = uIDs
  else:
    Plan['targetSS'] += targetSSchunk
    if uIDs is not None:
      assert np.allclose(Plan['targetSS'].uIDs, uIDs)

def selectRelevantSubset(Dchunk, hmodel, LPchunk, Plan,
                                 dtargetMinCount=0.01):
  ''' Returns subset of input DataObj that is most representative of the Plan
  '''
  for dd, delCompID in enumerate(Plan['selectIDs']):
    if 'DocTopicCount' in LPchunk:
      curkeepmask = LPchunk['DocTopicCount'][:, delCompID] >= dtargetMinCount
    elif str(type(hmodel.allocModel)).count('HMM'):
      curkeepmask = np.zeros(Dchunk.nDoc, dtype=np.int32)
      for n in xrange(Dchunk.nDoc):
        start = Dchunk.doc_range[n]
        stop = Dchunk.doc_range[n+1]
        Usage_n = np.sum(LPchunk['resp'][start:stop, delCompID])
        curkeepmask[n] = Usage_n >= dtargetMinCount
    else:
      curkeepmask = LPchunk['resp'][:, delCompID] >= dtargetMinCount

    ## Aggregate current mask into one combining all deleteIDs
    if dd > 0:
      keepmask = np.logical_or(keepmask, curkeepmask)
    else:
      keepmask = curkeepmask
  relDocIDs = np.flatnonzero(keepmask)
  if len(relDocIDs) < 1:
    return None, relDocIDs

  relData = Dchunk.select_subset_by_mask(relDocIDs, doTrackFullSize=False)
  return relData, relDocIDs

def hasValidKey(dict, key):
  ''' Return True of key is in the dict and not None, False otherwise 
  '''
  return key in dict and dict[key] is not None

def getSize(Data):
  ''' Return the integer size of the provided dataset
  '''
  if Data is None:
    return 0
  elif hasattr(Data, 'nDoc'):
    return Data.nDoc
  else:
    return Data.nObs
