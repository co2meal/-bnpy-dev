'''
TargetDataSampler.py

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
                    **kwargs): 
  ''' Add relevant data from the provided Dchunk to the Plan
  '''
  if isFirstBatch:
    DeleteLogger.log('<<<<<<<<<<<<<<<<<<<<<<<<< DTargetCollector')

  ## Grab subset of the docs that meet minimum standards
  for dd, delCompID in enumerate(Plan['selectIDs']):
    curkeepmask = LPchunk['DocTopicCount'][:, delCompID] >= dtargetMinCount
    if dd > 0:
      keepmask = np.logical_or(keepmask, curkeepmask)
    else:
      keepmask = curkeepmask
  relDocIDs = np.flatnonzero(keepmask)
  nRelDocs = len(relDocIDs)

  if lapFrac is not None:
    DeleteLogger.log(' lap %6.3f | batch %d | %d relevant docs' % (lapFrac, batchID, nRelDocs))

  if nRelDocs < 1:
    return



  ## Add all these docs to the Plan
  batchIDs = [batchID for n in xrange(nRelDocs)]
  relData = Dchunk.select_subset_by_mask(relDocIDs, doTrackFullSize=False)

  if hasValidKey(Plan, 'DTargetData'):
    Plan['DTargetData'].add_data(relData)
    Plan['batchIDs'].extend(batchIDs)
  else:
    Plan['DTargetData'] = relData
    Plan['batchIDs'] = batchIDs

  if not hasValidKey(Plan, 'targetSSByBatch'):
    Plan['targetSSByBatch'] = dict()

  if getSize(Plan['DTargetData']) > dtargetMaxSize * 3:
    print 'WARNING: max size exceeded'
    
  ## Track a summary of the selected docs
  targetLPchunk = hmodel.allocModel.selectSubsetLP(Dchunk, LPchunk, relDocIDs)
  targetSSchunk = hmodel.get_global_suff_stats(relData, targetLPchunk)
  Plan['targetSSByBatch'][batchID] = targetSSchunk

  if not hasValidKey(Plan, 'targetSS'):
    Plan['targetSS'] = targetSSchunk.copy()
  else:
    Plan['targetSS'] += targetSSchunk


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
