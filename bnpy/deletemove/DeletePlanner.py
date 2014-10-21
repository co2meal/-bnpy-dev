''' DeletePlanner.py
'''
import numpy as np
from bnpy.deletemove import DeleteLogger
import bnpy.data

def makePlanForEmptyTopics(curSS, dtargetMinCount=0.01, **kwargs):
  Nvec = curSS.getCountVec()
  emptyIDs = np.flatnonzero(Nvec < dtargetMinCount)
  if len(emptyIDs) == 0:
    return dict()
  Plan = dict(selectIDs=emptyIDs.tolist(),
              uIDs=curSS.uIDs[emptyIDs].tolist(),
             )
  return Plan

def getEligibleCount(curSS):
  ''' Get list of all eligibleIDs
  '''
  ## TODO
  return 0

def makePlans(curSS, Dchunk=None, 
              lapFrac=0,
              dtargetMaxSize=1000,
              deleteFailLimit=2,
              DRecordsByComp=None,
              **kwargs):
  ''' Create plans for collected targeted subsets for delete proposals

      Returns
      --------
      Plans : list of dicts, where each dict has fields
      * selectIDs : list of int IDs in {0, 1, ... K-1}
      * uIDs : list of int IDs in curSS.uIDs
      * selectN : current size of each selected topic
  '''
  if DRecordsByComp is None:
    DRecordsByComp = dict()

  DeleteLogger.log('<<<<<<<<<<<<<<<<<<<<<<<<< DeletePlanner @ %.2f' % (lapFrac))

  ## Measure size of each current topic
  # SizeVec : refers to docs/units/sequences
  # Nvec/count : refers to tokens/atoms
  Nvec = curSS.getCountVec()
  Nvec, SizeVec = Count2Size(Nvec, Dchunk, curSS, lapFrac)
  
  ## Determine eligibleIDs for deletion
  eligibleIDs = np.flatnonzero(SizeVec < dtargetMaxSize)
  eligibleUIDs = curSS.uIDs[eligibleIDs]
  
  if len(eligibleIDs) < 1:
    DeleteLogger.log('No eligible topics for deletion. Size too big.')
    SizeVec = np.sort(SizeVec)[:10]
    DeleteLogger.logPosVector(SizeVec)
    return []

  CountMap = dict()
  SizeMap = dict()
  for ii, uID in enumerate(eligibleUIDs):
    SizeMap[uID] = SizeVec[eligibleIDs[ii]]
    CountMap[uID] = Nvec[eligibleIDs[ii]]

  for uID in DRecordsByComp.keys():
    if uID not in CountMap:
      continue
    count = DRecordsByComp[uID]['count']
    percDiff = np.abs(CountMap[uID] - count) / count
    if percDiff > 0.15:
      del DRecordsByComp[uID]

  ## Sort by size,  smallest to biggest
  sortIDs = np.argsort(SizeVec[eligibleIDs])
  firstUIDs = list()
  secondUIDs = list()
  elimUIDs = list()
  for ii in sortIDs:
    uID = eligibleUIDs[ii]
    if uID not in DRecordsByComp:
      firstUIDs.append(uID)
    elif DRecordsByComp[uID]['nFail'] < deleteFailLimit:
      secondUIDs.append(uID)
    else:
      elimUIDs.append(uID)

  if len(firstUIDs) < 1 and len(secondUIDs) < 1:
    DeleteLogger.log('No eligible topics for deletion. Fail limit exceeded.')
    DeleteLogger.log('Eliminated IDs:')
    DeleteLogger.logPosVector(elimUIDs, fmt='%5d')
    return []

  DeleteLogger.log('eligibleUIDs')
  eligibleString = ' '.join(['%3d' % (x) for x in firstUIDs[:10]])
  DeleteLogger.log(eligibleString)
  eligibleString = ' '.join(['%3d' % (x) for x in secondUIDs[:10]])
  DeleteLogger.log(eligibleString)

  ## Select a subset of eligible topics to use with target set
  # aggregate smallest to biggest until we reach capacity
  if len(firstUIDs) > 0:
    eligibleMass = np.cumsum([SizeMap[x] for x in firstUIDs])
    # maxLoc gives output in {0, 1, ... nF-1, nF}
    #  maxLoc equals m if we want everything at positions 0:m
    maxLoc = np.searchsorted(eligibleMass, dtargetMaxSize)
    selectUIDs = firstUIDs[:maxLoc]
    
    maxPos = np.maximum(0, maxLoc-1)
    curTargetSize = eligibleMass[maxPos]
    
  else:
    selectUIDs = list()
    curTargetSize = 0

  if curTargetSize < dtargetMaxSize:
    secondMass = np.cumsum([SizeMap[x] for x in secondUIDs])
    maxLoc = np.searchsorted(secondMass, dtargetMaxSize-curTargetSize)
    selectUIDs = np.hstack([selectUIDs, secondUIDs[:maxLoc]])

  selectMass = [SizeMap[x] for x in selectUIDs]
  DeleteLogger.log('selectUIDs: total=%.2f' % (np.sum(selectMass)))
  DeleteLogger.logPosVector(selectUIDs, fmt='%5d')
  DeleteLogger.logPosVector(selectMass, fmt='%5.2f')
  
  selectIDs = list()
  for uid in selectUIDs:
    jj = np.flatnonzero(uid == eligibleUIDs)[0]
    selectIDs.append(eligibleIDs[jj])
  Plan = dict(selectIDs=[x for x in selectIDs],
              uIDs=[x for x in selectUIDs],
             )
  return [Plan]

def Count2Size(Nvec, Dchunk, curSS, lapFrac):
  if lapFrac < 1:
    ampF = Dchunk.get_total_size() / float(Dchunk.get_size())
  else:
    ampF = 1.0
  ampF = np.maximum(ampF, 1.0)
  Nvec = Nvec * ampF
  if isinstance(Dchunk, bnpy.data.WordsData) and hasattr(curSS, 'sumLogPi'):
    # HDP+Mult needs to track SizeVec = nDocsPerTopic
    tokenPerDoc = Dchunk.word_count.sum() / Dchunk.nDoc
    tokenPerTopic = tokenPerDoc / np.ceil(np.sqrt(curSS.K))
    SizeVec = Nvec / tokenPerTopic
  elif isinstance(Dchunk, bnpy.data.GroupXData):
    atomPerTopic = Nvec
    atomPerDoc = np.median(np.diff(Dchunk.doc_range))
    docPerTopic = atomPerTopic / atomPerDoc
    SizeVec = docPerTopic * 3 # counting multiples
  else:
    SizeVec = Nvec
  return Nvec, SizeVec
