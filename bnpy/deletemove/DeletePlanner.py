''' DeletePlanner.py
'''
import numpy as np
from bnpy.deletemove import DeleteLogger

def makePlans(curSS, curmodel=None, curLP=None, 
              ampF=1.0,
              lapFrac=0,
              dtargetMaxSize=1000,
              deleteFailLimit=2,
              NperDoc=1,
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

  ## Determine which topics are eligible for deletion
  Nvec = curSS.getCountVec()
  Nvec = Nvec * ampF
  nDocVec = Nvec / NperDoc
  eligibleIDs = np.flatnonzero(nDocVec < dtargetMaxSize)
  eligibleUIDs = curSS.uIDs[eligibleIDs]
  
  if len(eligibleIDs) < 1:
    DeleteLogger.log('No eligible topics for deletion. Size too big.')
    nDocVec = np.sort(nDocVec)[:10]
    DeleteLogger.logPosVector(nDocVec)
    return []

  CountMap = dict()
  SizeMap = dict()
  for ii, uID in enumerate(eligibleUIDs):
    SizeMap[uID] = nDocVec[eligibleIDs[ii]]
    CountMap[uID] = Nvec[eligibleIDs[ii]]

  for uID in DRecordsByComp.keys():
    if uID not in CountMap:
      continue
    count = DRecordsByComp[uID]['count']
    percDiff = np.abs(CountMap[uID] - count) / count
    if percDiff > 0.15:
      del DRecordsByComp[uID]

  ## Sort by size,  smallest to biggest
  sortIDs = np.argsort(nDocVec[eligibleIDs])
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
    DeleteLogger.logPosVector(elimUIDs)
    return []

  DeleteLogger.log('eligibleUIDs')
  eligibleString = ' '.join(['%3d' % (x) for x in firstUIDs])
  DeleteLogger.log(eligibleString)
  eligibleString = ' '.join(['%3d' % (x) for x in secondUIDs])
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
  DeleteLogger.logPosVector(selectUIDs)
  DeleteLogger.logPosVector(selectMass)
  
  selectIDs = list()
  for uid in selectUIDs:
    jj = np.flatnonzero(uid == eligibleUIDs)[0]
    selectIDs.append(eligibleIDs[jj])
  Plan = dict(selectIDs=selectIDs,
              uIDs=selectUIDs,
              ampF=ampF,
             )
  return [Plan]