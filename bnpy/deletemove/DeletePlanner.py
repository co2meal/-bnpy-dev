''' DeletePlanner.py
'''
import numpy as np
from bnpy.deletemove import DeleteLogger

def makePlans(curSS, curmodel=None, curLP=None, 
              ampF=1.0,
              deleteSizeThr=100,
              dtargetMaxSize=1000,
              **kwargs):
  ''' Create plans for collected targeted subsets for delete proposals

      Returns
      --------
      Plans : list of dicts, where each dict has fields
      * selectIDs : list of int IDs in {0, 1, ... K-1}
      * uIDs : list of int IDs in curSS.uIDs
      * selectN : current size of each selected topic
  '''
  ## Determine which topics are eligible for deletion
  Nvec = curSS.getCountVec()
  Nvec = Nvec * ampF
  eligibleIDs = np.flatnonzero(Nvec < deleteSizeThr)

  DeleteLogger.log('<<<<<<<<<<<<<<<<<<<<<<<<< DeletePlanner')
  if len(eligibleIDs) < 1:
    DeleteLogger.log('No eligible topics for deletion.')
    DeleteLogger.logPosVector( np.sort(Nvec))
    return []

  DeleteLogger.log('eligibleIDs')
  eligibleString = ' '.join(['%3d' % (x) for x in eligibleIDs])
  DeleteLogger.log(eligibleString)

  ## Select a subset of eligible topics to use with target set
  # aggregate smallest to biggest until we reach capacity
  eligibleN = Nvec[eligibleIDs] 
  sIDs = np.argsort(eligibleN)
  eligibleMass = np.cumsum(eligibleN[sIDs] )
  maxLoc = np.searchsorted(eligibleMass, dtargetMaxSize)

  selectIDs = eligibleIDs[sIDs[:maxLoc]]
  if len(selectIDs) < 1:
    DeleteLogger.log('No selectable topics for deletion.')
    return []

  DeleteLogger.log('selectIDs')
  selectString = ' '.join(['%3d' % (x) for x in selectIDs])
  DeleteLogger.log(selectString)
  DeleteLogger.logPosVector(eligibleN, fmt='%6.2f')

  uIDs = curSS.uIDs[selectIDs]
  Plan = dict(selectIDs=selectIDs,
              uIDs=uIDs,
              selectN=eligibleN[sIDs[:maxLoc]],
              ampF=ampF,
             )
  return [Plan]