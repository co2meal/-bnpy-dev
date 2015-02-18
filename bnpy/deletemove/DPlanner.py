''' 
Functions for planning a delete move.

- makePlanForEmptyComps

- makePlanForEligibleComps

- getEligibleCompInfo

- getEligibleCount

'''

import numpy as np
from bnpy.deletemove import DeleteLogger

def makePlanForEmptyComps(curSS, dtargetMinCount=0.01, **kwargs):
    ''' Create a Plan dict for any empty states.

        Returns
        -------
        Plan : dict with either no fields, or two fields named
               * candidateIDs
               * candidateUIDs

        Any "empty" Plan dict indicates that no empty comps exist.
    '''
    Nvec = curSS.getCountVec()
    emptyIDs = np.flatnonzero(Nvec < dtargetMinCount)
    if len(emptyIDs) == 0:
      return dict()
    Plan = dict(candidateIDs=emptyIDs.tolist(),
                candidateUIDs=curSS.uIDs[emptyIDs].tolist(),
               )
    return Plan

def makePlanForEligibleComps(SS, DRecordsByComp=None,
                                 dtargetMaxSize=10,
                                 deleteFailLimit=2,
                                 lapFrac=-1,
                                 **kwargs):
    ''' Create a Plan dict for any non-empty states eligible for a delete move.

        Really just a thin wrapper around getEligibleCompInfo,
        that does logging and verification of correctness.

        Returns
        -------
        Plan : dict with either no fields, or fields named
               * candidateIDs
               * candidateUIDs

        Any "empty" Plan dict indicates that no eligible comps exist.
    '''

    if lapFrac > -1:
        msg = '<<<<<<<<<<<<<<<<<<<< makePlanForEligibleComps @ lap %.2f' \
              % (lapFrac)
        DeleteLogger.log(msg)

    Plan = getEligibleCompInfo(SS, DRecordsByComp, dtargetMaxSize,
                                  deleteFailLimit,
                                  **kwargs)
    nEligibleBySize = len(Plan['eligible-by-size-UIDs'])
    nRemovedByFailLimit = len(Plan['eliminatedUIDs'])
    nFinalCandidates = len(Plan['candidateUIDs'])
    
    DeleteLogger.log('Comp UIDs eligible by size: ')
    if nEligibleBySize == 0:
        DeleteLogger.log('  ZERO.')
    else:
        DeleteLogger.logPosVector(Plan['eligible-by-size-UIDs'], fmt='%5d')

        DeleteLogger.log('Eligible UIDs eliminated by failure count:')
        if nRemovedByFailLimit > 0:
            DeleteLogger.logPosVector(Plan['eliminated-UIDs'], fmt='%5d')
        else:
            DeleteLogger.log('  ZERO.')

        DeleteLogger.log('Comp UIDs selected as candidates:')
        if nFinalCandidates > 0:
            DeleteLogger.logPosVector(Plan['candidateUIDs'], fmt='%5d')
        else:
            DeleteLogger.log('  ZERO. All disqualified.')
    return Plan


def getEligibleCompInfo(SS, DRecordsByComp=None,
                           dtargetMaxSize=10,
                           deleteFailLimit=2,
                           **kwargs):
    ''' Get a dict containing lists of component ids eligible for deletion.

        Returns
        -------
        Info : dict with either no fields, or fields named
               * candidateIDs
               * candidateUIDs

        Any "empty" Info dict indicates that no eligible comps exist.
    '''
    assert hasattr(SS, 'uIDs')
    if DRecordsByComp is None:
        DRecordsByComp = dict()

    ## -------------------------    Measure size of each current state
    # CountVec refers to individual tokens/atoms
    CountVec = SS.getCountVec()

    # SizeVec refers to smallest-possible exchangeable units of data 
    #         e.g. documents in a topic-model, sequences for an HMM
    if SS.hasSelectionTerm('DocUsageCount'):
        SizeVec = SS.getSelectionTerm('DocUsageCount')
    else:
        SizeVec = Nvec

    ## -------------------------    Find states small enough for delete
    eligibleIDs = np.flatnonzero(SizeVec < dtargetMaxSize)
    eligibleUIDs = SS.uIDs[eligibleIDs]

    ## -------------------------    Return blank dict if no eligibles found
    if len(eligibleIDs) == 0:
        return dict()

    ## -------------------------    Filter out records of states 
    ##                              that changed recently
    CountMap = dict()
    SizeMap = dict()
    for ii, uID in enumerate(eligibleUIDs):
        SizeMap[uID] = SizeVec[eligibleIDs[ii]]
        CountMap[uID] = CountVec[eligibleIDs[ii]]

    for uID in DRecordsByComp.keys():
        if uID not in CountMap:
            continue
        count = DRecordsByComp[uID]['count']
        percDiff = np.abs(CountMap[uID] - count) / (count + 1e-14)
        if percDiff > 0.15:
            del DRecordsByComp[uID]

    tier1UIDs = list()
    tier2UIDs = list()
    eliminatedUIDs = list()
    ## --------------------------    Prioritize eligible comps by 
    ##                               * size (smaller preferred)
    ##                               * previous failures (fewer preferred)
    sortIDs = np.argsort(SizeVec[eligibleIDs])
    for ii in sortIDs:
        uID = eligibleUIDs[ii]
        if uID not in DRecordsByComp:
            tier1UIDs.append(uID)
        elif DRecordsByComp[uID]['nFail'] < deleteFailLimit:
            tier2UIDs.append(uID)
        else:
            # Any uID here is ineligible for a delete proposal.
            eliminatedUIDs.append(uID)

    ## --------------------------    Select as many first tier as possible
    ##                               until the target dataset budget is exceeded 
    if len(tier1UIDs) > 0:
        tier1AggSize = np.cumsum([SizeMap[uID] for uID in tier1UIDs])

        # maxLoc is an integer in {0, 1, ... |tier1UIDs|}
        # maxLoc equals m if we want everything at positions 0:m
        maxLoc = np.searchsorted(tier1AggSize, dtargetMaxSize)
        maxPos = np.maximum(0, maxLoc-1)

        selectUIDs = tier1UIDs[:maxLoc]
        curTargetSize = tier1AggSize[maxPos]
     else:
        selectUIDs = list()
        curTargetSize = 0

    ## --------------------------    Fill remaining budget from second tier
    if curTargetSize < dtargetMaxSize:
        tier2AggSize = np.cumsum([SizeMap[x] for x in tier2UIDs])
        maxLoc = np.searchsorted(tier2AggSize, dtargetMaxSize-curTargetSize)
        selectUIDs = np.hstack([selectUIDs, secondUIDs[:maxLoc]])

    selectMass = [SizeMap[x] for x in selectUIDs]
    selectIDs = list()
    for uid in selectUIDs:
        jj = np.flatnonzero(uid == eligibleUIDs)[0]
        selectIDs.append(eligibleIDs[jj])

    Output = dict(CountMap=CountMap, SizeMap=SizeMap)
    Output['eligible-by-size-IDs'] = eligibleIDs
    Output['eligible-by-size-UIDs'] = eligibleUIDs
    Output['eliminatedUIDs'] = eliminatedUIDs
    Output['tier1UIDs'] = tier1UIDs
    Output['tier2UIDs'] = tier2UIDs

    Output['candidateUIDs'] = selectUIDs
    Output['candidateIDs'] = selectIDs
    return Output


def getEligibleCount(SS, **kwargs):
  ''' Get count of all current active comps eligible for deletion

      Returns
      -------
      count : int 
  '''
  Plan = getEligibleCompInfo(SS, **kwargs)
  nTotalEligible = len(Plan['tier1UIDs']) + len(Plan['tier2UIDs'])
  return nTotalEligible
          