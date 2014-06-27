import numpy as np

import MergeLogger

def run_many_merge_moves(curModel, curSS, curELBO, mPairIDs, M=None, **kwargs):
  ''' Run many pre-selected merge candidates, keeping all that improve ELBO.

      Returns
      --------
      model : new bnpy HModel
      SS : bnpy SuffStatBag,
      MergeInfo : dict with info about all accepted merges
  '''
  if kwargs['mergeLogVerbose']:
    MergeLogger.logPhase('MERGE Decisions')

  # eligibleIDs : list from 0, 1, ... len(mPairIDs)
  #  provides index of which original candidate we are now attempting
  eligibleIDs = range(len(mPairIDs))

  CompIDShift = np.zeros(curSS.K, dtype=np.int32)

  nMergeTrials = kwargs['mergePerLap']
  trialID = 0
  AcceptedPairs = list()
  ELBOGain = 0
  while trialID < nMergeTrials and len(eligibleIDs) > 0:
    if len(eligibleIDs) == 0:
      break
    curID = eligibleIDs.pop(0)

    # kA, kB are the "original" indices, under input model with K comps
    kA, kB = mPairIDs[curID]    
    assert kA < kB

    if CompIDShift[kA] == -1 or CompIDShift[kB] == -1:
      continue

    # jA, jB are the "shifted" indices under our new model, with K- Kaccepted comps
    jA = kA - CompIDShift[kA]
    jB = kB - CompIDShift[kB]
    curModel, curSS, curELBO, MoveInfo = buildMergeCandidateAndKeepIfImproved(
                                          curModel, curSS, curELBO, jA, jB, M[kA,kB])
    
    if M is not None:
      scoreMsg = '%.3e' % (M[kA, kB])
    else:
      scoreMsg = ''

    if kwargs['mergeLogVerbose']:
      MergeLogger.log( '%3d | %3d %3d | % .4e | %s' 
                        % (trialID, jA, jB, MoveInfo['ELBOGain'], scoreMsg)
                     )
    if MoveInfo['didAccept']:
      CompIDShift[kA] = -1
      CompIDShift[kB] = -1
      offIDs = CompIDShift < 0
      CompIDShift[kB+1:] += 1
      CompIDShift[offIDs] = -1

      AcceptedPairs.append((jA, jB))
      ELBOGain += MoveInfo['ELBOGain']
    trialID += 1

  if kwargs['mergeLogVerbose']:
    MergeLogger.logPhase('MERGE summary')
  MergeLogger.log( ' %d/%d accepted. ev increased % .4e' 
                    % (len(AcceptedPairs), trialID, ELBOGain))

  Info = dict(AcceptedPairs=AcceptedPairs, ELBOGain=ELBOGain)
  return curModel, curSS, curELBO, Info


def buildMergeCandidateAndKeepIfImproved(curModel, curSS, curELBO, kA, kB, Mcur=0):
  ''' Create candidate model/SS with kA,kB merged, and keep if ELBO improves.
  '''
  assert not np.isnan(curELBO)

  # Rewrite candidate's kA component to be the merger of kA+kB
  propSS = curSS.copy()
  propSS.mergeComps(kA, kB)
  assert propSS.K == curSS.K - 1

  propModel = curModel.copy()
  propModel.update_global_params(propSS, mergeCompA=kA, mergeCompB=kB)

  # After update_global_params, propModel's comps exactly match propSS's.
  # So at this point, kB has been deleted, and propModel has K-1 components.
  assert propModel.obsModel.K == curModel.obsModel.K - 1
  assert propModel.allocModel.K == curModel.allocModel.K - 1

  # Verify Merge improves the ELBO 
  propELBO = propModel.calc_evidence(SS=propSS)
  assert not np.isnan(propELBO)

  Info = dict(didAccept=propELBO > curELBO, 
              ELBOGain=propELBO - curELBO)

  if propELBO > curELBO:
    return propModel, propSS, propELBO, Info
  else:
    return curModel, curSS, curELBO, Info

"""

def run_many_merge_moves(curModel, curSS, curELBO, mPairIDs, **kwargs):
  ''' Run many pre-selected merge candidates, keeping all that improve ELBO.

      Returns
      --------
      model : new bnpy HModel
      SS : bnpy SuffStatBag,
      MergeInfo : dict with info about all accepted merges
  '''
  nMergeTrials = kwargs['mergePerLap']
  mPairIDs = [x for x in mPairIDs] # Local copy

  MergeLogger.logPhase('MERGE Decisions')

  trialID = 0
  AcceptedPairs = list()
  ELBOGain = 0
  while trialID < nMergeTrials and len(mPairIDs) > 0:
    if len(mPairIDs) == 0:
      break
    kA, kB = mPairIDs.pop(0)
    assert kA < kB

    curModel, curSS, curELBO, MoveInfo = buildMergeCandidateAndKeepIfImproved(
                                          curModel, curSS, curELBO, kA, kB)

    MergeLogger.log( '%3d | %3d %3d | % .4e' 
                      % (trialID, kA, kB, MoveInfo['ELBOGain'])
                   )
    if MoveInfo['didAccept']:
      mPairIDs = updateCandidatesAfterAcceptedMerge(mPairIDs, kA, kB)
      AcceptedPairs.append((kA, kB))
      ELBOGain += MoveInfo['ELBOGain']
    trialID += 1

  MergeLogger.logPhase('MERGE summary')
  MergeLogger.log( ' %3d / %3d % .4e' % (len(AcceptedPairs), trialID, ELBOGain))

  Info = dict(AcceptedPairs=AcceptedPairs, ELBOGain=ELBOGain)
  return curModel, curSS, curELBO, Info



def updateCandidatesAfterAcceptedMerge(mPairIDs, kA, kB):
  ''' Update list of candidate component pairs after a successful merge.

      Args
      --------
      mPairIDs : list of tuples representing candidate pairs
      
      Returns
      --------
      mPairIDs, with updated, potentially fewer entries
      for example, if (0,4) is accepted,
                   then (0,3) and (1, 4) would be discarded.
                    and (2,5), (5,6) would be rewritten as (2,4), (4,5)
  '''
  newPairIDs = list()
  for x0,x1 in mPairIDs:
    if x0 == kA or x1 == kA or x1 == kB or x0 == kB:
      continue
    if x0 > kB: x0 -= 1
    if x1 > kB: x1 -= 1
    newPairIDs.append((x0,x1))
  return newPairIDs
"""
