import numpy as np

import MergeLogger
ELBO_GAP_ACCEPT_TOL = 1e-6

def run_many_merge_moves(curModel, curSS, curELBO, mPairIDs, M=None, 
                                   logFunc=MergeLogger.log,
                                   **kwargs):
  ''' Run many pre-selected merge candidates, keeping all that improve ELBO.

      Returns
      --------
      model : new bnpy HModel
      SS : bnpy SuffStatBag,
      MergeInfo : dict with info about all accepted merges
  '''
  if 'mergeLogVerbose' not in kwargs:
    kwargs['mergeLogVerbose'] = 0

  if kwargs['mergeLogVerbose']:
    MergeLogger.logPhase('MERGE Decisions')

  # eligibleIDs : list from 0, 1, ... len(mPairIDs)
  #  provides index of which original candidate we are now attempting
  eligibleIDs = range(len(mPairIDs))

  CompIDShift = np.zeros(curSS.K, dtype=np.int32)

  if 'mergePerLap' in kwargs:
    nMergeTrials = kwargs['mergePerLap']
  else:
    nMergeTrials = len(mPairIDs)
  trialID = 0
  AcceptedPairs = list()
  AcceptedPairOrigIDs = list()
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

    if M is not None:
      Mcand = M[kA, kB]
      scoreMsg = '%.3e' % (M[kA, kB])
    else:
      Mcand = None
      scoreMsg = ''

    # jA, jB are "shifted" indices under our new model, with K- Kaccepted comps
    jA = kA - CompIDShift[kA]
    jB = kB - CompIDShift[kB]
    curModel, curSS, curELBO, MoveInfo = buildMergeCandidateAndKeepIfImproved(
                                          curModel, curSS, curELBO,
                                          jA, jB, Mcand)
    if kwargs['mergeLogVerbose']:
      MergeLogger.log( '%3d | %3d %3d | % .7e | %s' 
                        % (trialID, kA, kB, MoveInfo['ELBOGain'], scoreMsg)
                     )
    if MoveInfo['didAccept']:
      CompIDShift[kA] = -1
      CompIDShift[kB] = -1
      offIDs = CompIDShift < 0
      CompIDShift[kB+1:] += 1
      CompIDShift[offIDs] = -1

      AcceptedPairs.append((jA, jB))
      AcceptedPairOrigIDs.append((kA, kB))
      ELBOGain += MoveInfo['ELBOGain']
    trialID += 1

  if kwargs['mergeLogVerbose']:
    MergeLogger.logPhase('MERGE summary')
  logFunc( ' %d/%d accepted. ev increased % .4e' 
                    % (len(AcceptedPairs), trialID, ELBOGain))

  Info = dict(AcceptedPairs=AcceptedPairs,
              AcceptedPairOrigIDs=AcceptedPairOrigIDs,
              ELBOGain=ELBOGain)
  return curModel, curSS, curELBO, Info


def buildMergeCandidateAndKeepIfImproved(curModel, curSS, curELBO, kA, kB, Mcur=0):
  ''' Create candidate model/SS with kA,kB merged, and keep if ELBO improves.
  '''
  assert not np.isnan(curELBO)
  assert not np.isinf(curELBO)

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
  assert not np.isinf(propELBO)

  didAccept = propELBO > curELBO - ELBO_GAP_ACCEPT_TOL
  Info = dict(didAccept=didAccept, 
              ELBOGain=propELBO - curELBO,
             )

  if didAccept:
    return propModel, propSS, propELBO, Info
  else:
    return curModel, curSS, curELBO, Info
