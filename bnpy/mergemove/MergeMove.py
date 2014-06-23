import numpy as np

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
    if MoveInfo['didAccept']:
      mPairIDs = updateCandidatesAfterAcceptedMerge(mPairIDs, kA, kB)
      AcceptedPairs.append((kA, kB))
      ELBOGain += MoveInfo['ELBOGain']
    trialID += 1

  Info = dict(AcceptedPairs=AcceptedPairs, ELBOGain=ELBOGain)
  return curModel, curSS, curELBO, Info


def buildMergeCandidateAndKeepIfImproved(curModel, curSS, curELBO, kA, kB):
  ''' Create candidate model/SS with kA,kB merged, and keep if ELBO improves.
  '''
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

  Info = dict(didAccept=propELBO > curELBO, 
              ELBOGain=propELBO - curELBO)
  if propELBO > curELBO:
    return propModel, propSS, propELBO, Info
  else:
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