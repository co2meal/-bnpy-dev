''' MergePlanner.py

Contains methods necessary for advanced selection of which components to merge.
'''
import numpy as np
from bnpy.mergemove.MergePairSelector import MergePairSelector

def preselect_candidate_pairs(curModel, SS, 
                               randstate=np.random.RandomState(0),
                               preselectroutine='random',
                               mergePerLap=10,
                               **kwargs):
  '''Create and return a list of tuples representing candidates pairs to merge.

     Args
     --------
     curModel : bnpy HModel 
     SS : bnpy SuffStatBag. If None, defaults to random selection.
     randstate : numpy random number generator
     preselectroutine : name of procedure to select candidate pairs
     mergePerLap : int number of candidates to identify 
                     (may be less if K small)            

     Returns
     --------
     mPairList : list of component ID candidates for positions kA, kB
                  each entry is a tuple of two integers
  '''
  kwargs['preselectroutine'] = preselectroutine
  kwargs['randstate'] = randstate
  kwargs['mergePerLap'] = mergePerLap
  if 'excludePairs' not in kwargs:
    excludePairs = list()
  else:
    excludePairs = kwargs['excludePairs']

  K = curModel.allocModel.K  
  nMergeTrials = kwargs['mergePerLap']
  if SS is None: # Handle first lap
    kwargs['preselectroutine'] = 'random'

  # ------------------------------------------------------- Score matrix
  # M : 2D array, shape K x K
  #     M[j,k] = score for viability of j,k.  Larger = better.
  if kwargs['preselectroutine'].count('random') > 0:
    M = kwargs['randstate'].rand(K, K)
  elif kwargs['preselectroutine'].count('obsmodelELBO') > 0:
    M = calcScoreMatrix_obsmodel(curModel, SS, excludePairs)
  elif kwargs['preselectroutine'].count('marglik') > 0:
    MSelector = MergePairSelector()
    M = np.zeros((K, K))
    excludeSet = set(excludePairs)
    for kA in xrange(K):
      for kB in xrange(kA+1, K):
        if (kA,kB) not in excludeSet:
          M[kA, kB] = MSelector._calcMScoreForCandidatePair(curModel,
                                                            SS, kA, kB)
  elif kwargs['preselectroutine'].count('corr') > 0:
    # Use correlation matrix as score for selecting candidates!
    M = calcCorrInCompUsageFromSuffStats(SS)
  else:
    raise NotImplementedError(kwargs['preselectroutine'])

  # Only upper-triangular indices are allowed.
  M[np.tril_indices(K)] = 0
  # Excluded pairs are not allowed.
  M[zip(*excludePairs)] = 0

  # ------------------------------------------------------- Select candidates
  if kwargs['preselectroutine'].count('balanced') > 0:
    aList, bList = _scorematrix2rankedlist_balanced(M, kwargs['mergePerLap'])
  else:
    aList, bList = _scorematrix2rankedlist_greedy(M, kwargs['mergePerLap'])

  # Return completed lists
  assert len(aList) == len(bList)
  assert len(aList) <= kwargs['mergePerLap']
  assert len(aList) <= K * (K-1)/2
  assert np.all( np.asarray(aList) < np.asarray(bList))
  
  if 'returnScoreMatrix' in kwargs and kwargs['returnScoreMatrix']:
    return zip(aList, bList), M

  return zip(aList, bList)


def _scorematrix2rankedlist_greedy(M, nPairs):
  ''' Return the nPairs highest-ranked pairs in score matrix M

      Args
      -------
        M : score matrix, K x K
            should have only entries kA,kB where kA <= kB

      Returns
      --------
        aList : list of integer ids for rows of M
        bList : list of integer ids for cols of M

      Example
      ---------
      _scorematrix2rankedlist( [0 2 3], [0 0 1], [0 0 0], 3)
      >> [ (0,2), (0,1), (1,2)]
  '''
  Mflat = M.flatten()
  sortIDs = np.argsort(-1*Mflat)
  sortIDs = sortIDs[Mflat[sortIDs] != 0]
  bestrs, bestcs = np.unravel_index(sortIDs, M.shape)
  return bestrs[:nPairs].tolist(), bestcs[:nPairs].tolist()


def _scorematrix2rankedlist_balanced(M, mergePerLap):
  ''' Return ranked list of entries in upper-triangular score matrix M,
        *balanced* so that each set of K entries has one each of {1,2,...K}

      Args
      -------
        M : score matrix, K x K
            should have only entries kA,kB where kA <= kB

      Returns
      --------
        aList : list of integer ids for rows of M
        bList : list of integer ids for cols of M
  '''
  K = M.shape[0]
  nKeep = mergePerLap / K + 5
  outPartners = -1 * np.ones( (K, nKeep), dtype=np.int32 )
  outScores = np.zeros( (K, nKeep) )
  for k in xrange(K):
    partnerScores = np.hstack( [M[:k, k], [0], M[k, k+1:]] )
    sortIDs = np.argsort( -1 * partnerScores )
    sortIDs = sortIDs[ partnerScores[sortIDs] != 0 ]
    nK = np.minimum( len(sortIDs), nKeep)
    outPartners[k, :nK] = sortIDs[:nKeep]
    outScores[k, :nK] = partnerScores[sortIDs[:nKeep]]

  mPairSet = set()
  mPairs = list()
  colID = 0
  while len(mPairs) < mergePerLap and colID < nKeep:
    # Pop the next set of scores
    mask = outScores[:, colID] != 0
    curScores = outScores[mask, colID]
    curPartners = outPartners[mask, colID]
    comps = np.arange(K, dtype=np.int32)[mask]
    
    assert not np.any(comps == curPartners)
    aList = np.minimum(comps, curPartners)
    bList = np.maximum(comps, curPartners)

    for k in np.argsort(-1*curScores):
      a = aList[k]
      b = bList[k]
      if (a, b) not in mPairSet:
        mPairs.append( (a,b))
        mPairSet.add((a,b))
    colID += 1
  mPairs = mPairs[:mergePerLap]
  if len(mPairs) < 1:
    return [], []
  elif len(mPairs) == 1:
    return [mPairs[0][0]], [mPairs[0][1]]
  else:
    a,b = zip(*mPairs)
  return list(a), list(b)


def calcScoreMatrix_obsmodel(curModel, SS, excludePairs):
  K = SS.K
  M = np.zeros((K,K))
  excludeSet = set(excludePairs)

  curModel = curModel.copy()
  curModel.obsModel.update_global_params(SS)
  origELBO = curModel.obsModel.calc_evidence(None, SS, None)

  propModel = curModel
  for kA in xrange(K):
    for kB in xrange(kA+1, K):
      if (kA, kB) in excludeSet:
        continue

      mergeSS = SS.copy()
      mergeSS.mergeComps(kA, kB)
      propModel.obsModel.update_global_params(mergeSS)
      propELBO = propModel.obsModel.calc_evidence(None, mergeSS, None)

      if propELBO > origELBO:
        M[kA, kB] = propELBO - origELBO
      else:
        M[kA, kB] = 0
  return M

def calcCorrInCompUsageFromSuffStats(SS):
  '''
     Returns
     -------
     CorrMat : 2D array, size K x K 
               CorrMat[j,k] = correlation coef for comps j,k
  '''
  K = SS.K
  Smat = SS.getSelectionTerm('DocTopicPairMat')
  svec = SS.getSelectionTerm('DocTopicSum')

  nanIDs = np.isnan(Smat)
  offlimitcompIDs = np.logical_or(np.isnan(svec), svec == 0)
  Smat[nanIDs] = 0
  svec[np.isnan(svec)] = 0

  CovMat = Smat / SS.nDoc - np.outer(svec / SS.nDoc, svec / SS.nDoc)
  varc = np.diag(CovMat)
  assert varc.min() >= 0

  sqrtc = np.sqrt(varc)
  sqrtc[offlimitcompIDs] = 1e-20
  CorrMat = CovMat / np.outer(sqrtc, sqrtc)

  # Now, filter to leave only *positive* entries in upper diagonal
  #  we shouldn't even bother trying to merge topics with neg correlations
  CorrMat[np.tril_indices(K)] = 0
  CorrMat[CorrMat < 0] = 0
  CorrMat[nanIDs] = 0
  return CorrMat


'''
def _getAllPairs(K):
  mPairIDs = list()
  for kA in xrange(K):
    for kB in xrange(kA+1, K):
      mPairIDs.append( (kA,kB) )
  return mPairIDs

def _preselect_mergepairs_fromlist(curModel, SS, compIDs, **kwargs):
  K = curModel.obsModel.K
  partnerIDs = set(range(K))
  partnerIDs.difference_update(compIDs)
  partnerIDs = list(partnerIDs)

  allPairs = set(_getAllPairs(K))
  includePairs = list()
  if kwargs['preselectroutine'].count('inclusive') > 0:
    for aa, kA in enumerate(compIDs):
      for kB in compIDs[aa+1:]:
        if kA < kB:
          includePairs.append( (kA, kB) )
        elif kB < kA:
          includePairs.append( (kB, kA) )
  elif kwargs['preselectroutine'].count('bipartite') > 0:
    for kA in compIDs:
      for kB in partnerIDs:
        if kA < kB:
          includePairs.append( (kA, kB) )
        elif kB < kA:
          includePairs.append( (kB, kA) )
  allPairs.difference_update(includePairs)
  excludePairs = list(allPairs)
  return _preselect_mergepairs_simple(curModel, SS, excludePairs=excludePairs,
                                          **kwargs)
'''  