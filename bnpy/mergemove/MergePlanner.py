'''
MergePlanner.py

Contains methods necessary for advanced selection of which components to merge.
'''
import numpy as np
import bnpy.mergemove.MergeLogger as MergeLogger

# Constant defining how far calculated ELBO gap can be from zero and still be
# considered accepted or favorable
from bnpy.mergemove.MergeMove import ELBO_GAP_ACCEPT_TOL

def preselect_candidate_pairs(curModel, SS, 
                               randstate=np.random.RandomState(0),
                               preselectroutine='random',
                               mergePerLap=10,
                               doLimitNumPairs=1,
                               M=None,
                               **kwargs):
  '''Create and return a list of tuples representing candidate pairs to merge.

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
  if 'excludePairs' not in kwargs:
    excludePairs = list()
  else:
    excludePairs = kwargs['excludePairs']

  K = curModel.allocModel.K
  if doLimitNumPairs:
    nMergeTrials = mergePerLap + kwargs['mergeNumExtraCandidates']
  else:
    nMergeTrials = K * (K-1) // 2

  if SS is None: # Handle first lap
    kwargs['preselectroutine'] = 'random'

  Mraw = None
  # ------------------------------------------------------- Score matrix
  # M : 2D array, shape K x K
  #     M[j,k] = score for viability of j,k.  Larger = better.
  if kwargs['preselectroutine'].count('random') > 0:
    M = kwargs['randstate'].rand(K, K)
  elif kwargs['preselectroutine'].count('marglik') > 0:
    M = calcScoreMatrix_marglik(curModel, SS, excludePairs)
  elif kwargs['preselectroutine'].count('wholeELBO') > 0:
    M, Mraw = calcScoreMatrix_wholeELBO(curModel, SS, excludePairs, M=M)
  elif kwargs['preselectroutine'].count('corr') > 0:
    # Use correlation matrix as score for selecting candidates!
    #M = calcScoreMatrix_corr(SS)
    M = calcScoreMatrix_corrOrEmpty(SS)
  else:
    raise NotImplementedError(kwargs['preselectroutine'])

  # Only upper-triangular indices are allowed.
  M[np.tril_indices(K)] = 0
  # Excluded pairs are not allowed.
  M[zip(*excludePairs)] = 0

  # ------------------------------------------------------- Select candidates
  aList, bList = _scorematrix2rankedlist_greedy(M, nMergeTrials)

  # Return completed lists
  assert len(aList) == len(bList)
  assert len(aList) <= nMergeTrials
  assert len(aList) <= K * (K-1) // 2
  assert np.all( np.asarray(aList) < np.asarray(bList))
  
  if 'returnScoreMatrix' in kwargs and kwargs['returnScoreMatrix']:
    if Mraw is None:
      return zip(aList, bList), M
    else:
      return zip(aList, bList), Mraw
  return zip(aList, bList)


def _scorematrix2rankedlist_greedy(M, nPairs, doKeepZeros=False):
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
  M = M.copy()
  M[ np.tril_indices(M.shape[0]) ] = - np.inf
  Mflat = M.flatten()
  sortIDs = np.argsort(-1*Mflat)
  # Remove any entries that are -Inf
  sortIDs = sortIDs[Mflat[sortIDs] != -np.inf]
  if not doKeepZeros:
    # Remove any entries that are zero
    sortIDs = sortIDs[Mflat[sortIDs] != 0]
  bestrs, bestcs = np.unravel_index(sortIDs, M.shape)
  return bestrs[:nPairs].tolist(), bestcs[:nPairs].tolist()


########################################################### ELBO cues
###########################################################
def calcScoreMatrix_wholeELBO(curModel, SS, excludePairs=list(), M=None):
  ''' Calculate upper-tri matrix of exact ELBO gap for each candidate pair

      Returns
      ---------
      M : 2D array, size K x K. Upper triangular entries carry the content.
          M[j,k] is positive iff merging j,k improves the ELBO
                    0 otherwise
      Mraw : 2D array, size K x K. Uppert tri entries carry content.
          Mraw[j,k] gives the scalar ELBO gap for the potential merge of j,k
  '''
  K = SS.K
  if M is None:
    AGap = curModel.allocModel.calcHardMergeGap_AllPairs(SS)
    OGap = curModel.obsModel.calcHardMergeGap_AllPairs(SS)
    Mraw = AGap + OGap
  else:
    assert M.shape[0] == K
    assert M.shape[1] == K
    nZeroEntry = np.sum(M == 0) - K - K*(K-1)/2
    assert nZeroEntry >= 0
    aList, bList = _scorematrix2rankedlist_greedy(M, SS.K + nZeroEntry, 
                                                     doKeepZeros=True)
    pairList = zip(aList, bList)
    AGap = curModel.allocModel.calcHardMergeGap_SpecificPairs(SS, pairList)
    OGap = curModel.obsModel.calcHardMergeGap_SpecificPairs(SS, pairList)
    M[aList, bList] = AGap + OGap
    Mraw = M

  Mraw[np.triu_indices(K,1)] += ELBO_GAP_ACCEPT_TOL
  M = Mraw.copy()
  M[M<0] = 0
  return M, Mraw

########################################################### Correlation cues
###########################################################
def calcScoreMatrix_corr(SS, MINVAL=1e-8):
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
  Smat[nanIDs] = 0
  svec[np.isnan(svec)] = 0
  offlimitcompIDs = np.logical_or(np.isnan(svec), svec < MINVAL)

  CovMat = Smat / SS.nDoc - np.outer(svec / SS.nDoc, svec / SS.nDoc)
  varc = np.diag(CovMat)

  sqrtc = np.sqrt(varc)
  sqrtc[offlimitcompIDs] = MINVAL

  assert sqrtc.min() >= MINVAL
  CorrMat = CovMat / np.outer(sqrtc, sqrtc)

  # Now, filter to leave only *positive* entries in upper diagonal
  #  we shouldn't even bother trying to merge topics with neg correlations
  CorrMat[np.tril_indices(K)] = 0
  CorrMat[CorrMat < 0] = 0
  CorrMat[nanIDs] = 0

  return CorrMat

def calcScoreMatrix_corrOrEmpty(SS, EMPTYTHR=100):
  ''' Score candidate merge pairs favoring correlations or empty components

      Returns
      -------
      M : 2D array, size K x K
      M[j,k] provides score in [0, 1] for each pair of components (j,k)
      larger score indicates better candidate for merge
  '''
  ## 1) Use correlation scores
  M = calcScoreMatrix_corr(SS)

  ## 2) Add in pairs of (large mass, small mass)
  Nvec = None
  if hasattr(SS, 'N'):
    Nvec = SS.N
  elif hasattr(SS, 'SumWordCounts'):
    Nvec = SS.SumWordCounts

  assert Nvec is not None
  sortIDs = np.argsort(Nvec)
  emptyScores = np.zeros(SS.K)
  for ii in xrange(SS.K/2):
    worstID = sortIDs[ii]
    bestID = sortIDs[-(ii+1)]
    if Nvec[worstID] < EMPTYTHR and Nvec[bestID] > EMPTYTHR:
      # Want to prefer trying *larger* comps before smaller ones 
      # So boost the score of larger comps slightly
      M[worstID, bestID] = 0.5 + 0.1 * Nvec[worstID] / Nvec.sum()
      M[bestID, worstID] = 0.5 + 0.1 * Nvec[worstID] / Nvec.sum()
      if Nvec[worstID] > EMPTYTHR:
        break
      emptyScores[worstID] = Nvec[worstID] / Nvec.sum()

  ## 3) Add in pairs of (small mass, small mass)
  emptyIDs = np.flatnonzero(emptyScores)
  nEmpty = emptyIDs.size
  for jID in xrange(nEmpty-1):
    for kID in xrange(jID+1, nEmpty):
      j = emptyIDs[jID]
      k = emptyIDs[kID]
      M[j, k] = 0.4 + 0.1 * (emptyScores[j] + emptyScores[k])
  return M

########################################################### Marglik cues
###########################################################
def calcScoreMatrix_marglik(curModel, SS, excludePairs):
  K = SS.K
  M = np.zeros((K, K))
  excludeSet = set(excludePairs)
  myCalculator = MargLikScoreCalculator()
  for kA in xrange(K):
    for kB in xrange(kA+1, K):
      if (kA,kB) not in excludeSet:
        M[kA, kB] = myCalculator._calcMScoreForCandidatePair(curModel,
                                                             SS, kA, kB)
  return M

class MargLikScoreCalculator(object):
  ''' Calculate marglik scores quickly by caching 
  '''

  def __init__(self):
    self.MScores = dict()
    self.PairMScores = dict()

  def _calcMScoreForCandidatePair(self, hmodel, SS, kA, kB):
    logmA = self._calcLogMargLikForComp(hmodel, SS, kA)
    logmB = self._calcLogMargLikForComp(hmodel, SS, kB)
    logmAB = self._calcLogMargLikForPair(hmodel, SS, kA, kB)
    return logmAB - logmA - logmB

  def _calcLogMargLikForComp(self, hmodel, SS, kA):
    if kA in self.MScores:
      return self.MScores[kA]
    mA = hmodel.obsModel.calcLogMargLikForComp(SS, kA, doNormConstOnly=True)  
    self.MScores[kA] = mA
    return mA

  def _calcLogMargLikForPair(self, hmodel, SS, kA, kB):
    if (kA,kB) in self.PairMScores:
      return self.PairMScores[ (kA,kB)]
    elif (kB,kA) in self.PairMScores:
      return self.PairMScores[ (kB,kA)]
    else:
      mAB = hmodel.obsModel.calcLogMargLikForComp(SS, kA, kB, doNormConstOnly=True)  
      self.PairMScores[(kA,kB)] = mAB
      return mAB
