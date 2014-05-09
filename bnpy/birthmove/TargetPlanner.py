'''
TargetPlanner.py

Handles advanced selection of a plan-of-attack for improving a current model via a birthmove.  

Key methods
--------
  * select_target_comp
'''
import numpy as np
from collections import defaultdict

from BirthProposalError import BirthProposalError

EPS = 1e-14

def select_target_comp(K, SS=None, model=None, LP=None, Data=None,
                           lapsSinceLastBirth=defaultdict(int),
                           excludeList=list(), doVerbose=False, return_ps=False,
                           **kwargs):
  ''' Choose a single component among possible choices {0,1,2, ... K-2, K-1}
      to target with a birth proposal.

      Keyword Args
      -------
      randstate : numpy RandomState object, allows random choice of ktarget
      targetSelectName : string, identifies procedure for selecting ktarget
                          {'uniform','sizebiased', 'delaybiased',
                           'delayandsizebiased'}

      Returns
      -------
      ktarget : int, range: 0, 1, ... K-1
                identifies single component in the current model to target

      Raises
      -------
      BirthProposalError, if cannot select a valid choice
  '''
  targetSelectName = kwargs['targetSelectName']
  if SS is None:
    targetSelectName = 'uniform'
    assert K is not None
  else:
    assert K == SS.K
  
  if len(excludeList) >= K:
    msg = 'BIRTH not possible. All K=%d targets used or excluded.' % (K)
    raise BirthProposalError(msg)

  ######## Build vector ps: gives probability of each choice
  ########
  ps = np.zeros(K)
  if targetSelectName == 'uniform':
    ps = np.ones(K)
  elif targetSelectName == 'sizebiased':
    ps = SS.N.copy()
  elif targetSelectName == 'delaybiased':
    # Bias choice towards components that have not been selected in a long time
    lapDist = np.asarray([lapsSinceLastBirth[kk] for kk in range(K)])
    ps = np.maximum(lapDist + 1e-5, 0)
    ps = ps * ps
  elif targetSelectName == 'delayandsizebiased':
    # Bias choice towards components that have not been selected in a long time
    #  *and* which have many members
    lapDist = np.asarray([lapsSinceLastBirth[kk] for kk in range(K)])
    ps = np.maximum(lapDist + 1e-5, 0)
    ps = ps * ps * SS.N
  elif targetSelectName == 'predictionQuality':
    ps = calc_underprediction_scores_per_topic(K, model, Data, LP,
                                               excludeList, **kwargs)
    ps = ps * ps # make more peaked!
  else:
    raise NotImplementedError('Unrecognized procedure: ' + targetSelectName)
  if SS is not None:
    ps[SS.N < kwargs['targetMinSize']] = 0

  ######## Make a choice using vector ps, if possible. Otherwise, raise error.
  ########
  ps[excludeList] = 0
  if np.sum(ps) < EPS:
    msg = 'BIRTH not possible. All K=%d targets too small or excluded.' % (K)
    raise BirthProposalError(msg)
  ps = ps / np.sum(ps)  
  assert np.allclose( np.sum(ps), 1.0)

  ktarget = kwargs['randstate'].choice(K, p=ps)
  if return_ps:
    return ktarget, ps
  return ktarget

'''
  if doVerbose:
    sortIDs = np.argsort(-1.0 * ps) # sort in decreasing order
    for kk in sortIDs[:6]:
      msg = "comp %3d : %.2f prob | %3d delay | %8d size"
      print msg % (kk, ps[kk], lapsSinceLastBirth[kk], SS.N[kk])
'''


def select_target_words(K, model=None, LP=None, Data=None, SS=None,
                           excludeList=list(), doVerbose=False, 
                           **kwargs):
  ''' Choose a set of vocabulary words to target with a birth proposal.

      Keyword Args
      -------
      randstate : numpy RandomState object, allows random choice of relWords
      targetSelectName : string, identifies procedure for selecting ktarget
                          {'uniform','predictionQuality'}

      Returns
      -------
      relWords : list, each entry in {0, 1, ... Data.vocab_size-1}

      Raises
      -------
      BirthProposalError, if cannot select a valid choice
  '''
  targetSelectName = kwargs['targetSelectName']
  if targetSelectName == 'uniform':
    pWords = np.ones(Data.vocab_size)
  elif targetSelectName == 'predictionQuality':
    pWords = calc_underprediction_scores_per_word(model, 
                                                  Data, LP=None, **kwargs)
  else:
    raise NotImplementedError('Unrecognized procedure: ' + targetSelectName)
  pWords[excludeList] = 0
  if np.sum(pWords) < EPS:
    msg = 'BIRTH not possible. All words have zero probability.'
    raise BirthProposalError(msg)
  relWords = sample_related_words_by_score(Data, pWords, **kwargs)
  if relWords is None or len(relWords) < 1:
    msg = 'BIRTH not possible. Word selection failed.'
    raise BirthProposalError(msg)
  return relWords

def calc_underprediction_scores_per_topic(K, model, Data, LP=None, 
                                             excludeList=list(), **kwargs):
  ''' Calculate for each topic a scalar weight. Larger => worse prediction.
  '''
  if str(type(model.allocModel)).count('HDP') > 0:
    return _hdp_calc_underprediction_scores(K, model, Data, LP, excludeList, 
                                            **kwargs)
  else:
    return _dp_calc_underprediction_scores(K, model, Data, LP, excludeList,         
                                            **kwargs)

def _dp_calc_underprediction_scores(K, model, Data, LP, xList, **kwargs):
  ''' Calculate for each topic a scalar weight. Larger => worse prediction.
  '''
  if LP is None:
    LP = model.calc_local_params(Data)
  assert K == model.allocModel.K

  # Empirical word frequencies (only for docs with enough words)
  empWordFreq = Data.to_sparse_docword_matrix()
  Nd = np.squeeze(np.asarray( empWordFreq.sum(axis=1)))
  candidateDocs = np.flatnonzero(Nd > kwargs['targetMinWordsPerDoc'])
  empWordFreq = empWordFreq[candidateDocs].toarray()
  empWordFreq /= empWordFreq.sum(axis=1)[:,np.newaxis]    
  resp = LP['resp'][candidateDocs]

  # Compare to model's expected frequencies
  score = np.zeros(K)
  for k in range(K):
    if k in xList:
      continue
    lamvec = model.obsModel.comp[k].lamvec
    modelWordFreq_k = lamvec / lamvec.sum()
    for d in np.flatnonzero(resp[:,k] > kwargs['targetCompFrac']):
      score[k] += resp[d,k] * calcKL_discrete(empWordFreq, modelWordFreq_k)
  score = np.maximum(score, 0)
  score /= score.sum()
  return score

def _hdp_calc_underprediction_scores(K, model, Data, LP, xList, **kwargs):
  ''' Calculate for each topic a scalar weight. Larger => worse prediction.
  '''
  if LP is None:
    LP = model.calc_local_params(Data)
  assert K == model.allocModel.K
  NdkThr = 1.0/K * kwargs['targetMinWordsPerDoc']
  score = np.zeros(K)
  for k in range(K):
    if k in xList:
      continue
    lamvec = model.obsModel.comp[k].lamvec
    modelWordFreq_k = lamvec / lamvec.sum()
    candidateDocs = np.flatnonzero(LP['DocTopicCount'][:,k] > NdkThr)
    for d in candidateDocs:
      start = Data.doc_range[d,0]
      stop = Data.doc_range[d,1]
      word_id = Data.word_id[start:stop]
      word_count = Data.word_count[start:stop]
      resp = LP['word_variational'][start:stop, k]
      empWordFreq_k = np.zeros(Data.vocab_size)
      empWordFreq_k[word_id] = (resp * word_count)
      empWordFreq_k = empWordFreq_k / empWordFreq_k.sum()
      probInDoc_k = LP['theta'][d,k] / LP['theta'][d,:].sum()
      score[k] += probInDoc_k * calcKL_discrete(empWordFreq_k, 
                                                     modelWordFreq_k)
  # Make score a valid probability vector
  score = np.maximum(score, 0)
  score /= score.sum()
  return score

def calc_underprediction_scores_per_word(model, Data, LP=None, **kwargs):
  ''' Calculate for each vocab word a scalar score. Larger => worse prediction.
  '''
  if LP is None:
    LP = model.calc_local_params(Data)
  DocWordFreq_emp = calcWordFreqPerDoc_empirical(Data)
  DocWordFreq_model = calcWordFreqPerDoc_model(model, LP)
  uError = np.maximum( DocWordFreq_emp - DocWordFreq_model, 0)
  # For each word, identify set of relevant documents
  DocWordMat = Data.to_sparse_docword_matrix().toarray()
  medianWordCount = np.median(DocWordMat, axis=0)
  score = np.zeros(Data.vocab_size)
  for vID in xrange(Data.vocab_size):
    candidateDocs = np.flatnonzero(DocWordMat[:, vID] > medianWordCount[vID])
    if len(candidateDocs) < 10:
      continue
    score[vID] = np.sum(uError[candidateDocs])
  score = score - np.mean(score) # Center the scores!
  score = np.maximum(score, 0)
  score /= score.sum()
  return score
                  

def sample_related_words_by_score(Data, pscore, nWords=3, anchor=None,
                                                          doVerbose=False,     
                                                          **kwargs):
  ''' Sample set of words that have high underprediction score AND cooccur.
  '''
  DocWordArr = Data.to_sparse_docword_matrix().toarray()
  Cov = np.cov(DocWordArr.T, bias=1)
  sigs = np.sqrt(np.diag(Cov))
  Corr = Cov / np.maximum(np.outer(sigs, sigs), 1e-10)
  posCorr = np.maximum(Corr, 0)
  assert not np.any(np.isnan(posCorr))

  randstate = kwargs['randstate']
  if anchor is None:
    anchors = randstate.choice(Data.vocab_size, nWords, replace=False, p=pscore)
  else:
    anchors = [anchor]
  for firstWord in anchors:
    curWords = [firstWord]
    while len(curWords) < nWords:
      relWordProbs = calc_prob_related_words(posCorr, pscore, curWords)
      if np.sum(relWordProbs) < 1e-14 or np.any(np.isnan(relWordProbs)):
        break
      newWord = randstate.choice(Data.vocab_size, 1, replace=False,
                                                     p=relWordProbs)
      curWords.append(int(newWord))
      if doVerbose:
        print curWords
    if len(curWords) == nWords:
      return curWords
  return anchors

def calc_prob_related_words(posCorr, pscore, curWords):
  relWordProbs = np.prod(posCorr[curWords,:], axis=0) * pscore
  relWordProbs[curWords] = 0
  relWordProbs = np.maximum(relWordProbs, 0)
  relWordProbs /= relWordProbs.sum()
  return relWordProbs
                         
def calcWordFreqPerDoc_empirical(Data, candidateDocs=None):
  ''' Build empirical distribution over words for each document.
  '''
  DocWordMat = Data.to_sparse_docword_matrix()
  if candidateDocs is None:
    DocWordFreq_emp = DocWordMat.toarray()
  else:
    DocWordFreq_emp = DocWordMat[candidateDocs].toarray()
  DocWordFreq_emp /= 1e-9 + DocWordFreq_emp.sum(axis=1)[:,np.newaxis]
  return DocWordFreq_emp

def calcWordFreqPerDoc_model(model, LP, candidateDocs=None):
  ''' Build model's expected word distribution for each document
  '''
  if candidateDocs is None:
    Prior = np.exp( LP['E_logPi']) # D x K
  else:
    Prior = np.exp( LP['E_logPi'][candidateDocs]) # D x K
  Lik = np.exp(model.obsModel.getElogphiMatrix()) # K x V
  DocWordFreq_model = np.dot(Prior, Lik)
  DocWordFreq_model /= DocWordFreq_model.sum(axis=1)[:,np.newaxis]
  return DocWordFreq_model

def calcKL_discrete(P1, P2):
  KL = np.log(P1 + 1e-100) - np.log(P2 + 1e-100)
  KL *= P1
  return KL.sum()
