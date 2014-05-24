'''
TargetDataSampler.py

Provides methods that sample target dataset

Sample selection criteria
---------
* targetMinNumWords (for bag-of-words data only)

'''
import numpy as np
from scipy.spatial.distance import cdist

########################################################### sample_target_data
###########################################################
def sample_target_data(Data, model=None, LP=None, **kwargs):
  ''' Obtain subsample of provided dataset, 

      Returns
      -------
      targetData : bnpy DataObj, with size at most kwargs['targetMaxSize']
  '''
  if hasattr(Data, 'nDoc'):
    return _sample_target_WordsData(Data, model, LP, **kwargs)
  else:
    raise NotImplementedError('TODO: sample_target_data for mix models')

########################################################### WordsData sampling
###########################################################
def _sample_target_WordsData(Data, model=None, LP=None, **kwargs):
  ''' Obtain a subsample of provided set of documents,
        which satisfy criteria set by provided keyword arguments, including
        minimum size of each document, relationship to targeted component, etc.

    Keyword Args
    --------
    targetCompID : int, range: [0, 1, ... K-1]. **optional**
                   if present, we target documents that use a specific topic

    targetMinWordsPerDoc : int,
                           each document in returned targetData
                           must have at least this many words

    targetMinKLPerDoc : int,
                           each document in returned targetData
                           must have at least this KL divergence
                           between model and empirical distribution

    Returns
    --------
    targetData : WordsData dataset,
                  with at most targetMaxSize documents
  '''
  DocWordMat = Data.to_sparse_docword_matrix()
  if kwargs['targetMinWordsPerDoc'] > 0:
    nWordPerDoc = np.asarray(DocWordMat.sum(axis=1))
    candidates = nWordPerDoc >= kwargs['targetMinWordsPerDoc']
    candidates = np.flatnonzero(candidates)
    if len(candidates) < 1:
      return None
  else:
    candidates = None

  hasCompID = 'targetCompID' in kwargs and kwargs['targetCompID'] is not None
  hasDocTopicCountInLP = LP is not None and 'DocTopicCount' in LP
  hasRespInLP = LP is not None and 'resp' in LP
  if hasCompID:
    if hasDocTopicCountInLP:
      if candidates is None:
        Ndk = LP['DocTopicCount'].copy()
      else:
        Ndk = LP['DocTopicCount'][candidates].copy()
      Ndk /= np.sum(Ndk,axis=1)[:,np.newaxis] + 1e-9
      mask = Ndk[:, kwargs['targetCompID']] > kwargs['targetCompFrac']
    elif hasRespInLP:
      mask = LP['resp'][:, kwargs['targetCompID']] > kwargs['targetCompFrac']
      if candidates is not None:
        mask = mask[candidates]
    else:
      raise ValueError('LP must have either DocTopicCount or resp')
    if np.sum(mask) < 1:
      return None
    if candidates is None:
      candidates = np.flatnonzero(mask)
    else:
      candidates = candidates[mask]

  hasWordIDs = 'targetWordIDs' in kwargs and kwargs['targetWordIDs'] is not None
  if hasWordIDs:
    wordIDs = kwargs['targetWordIDs']
    print wordIDs
    TinyMatrix = DocWordMat[candidates, :].toarray()[:, wordIDs]
    targetCountPerDoc = np.sum(TinyMatrix > 0, axis=1)
    mask = targetCountPerDoc >= kwargs['targetWordMinCount']
    candidates = candidates[mask]

  if kwargs['targetMinKLPerDoc'] > 0:
    ### Build model's expected word distribution for each document
    Prior = np.exp( LP['E_logPi'][candidates])
    Lik = np.exp(model.obsModel.getElogphiMatrix())
    DocWordFreq_model = np.dot(Prior, Lik)
    DocWordFreq_model /= DocWordFreq_model.sum(axis=1)[:,np.newaxis]
  
    ### Build empirical word distribution for each document
    DocWordFreq_emp = DocWordMat[candidates].toarray()
    DocWordFreq_emp /= 1e-9 + DocWordFreq_emp.sum(axis=1)[:,np.newaxis]
    KLperDoc = calcKLdivergence_discrete(DocWordFreq_emp, DocWordFreq_model)

    mask = KLperDoc >= kwargs['targetMinKLPerDoc']
    candidates = candidates[mask]
    probCandidates = KLperDoc[mask]
  elif type(kwargs['targetExample']) != int:
    topic = model.obsModel.comp[kwargs['targetCompID']].lamvec
    thr = kwargs['targetMinSize'] + model.obsModel.obsPrior.lamvec
    onTopicWs = np.flatnonzero( topic > thr )
    if len(onTopicWs) == 0:
      probCandidates = None
    else:
      DocWordFreq_emp = DocWordMat[candidates].toarray()
      DocWordFreq_emp[ DocWordFreq_emp > 0] = 1.0
      intersect = np.dot(DocWordFreq_emp[:,onTopicWs],
                         kwargs['targetExample'][onTopicWs] > 0)
      intersect[intersect < 4] = 0
      probCandidates = intersect 
  else:
    probCandidates = None

  if probCandidates is not None:
    if probCandidates.ndim == 0:
      probCandidates = probCandidates[np.newaxis]
    probCandidates = probCandidates * probCandidates # make more peaked
    probCandidates /= probCandidates.sum()

  targetData = Data.get_random_sample(kwargs['targetMaxSize'],
                           randstate=kwargs['randstate'],
                           candidates=candidates, p=probCandidates) 

  if 'targetHoldout' in kwargs and kwargs['targetHoldout']:
    nDoc = targetData.nDoc
    nHoldout = nDoc / 5
    holdIDs = kwargs['randstate'].choice(nDoc, nHoldout, replace=False)
    trainIDs = [x for x in xrange(nDoc) if x not in holdIDs]
    holdData = targetData.select_subset_by_mask(docMask=holdIDs, 
                                             doTrackFullSize=False)
    targetData = targetData.select_subset_by_mask(docMask=trainIDs,
                                              doTrackFullSize=False)
    return targetData, holdData

  return targetData

def calcKLdivergence_discrete(P1, P2):
  KL = np.log(P1 + 1e-100) - np.log(P2 + 1e-100)
  KL *= P1
  return KL.sum(axis=1)


def getDataExemplar(Data):
  ''' Return 'exemplar' for this dataset
  '''
  if Data is None:
    return 0
  start = 0
  stop = Data.doc_range[0,1]
  wordFreq = np.zeros(Data.vocab_size)
  wordFreq[ Data.word_id[ start:stop]] = Data.word_count[start:stop]
  return wordFreq / wordFreq.sum()

def getSize(Data):
  ''' Return the integer size of the provided dataset
  '''
  if Data is None:
    return 0
  elif hasattr(Data, 'nDoc'):
    return Data.nDoc
  else:
    return Data.nObs
