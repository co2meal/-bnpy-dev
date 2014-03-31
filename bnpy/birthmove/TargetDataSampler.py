'''
TargetDataSampler.py

Provides methods that sample target dataset

Sample selection criteria
---------
* targetMinNumWords (for bag-of-words data only)

'''
import numpy as np

def sample_target_data(Data, model=None, LP=None, **kwargs):
  if hasattr(Data, 'nDoc'):
    return _sample_target_WordsData(Data, model, LP, **kwargs)
  else:
    raise NotImplementedError('TODO: sample_target_data for mix models')

def _sample_target_WordsData(Data, model=None, LP=None, **kwargs):
  '''

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

  if 'targetCompID' in kwargs and LP is not None:
    Ndk = LP['DocTopicCount'][candidates].copy()
    Ndk /= np.sum(Ndk,axis=1)[:,np.newaxis]
    mask = Ndk[:, kwargs['targetCompID']] > kwargs['targetCompFrac']
    if np.sum(mask) < 1:
      return None
    if candidates is None:
      candidates = np.flatnonzero(mask)
    else:
      candidates = candidates[mask]

  if kwargs['targetMinKLPerDoc'] > 0:
    ### Build model's expected word distribution for each document
    Prior = np.exp( LP['E_logPi'][candidates])
    Lik = np.exp(curModel.obsModel.getElogphiMatrix()[candidates])
    DocWordFreq_model = np.dot(Prior, Lik)
    DocWordFreq_model /= DocWordFreq_model.sum(axis=1)[:,np.newaxis]
  
    ### Build empirical word distribution for each document
    DocWordFreq_emp = DocWordMat[candidates].toarray()
    DocWordFreq_emp /= 1e-9 + DocWordFreq_emp.sum(axis=1)[:,np.newaxis]
    KLperDoc = calcKLdivergence_discrete(DocWordFreq_emp, DocWordFreq_model)

    mask = KLperDoc >= kwargs['targetMinKLPerDoc']
    candidates = candidates[mask]
    probCandidates = probCandidates[mask]
  else:
    probCandidates = None

  targetData = Data.get_random_sample(kwargs['targetMaxSize'],
                           randstate=kwargs['randstate'],
                           candidates=candidates, p=probCandidates) 
  return targetData



def calcKLdivergence_discrete(P1, P2):
  KL = np.log(P1 + 1e-100) - np.log(P2 + 1e-100)
  KL *= P1
  return KL.sum(axis=1)