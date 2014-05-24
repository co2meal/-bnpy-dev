'''
FromScratchMult.py

Initialize params of HModel with multinomial observations from scratch.
'''
import numpy as np
from scipy.special import digamma
from scipy.cluster import vq
import time
import os

from bnpy.birthmove.TargetDataSampler import _sample_target_WordsData

hasRexAvailable = True
hasSpectralAvailable = True
try:
  import KMeansRex
except ImportError:
  hasRexAvailable = False
try:
  import LearnAnchorTopics
except ImportError:
  hasSpectralAvailable = False

def init_global_params(obsModel, Data, K=0, seed=0,
                                       initname='randexamples',
                                       initarg=None,
                                       initMinWordsPerDoc=100,
                                       **kwargs):
  ''' Initialize parameters for Mult obsModel, in place.

      Returns
      -------
      Nothing. obsModel is updated in place.
  '''    
  PRNG = np.random.RandomState(seed)

  if initMinWordsPerDoc > 0:
    print initMinWordsPerDoc
    targetDataArgs = dict(targetMinKLPerDoc=0, 
                            targetMinWordsPerDoc=initMinWordsPerDoc,
                            targetExample=0,
                            targetMinSize=0,
                            targetMaxSize=Data.nDoc,
                            randstate=PRNG)
    Data = _sample_target_WordsData(Data, **targetDataArgs)
  print 'INIT DATA: %d docs' % (Data.nDoc)
  print Data.get_doc_stats_summary()

  wc = Data.word_count.sum()
  if initname == 'randexamples':
    # Choose K documents at random, then do
    #  M-step to make K clusters, each based on one document
    DocWord = Data.to_sparse_docword_matrix()
    chosenDocIDs = PRNG.choice(Data.nDoc, K, replace=False)
    PhiTopicWord = np.asarray(DocWord[chosenDocIDs].todense())
    PhiTopicWord += 0.01 * PRNG.rand(K, Data.vocab_size)
    PhiTopicWord /= PhiTopicWord.sum(axis=1)[:,np.newaxis]
    obsModel.set_global_params(K=K, topics=PhiTopicWord, wordcountTotal=wc)
  elif initname == 'randomfromarg':
    # Draw K topic-word probability vectors 
    #  from a Dirichlet with symmetric parameter initarg
    PhiTopicWord = PRNG.gamma(initarg, 1., (K,Data.vocab_size))
    PhiTopicWord += 1e-200
    PhiTopicWord /= PhiTopicWord.sum(axis=1)[:,np.newaxis]
    obsModel.set_global_params(K=K, topics=PhiTopicWord, wordcountTotal=wc)

  elif initname == 'randomfromprior':
    # Draw K topic-word probability vectors from their prior
    lamvec = obsModel.obsPrior.lamvec
    PhiTopicWord = PRNG.gamma(lamvec, 1., (K,Data.vocab_size))
    PhiTopicWord += 1e-200
    PhiTopicWord /= PhiTopicWord.sum(axis=1)[:,np.newaxis]
    obsModel.set_global_params(K=K, topics=PhiTopicWord, wordcountTotal=wc)

  elif initname == 'randsoftpartition':
    # Assign responsibility for K topics at random to all words
    resp = PRNG.rand(Data.nObs, K)
    resp /= resp.sum(axis=1)[:,np.newaxis]

  elif initname == 'plusplus':
    # Set topic-word prob vectors by drawing K doc-word distributions
    #  using the 'plusplus' distance sampling heuristic
    if not hasRexAvailable:
      raise NotImplementedError("KMeansRex must be on python path")
    DocWord = Data.to_sparse_docword_matrix().toarray()
    PhiTopicWord = KMeansRex.SampleRowsPlusPlus(DocWord, K, seed=seed)
    PhiTopicWord += 0.01 * PRNG.rand(K, Data.vocab_size)
    PhiTopicWord /= PhiTopicWord.sum(axis=1)[:,np.newaxis]
    obsModel.set_global_params(K=K, topics=PhiTopicWord, wordcountTotal=wc)

  elif initname == 'kmeansplusplus':
    # Set topic-word prob vectors to output cluster centers of Kmeans
    #  given the doc-word distributions as input data
    if not hasRexAvailable:
      raise NotImplementedError("KMeansRex must be on python path")
    DocWord = Data.to_sparse_docword_matrix().toarray()
    PhiTopicWord, Z = KMeansRex.RunKMeans(DocWord, K, initname='plusplus',
                                          Niter=10, seed=seed)
    PhiTopicWord += 0.01 * PRNG.rand(K, Data.vocab_size)
    PhiTopicWord /= PhiTopicWord.sum(axis=1)[:,np.newaxis]
    obsModel.set_global_params(K=K, topics=PhiTopicWord, wordcountTotal=wc)

  elif initname == 'spectral':
    # Set topic-word prob vectors to output of anchor-words spectral method
    if not hasSpectralAvailable:
      raise NotImplementedError("AnchorWords must be on python path")

    DocWord = Data.to_sparse_docword_matrix()

    stime = time.time()
    topics = LearnAnchorTopics.run(DocWord, K, seed=seed, loss='L2')
    elapsedtime = time.time() - stime
    print 'SPECTRAL\n %5.1f sec | D=%d, K=%d' \
            % (elapsedtime, DocWord.shape[0], K)
    
    if 'savepath' in kwargs:
      import scipy.io
      scipy.io.savemat(os.path.join(kwargs['savepath'], 'InitTopics.mat'),
                       dict(topics=topics), oned_as='row')
    
    obsModel.set_global_params(K=K, topics=topics, wordcountTotal=wc)

  elif initname.count('mikespectral'):
    if not hasSpectralAvailable:
      raise NotImplementedError("AnchorWords must be on python path")

    stime = time.time()
    lowerDim = kwargs['spectralDim']
    topics, anchorRows = LearnAnchorTopics.runMike(Data, K, 
                        seed=seed, 
                        loss='L2',
                        minDocPerWord=kwargs['spectralMinDocPerWord'],
                        eps=kwargs['spectralEPS'],
                        doRecover=kwargs['spectralDoRecover'],
                        lowerDim=lowerDim,
                                      )
    elapsedtime = time.time() - stime
    print 'MIKE SPECTRAL\n %5.1f sec | D=%d, K=%d, V=%d, lowerDim=%d' \
           % (elapsedtime, Data.nDoc, K, Data.vocab_size, lowerDim)
    
    if 'savepath' in kwargs:
      import scipy.io
      scipy.io.savemat(os.path.join(kwargs['savepath'], 'InitTopics.mat'),
                       dict(topics=topics), oned_as='row')
    
    obsModel.set_global_params(K=K, topics=topics, wordcountTotal=wc)

  else:
    raise NotImplementedError('Unrecognized initname ' + initname)

  if len(obsModel.comp) == K and hasattr(obsModel.comp[0], 'lamvec'):
    assert obsModel.comp[0].lamvec.size == Data.vocab_size
    return

  if obsModel.dataAtomType == 'doc':
    tempLP = dict(resp=resp)
  else:
    tempLP = dict(word_variational=resp)
  SS = SuffStatBag(K=K, D=Data.vocab_size)
  SS = obsModel.get_global_suff_stats(Data, SS, LP)
  obsModel.update_global_params(SS)

