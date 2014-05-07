'''
FromScratchMult.py

Initialize params of HModel with multinomial observations from scratch.
'''
import numpy as np
from scipy.special import digamma
from scipy.cluster import vq
import time

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
                                       **kwargs):
  ''' Initialize parameters for Mult obsModel, in place.

      Returns
      -------
      Nothing. obsModel is updated in place.
  '''    
  PRNG = np.random.RandomState(seed)

  if initname == 'randexamples':
    # Choose K documents at random, then do
    #  M-step to make K clusters, each based on one document
    DocWord = Data.to_sparse_docword_matrix()
    chosenDocIDs = PRNG.choice(Data.nDoc, K, replace=False)
    PhiTopicWord = np.asarray(DocWord[chosenDocIDs].todense())
    PhiTopicWord += 0.01 * PRNG.rand(K, Data.vocab_size)
    PhiTopicWord /= PhiTopicWord.sum(axis=1)[:,np.newaxis]
    obsModel.set_global_params(K=K, topics=PhiTopicWord)

  elif initname == 'randomfromarg':
    # Draw K topic-word probability vectors 
    #  from a Dirichlet with symmetric parameter initarg
    PhiTopicWord = PRNG.gamma(initarg, 1., (K,Data.vocab_size))
    PhiTopicWord += 1e-200
    PhiTopicWord /= PhiTopicWord.sum(axis=1)[:,np.newaxis]
    obsModel.set_global_params(K=K, topics=PhiTopicWord)

  elif initname == 'randomfromprior':
    # Draw K topic-word probability vectors from their prior
    lamvec = hmodel.obsModel.obsPrior.lamvec
    PhiTopicWord = PRNG.gamma(lamvec, 1., (K,Data.vocab_size))
    PhiTopicWord += 1e-200
    PhiTopicWord /= PhiTopicWord.sum(axis=1)[:,np.newaxis]
    obsModel.set_global_params(K=K, topics=PhiTopicWord)

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
    obsModel.set_global_params(K=K, topics=PhiTopicWord)

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
    obsModel.set_global_params(K=K, beta=beta, topics=PhiTopicWord)

  elif initname == 'spectral':
    # Set topic-word prob vectors to output of anchor-words spectral method
    #  which solves an objective similar to LDA
    if not hasSpectralAvailable:
      raise NotImplementedError("AnchorWords must be on python path")
    DocWord = Data.to_sparse_docword_matrix()
    stime = time.time()
    topics = LearnAnchorTopics.run(DocWord, K, seed=seed, loss='L2')
    elapsedtime = time.time() - stime
    print 'SPECTRAL\n %5.1f sec | D=%d, K=%d' % (elapsedtime, DocWord.shape[0], K)
    obsModel.set_global_params(K=K, topics=topics)

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

