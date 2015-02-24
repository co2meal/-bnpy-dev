'''
FromScratchMult.py

Initialize params of an HModel with multinomial observations from scratch.
'''
import numpy as np
import time
import os

from scipy.special import digamma
from bnpy.birthmove.TargetDataSampler import _sample_target_WordsData

# Import Kmeans routine
hasRexAvailable = True
try:
    import KMeansRex
except ImportError:
    hasRexAvailable = False

# Import spectral anchor-words routine
hasAnchorTopicEstimator = True
try:
    import AnchorTopicEstimator
except ImportError:
    hasAnchorTopicEstimator = False


def init_global_params(obsModel, Data, K=0, seed=0,
                       initname='randexamples',
                       initarg=None,
                       initMinWordsPerDoc=0,
                       **kwargs):
    ''' Initialize parameters for Mult obsModel, in place.

        Returns
        -------
        Nothing. obsModel is updated in place.
    '''
    PRNG = np.random.RandomState(seed)
    K = np.minimum(Data.nDoc, K)

    # Apply pre-processing to initialization Dataset
    # this removes documents with too few tokens, etc.
    if initMinWordsPerDoc > 0:
        targetDataArgs = dict(targetMinWordsPerDoc=initMinWordsPerDoc,
                              targetMaxSize=Data.nDoc,
                              targetMinSize=0,
                              randstate=PRNG)
        tmpData, tmpInfo = _sample_target_WordsData(Data, None, None,
                                                    **targetDataArgs)
        if tmpData is None:
            raise ValueError(
                'InitData preprocessing left no viable docs left.')
        Data = tmpData

    lam = None
    topics = None
    if initname == 'randomlikewang':
        # Sample K topics i.i.d. from Dirichlet with specified parameter
        # this method is exactly done in Chong Wang's onlinehdp code
        lam = PRNG.gamma(1.0, 1.0, (K, Data.vocab_size))
        lam *= Data.nDocTotal * 100.0 / (K * Data.vocab_size)
    else:
        topics = _initTopicWordEstParams(obsModel, Data, PRNG,
                                         K=K,
                                         initname=initname,
                                         initarg=initarg,
                                         seed=seed,
                                         **kwargs)

    InitArgs = dict(lam=lam, topics=topics, Data=Data)
    obsModel.set_global_params(**InitArgs)

    if 'savepath' in kwargs:
        import scipy.io
        topics = obsModel.getTopics()
        scipy.io.savemat(os.path.join(kwargs['savepath'], 'InitTopics.mat'),
                         dict(topics=topics), oned_as='row')


def _initTopicWordEstParams(obsModel, Data, PRNG, K=0,
                            initname='',
                            initarg='',
                            seed=0,
                            **kwargs):
    ''' Create initial guess for the topic-word parameter matrix

        Returns
        --------
        topics : 2D array, size K x Data.vocab_size
                 non-negative entries, rows sum to one
    '''
    if initname == 'randexamples':
        # Choose K documents at random, then
        # use each doc's empirical distribution (+random noise) to seed a topic
        chosenDocIDs = PRNG.choice(Data.nDoc, K, replace=False)
        DocWord = Data.getDocTypeCountMatrix()
        topics = DocWord[chosenDocIDs].copy()
        topics += 0.01 * PRNG.rand(K, Data.vocab_size)

    elif initname == 'plusplus':
        # Sample K documents at random using the 'plusplus' distance criteria
        # then set each of K topics to the empirical distribution of chosen
        # docs
        if not hasRexAvailable:
            raise NotImplementedError("KMeansRex must be on python path")
        X = Data.getDocTypeCountMatrix()
        topics = KMeansRex.SampleRowsPlusPlus(X, K, seed=seed)
        topics += 0.01 * PRNG.rand(K, Data.vocab_size)

    elif initname == 'kmeansplusplus':
        # Cluster all documents into K hard clusters via K-means
        # then set each of K topics to the means of the resulting clusters
        if not hasRexAvailable:
            raise NotImplementedError("KMeansRex must be on python path")
        X = Data.getDocTypeCountMatrix()
        topics, Z = KMeansRex.RunKMeans(X, K, seed=seed,
                                        Niter=25,
                                        initname='plusplus')
        topics += 0.01 * PRNG.rand(K, Data.vocab_size)

    elif initname == 'randomfromarg':
        # Draw K topic-word probability vectors i.i.d. from a Dirichlet
        # using user-provided symmetric parameter initarg
        topics = PRNG.gamma(initarg, 1., (K, Data.vocab_size))

    elif initname == 'randomfromprior':
        # Draw K topic-word probability vectors i.i.d. from their prior
        lam = obsModel.Prior.lam
        topics = PRNG.gamma(lam, 1., (K, Data.vocab_size))

    elif initname == 'anchorword':
        # Set topic-word prob vectors to output of anchor-words spectral method
        if not hasAnchorTopicEstimator:
            raise NotImplementedError(
                "AnchorTopicEstimator must be on python path")

        stime = time.time()
        topics = AnchorTopicEstimator.findAnchorTopics(
            Data, K, seed=seed,
            minDocPerWord=kwargs['initMinDocPerWord'],
            lowerDim=kwargs['initDim'])
        elapsedtime = time.time() - stime

    else:
        raise NotImplementedError('Unrecognized initname ' + initname)

    # ...................................................... end initname switch

    # Double-check for suspicious NaN values
    # These can arise if kmeans delivers any empty clusters
    rowSum = topics.sum(axis=1)
    mask = np.isnan(rowSum)
    if np.any(mask):
        # Fill in any bad rows with uniform noise
        topics[mask] = PRNG.rand(np.sum(mask), Data.vocab_size)

    np.maximum(topics, 1e-100, out=topics)
    topics /= topics.sum(axis=1)[:, np.newaxis]

    # Raise error if any NaN detected
    if np.any(np.isnan(topics)):
        raise ValueError('topics should never be NaN')

    return topics
