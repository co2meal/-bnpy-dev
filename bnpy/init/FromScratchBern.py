'''
FromScratchBern.py

Initialize global params of Bernoulli data-generation model, from scratch.

'''

import numpy as np
from bnpy.util import discrete_single_draw
from bnpy.data import XData
from bnpy.suffstats import SuffStatBag
from scipy.cluster.vq import kmeans2

def init_global_params(obsModel, Data, K=0, seed=0,
                                       initname='randexamples',
                                       initBlockLen=20,
                                       **kwargs):
  ''' Initialize parameters for Bernoulli obsModel, in place.

      Returns
      -------
      Nothing. obsModel is updated in place.
  '''
  PRNG = np.random.RandomState(seed)
  X = Data.X
  if initname == 'randexamples':
    # Choose K items uniformly at random from the Data
    #    then component params by M-step given those single items
    resp = np.zeros((Data.nObs, K))
    permIDs = PRNG.permutation(Data.nObs).tolist()
    for k in xrange(K):
      resp[permIDs[k],k] = 1.0

  elif initname == 'randexamplesbydist':
    # Choose K items from the Data,
    #  selecting the first at random,
    #  then subsequently proportional to euclidean distance to the closest item
    objID = discrete_single_draw(np.ones(Data.nObs), PRNG)
    chosenObjIDs = list([objID])
    minDistVec = np.inf * np.ones(Data.nObs)
    for k in range(1, K):
      curDistVec = np.sum((Data.X - Data.X[objID])**2, axis=1)
      minDistVec = np.minimum(minDistVec, curDistVec)
      objID = discrete_single_draw(minDistVec, PRNG)
      chosenObjIDs.append(objID)
    resp = np.zeros((Data.nObs, K))
    for k in xrange(K):
      resp[chosenObjIDs[k], k] = 1.0


  elif initname == 'randcontigblocks':
    # Choose K contig blocks of provided size from the Data,
    #  selecting each block at random from a particular sequence
    if hasattr(Data, 'doc_range'):
      doc_range = Data.doc_range.copy()
    else:
      doc_range = [0, Data.X.shape[0]]
    nDoc = doc_range.size - 1
    docIDs = np.arange(nDoc)
    PRNG.shuffle(docIDs)
    resp = np.zeros((Data.nObs, K))
    for k in xrange(K):
      n = docIDs[k % nDoc]
      start = doc_range[n]
      stop = doc_range[n+1]
      T = stop - start
      if initBlockLen >= T:
        a = start
        b = stop
      else:
        a = start + PRNG.choice(T - initBlockLen)
        b = a + initBlockLen
      resp[a:b, k] = 1.0

  elif initname == 'kmeans':
    # Fill in resp matrix with hard-clustering from K-means
    # using an initialization with K randomly selected points from X
    np.random.seed(seed)
    centroids, labels = kmeans2(data=Data.X, k=K, minit='points')
    resp = np.zeros((Data.nObs, K))
    for n in xrange(Data.nObs):
      resp[n, labels[n]] = 1

  elif initname == 'PLACEHOLDER':
    print '\n\nTHIS IS A PLACEHOLDER ... SOME ACTUALLY GOOD INIT IS NEEDED\n\n'
    #resp = np.ones((Data.nNodes**2,K,K)) * 1 / (K**2)
    resp = np.random.normal(1.0/(K**2), 1.0/(K**3),
                            size=(Data.nNodes**2,K,K))
    resp = np.maximum(resp, 0)
    resp /= np.sum(resp, axis=(1,2))[:,np.newaxis,np.newaxis]
    print resp[0:2]


  else:
    raise NotImplementedError('Unrecognized initname ' + initname)

  # Using the provided resp for each token,
  # we summarize into sufficient statistics
  # then perform one global step (M step) to get initial global params
  tempLP = dict(resp=resp)
  SS = SuffStatBag(K=K, D=Data.dim)
  SS = obsModel.get_global_suff_stats(Data, SS, tempLP)
  obsModel.update_global_params(SS)
  
