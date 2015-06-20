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

  elif initname == 'priorAssortative':
    SS = SuffStatBag(K=K, D=Data.dim)
    Count1 = np.zeros((K,1))
    Count0 = np.zeros((K,1))
    SS.setField('Count1', Count1, dims=('K','D'))
    SS.setField('Count0', Count0, dims=('K','D'))
    obsModel.update_global_params(SS)
    return    

  elif initname == 'kmeansRelational':
    SS = SuffStatBag(K=K, D=Data.dim)
    N = Data.nNodes
    if Data.isSparse:
      X = np.zeros((N,N))
      print 'NUMBER EDGES IS ', len(Data.edgeSet)
      for e in Data.edgeSet:
        X[e[0], e[1]] = 1
    else:
      X = np.reshape(Data.X, (N,N))
    centroids, labels = kmeans2(data=X, k=K, minit='points')
    print np.unique(labels)
    count1 = np.zeros((K,K))
    count0 = np.zeros((K,K))
    for e in Data.edgeSet:
      count1[labels[e[0]],labels[e[1]]] += 1

    count0 = np.random.multinomial(n=N**2-N, pvals=np.ones(K*K)*1/(K*K))
    count0 = count0.reshape((K,K))
    count0 -= count1
    diag = np.diag_indices(K)
    
    count1 = count1[diag].reshape((K,1))
    count0 = count0[diag].reshape((K,1))
    np.maximum(count0, 0, out=count0)

    SS.setField('Count1', count1, dims=('K','D'))
    SS.setField('Count0', count0, dims=('K','D'))
    obsModel.update_global_params(SS)
    return

  elif initname == 'priorDirected':
    SS = SuffStatBag(K=K, D=Data.dim)
    Count1 = np.zeros((K,K,1))
    Count0 = np.zeros((K,K,1))
    SS.setField('Count1', Count1, dims=('K','K','D'))
    SS.setField('Count0', Count0, dims=('K','K','D'))
    obsModel.update_global_params(SS)
    return
  elif initname == 'priorUndirected':
    SS = SuffStatBag(K=K, D=Data.dim, KupperTriangular=K*(K+1)/2)
    Count1 = np.zeros((K*(K+1)/2,1))
    Count0 = np.zeros((K*(K+1)/2,1))
    SS.setField('Count1', Count1, dims=('KupperTriangular','D'))
    SS.setField('Count0', Count0, dims=('KupperTriangular','D'))
    obsModel.update_global_params(SS)
    return

    
  else:
    raise NotImplementedError('Unrecognized initname ' + initname)
  #np.random.seed(seed)
  #resp = np.random.normal(1.0/(K**2), 1.0/(K**2),
  #                        size=(Data.nNodes**2,K,K))
  #resp = np.random.uniform(0.0, 1.0, size=(Data.nNodes**2,K,K))
  #resp = np.maximum(resp, 0)
  #resp /= np.sum(resp, axis=(1,2))[:,np.newaxis,np.newaxis]

  # Using the provided resp for each token,
  # we summarize into sufficient statistics
  # then perform one global step (M step) to get initial global params
  tempLP = dict(resp=resp)
  SS = SuffStatBag(K=K, D=Data.dim)
  SS = obsModel.get_global_suff_stats(Data, SS, tempLP)
  obsModel.update_global_params(SS)
  
