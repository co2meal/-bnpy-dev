'''
FromScratchGauss.py

Initialize params of an HModel with gaussian observations from scratch.
'''
import numpy as np
from bnpy.util import discrete_single_draw
from bnpy.data import XData
from bnpy.suffstats import SuffStatBag
from scipy.cluster.vq import kmeans2

def init_global_params(obsModel, Data, K=0, seed=0,
                                       initname='randexamples',
                                       **kwargs):
  ''' Initialize parameters for Gaussian obsModel, in place.

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
  elif initname == 'randsoftpartition':
    # Randomly assign all data items some mass in each of K components
    #  then create component params by M-step given that soft partition
    resp = PRNG.gamma(1.0/(K*K), 1, size=(Data.nObs, K))
    resp[resp < 1e-3] = 0
    rsum = np.sum(resp,axis=1)
    badIDs = rsum < 1e-8
    if np.any(badIDs): # if any rows have no content, just set them to unif resp.
      resp[badIDs] = 1.0 / K
      rsum[badIDs] = 1
    resp = resp/rsum[:,np.newaxis]
    assert np.allclose(np.sum(resp, axis=1), 1.0)

  elif initname == 'randparams':
    CovMat = np.diag(np.cov(Data.X.T))
    mu = np.sqrt(CovMat) * PRNG.randn(K, Data.dim)
    Sigma = obsModel.get_covar_mat_for_comp('prior')
    Sigma = np.tile(Sigma, (K, 1, 1))
    obsModel.setEstParams(mu=mu, Sigma=Sigma)
    if obsModel.inferType != 'EM':
      obsModel.setPostFromEstParams(obsModel.EstParams, N=Data.nObs)
      del obsModel.EstParams
    return

  elif initname == 'randomnaive':
    # Generate K "fake" examples from the diagonalized data covariance,
    #  creating params by assigning each "fake" example to a component.
    Sig = np.sqrt(np.diag(np.cov(Data.X.T)))
    Xfake = Sig * PRNG.randn(K, Data.dim)
    Data = XData(Xfake)
    resp = np.eye(K)

  elif initname == 'kmeans':
    np.random.seed(seed)
    centroids, labels = kmeans2(data = Data.X, k = K, minit = 'points')
    resp = np.zeros((Data.nObs, K))
    for t in xrange(Data.nObs):
      resp[t,labels[t]] = 1

  else:
    raise NotImplementedError('Unrecognized initname ' + initname)

  tempLP = dict(resp=resp)
  SS = SuffStatBag(K=K, D=Data.dim)
  SS = obsModel.get_global_suff_stats(Data, SS, tempLP)
  obsModel.update_global_params(SS)

