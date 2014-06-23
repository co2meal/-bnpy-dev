'''
FromScratchGauss.py

Initialize params of a mixture model with gaussian observations from scratch.
'''
import numpy as np
from bnpy.util import discrete_single_draw
from bnpy.data import XData
from bnpy.suffstats import SuffStatBag

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
    resp = PRNG.rand(Data.nObs, K)
    resp = resp/np.sum(resp,axis=1)[:,np.newaxis]

  elif initname == 'randomnaive':
    # Generate K "fake" examples from the diagonalized data covariance,
    #  creating params by assigning each "fake" example to a component.
    Sig = np.sqrt(np.diag(np.cov(Data.X.T)))
    Xfake = Sig * PRNG.randn(K, Data.dim)
    Data = XData(Xfake)
    resp = np.eye(K)

  else:
    raise NotImplementedError('Unrecognized initname ' + initname)

  tempLP = dict(resp=resp)
  SS = SuffStatBag(K=K, D=Data.dim)
  SS = obsModel.get_global_suff_stats(Data, SS, tempLP)
  obsModel.update_global_params(SS)
