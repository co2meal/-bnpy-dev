'''
DeleteMove.py

'''
import numpy as np

from ..allocmodel.admix import OptimizerForHDPFullVarModel as OptimHDP
from ..allocmodel.admix import LocalStepBagOfWords as LStep

THR=0.99999999

def construct_LP_with_comp_removed(Data, model, compID=0, LP=None,
                                    betamethod='current'):
  ''' Create local params consistent with deleting component at compID.
      Every field in returned dict LP will have scale consistent with Data.

      Returns
      -------
      LP : dict of local parameters
  '''
  if LP is None:
    LP = model.calc_local_params(Data)

  nObs, K = LP['word_variational'].shape  
  Knew = K - 1
  rvec = LP['word_variational'][:,compID]

  Rnew = np.zeros((nObs, Knew))
  Rnew[:, :compID] = LP['word_variational'][:, :compID]
  Rnew[:, compID:] = LP['word_variational'][:, compID+1:]

  Rnew /= (1.0 - rvec)[:,np.newaxis]
  Rnew[rvec > THR] = 1./Knew
  assert np.allclose(1.0, np.sum(Rnew, axis=1))

  newLP = dict()
  newLP['word_variational'] = Rnew
  newLP['DocTopicCount'] = np.zeros((Data.nDoc, Knew))
  for d in xrange(Data.nDoc):
    start = Data.doc_range[d,0]
    stop = Data.doc_range[d,1]
    newLP['DocTopicCount'][d,:] = np.dot(
                                     Data.word_count[start:stop],        
                                     newLP['word_variational'][start:stop,:]
                                       )
  # Estimate the active beta probabilities for remaining comps
  if betamethod == 'current':
    aEbeta = model.allocModel.Ebeta[:-1]
    aEbeta = np.hstack([aEbeta[:compID], aEbeta[compID+1:]])
  elif betamethod == 'prior':
    Ev = 1.0/(1.0 + model.allocModel.alpha0)
    aEbeta = OptimHDP.v2beta(Ev * np.ones(Knew)) 
  assert aEbeta.size == Knew
  remBeta = 1 - np.sum(aEbeta)
  assert np.allclose(1.0, remBeta + np.sum(aEbeta))
  gamma = model.allocModel.gamma

  newLP = LStep.update_theta(newLP, gamma*aEbeta, 
                                    unusedTopicPrior=gamma*remBeta)
  newLP = LStep.update_ElogPi(newLP, gamma*remBeta)
  return newLP

def propose_modelAndSS_with_comp_removed__viaLP(Data, model, compID=0, 
                                              SS=None, newLP=None, LP=None):
  '''
     Return
     --------
     model
     SS
  '''
  if newLP is None:
    newLP = construct_LP_with_comp_removed(Data, model, compID=compID, LP=LP)

  newModel = model.copy()  
  newSS = newModel.get_global_suff_stats(Data, newLP, doPrecompEntropy=True)
  newModel.update_global_params(newSS)
  return newModel, newSS