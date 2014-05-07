'''
DeleteMove.py

'''
import numpy as np

from ..allocmodel.admix import OptimizerForHDPFullVarModel as OptimHDP
from ..allocmodel.admix import LocalStepBagOfWords as LStep

THR=0.99999999

def run_delete_move(Data, model, SS, LP, ELBO=None, compIDs=[], 
                                                    **kwargs):
  if ELBO is None:
    if SS.hasELBOTerms():
      ELBO = model.calc_evidence(SS=SS)
    else:
      ELBO = model.calc_evidence(Data, SS, LP)

  propModel, propSS = propose_model_and_SS_with_comps_removed__viaLP(
                                  Data, model, LP=LP,
                                  compIDs=compIDs, **kwargs)
  assert propSS.hasELBOTerms()
  propELBO = propModel.calc_evidence(SS=propSS)

  MoveInfo = _makeInfo(ELBO, propELBO, compIDs)
  MoveInfo['propModel'] = propModel
  MoveInfo['propSS'] = propSS

  if propELBO > ELBO:
    return propModel, propSS, MoveInfo
  else:
    return model, SS, MoveInfo

def _makeInfo(curELBO, propELBO, compIDs):
  ''' Create dictionary with info about result of attempted delete move.

      Returns
      -------
      MoveInfo : dict with fields
                  msg : string message
                  didRemoveComps : 1 if yes, 0 otherwise
                  compIDs : list of comps that attempted to remove
  '''
  if propELBO > curELBO:
    didRemoveComps = 1
    msg = 'Deletion accepted. propELBO %.4e > curELBO %.4e'
    msg = msg % (propELBO, curELBO)
    elbo = propELBO
  else:
    didRemoveComps = 0
    msg = 'Deletion rejected. propELBO %.4e < curELBO %.4e'
    msg = msg % (propELBO, curELBO)
    elbo = curELBO
  return dict(didRemoveComps=didRemoveComps, msg=msg, compIDs=compIDs,
                                             elbo=elbo)

def construct_LP_with_comps_removed(Data, model, compIDs=0, LP=None,
                                    betamethod='current',
                                    remBetaMaxFactor=1.1):
  ''' Create local params consistent with deleting component at compID.
      Every field in returned dict LP will have scale consistent with Data.

      Returns
      -------
      LP : dict of local parameters
  '''
  if type(compIDs) == int:
    compIDs = [compIDs]
  assert type(compIDs) == list or type(compIDs) == np.ndarray

  if LP is None:
    LP = model.calc_local_params(Data)

  nObs, K = LP['word_variational'].shape  
  Knew = K - len(compIDs)
  Rnew = np.zeros((nObs, Knew))
  if len(compIDs) == 1:
    rvec = np.squeeze(LP['word_variational'][:,compIDs[0]])
    remComps = range(compIDs[0]) + range(compIDs[0] + 1, K)
    #Rnew[:, :compIDs[0]] = LP['word_variational'][:, :compIDs[0]]
    #Rnew[:, compIDs[0]:] = LP['word_variational'][:, compIDs[0]+1:]
  else:
    rvec = LP['word_variational'][:, compIDs].sum(axis=1)
    remComps = [k for k in xrange(K) if k not in compIDs]
    
  assert len(remComps) == Knew
  Rnew[:] = LP['word_variational'][:, remComps]

  assert rvec.size == nObs
  assert rvec.ndim == 1
  #Rnew /= (1.0 - rvec)[:,np.newaxis]
  Rnew /= np.sum(Rnew, axis=1)[:,np.newaxis]
  
  mask = np.abs(1.0 - np.sum(Rnew, axis=1)) > 1e-7
  if np.sum(mask) > 0:
    Rnew[mask] = 1./Knew # fallback to 
    print 'NUM MANUALLY RESCUED:  %d/%d' % (np.sum(mask), mask.size)
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
  remEbeta = model.allocModel.Ebeta[-1]
  if betamethod == 'current':
    aEbeta = model.allocModel.Ebeta[:-1]
    aEbeta = aEbeta[remComps].copy()
  elif betamethod == 'prior':
    Ev = 1.0/(1.0 + model.allocModel.alpha0)
    aEbeta = OptimHDP.v2beta(Ev * np.ones(Knew))[:-1]

  # Adjust active and remaining probability vectors, so that
  #  * all sum together to equal unity
  #  * "remaining" left-over mass does not exceed specified limit
  assert aEbeta.size == Knew
  remBeta = 1 - np.sum(aEbeta)
  if remBeta > remEbeta * remBetaMaxFactor:
    aEbeta *= (1.0 - remEbeta * remBetaMaxFactor)/aEbeta.sum()
    remBeta = 1 - np.sum(aEbeta)

  assert np.allclose(1.0, remBeta + np.sum(aEbeta))
  gamma = model.allocModel.gamma

  newLP = LStep.update_theta(newLP, gamma*aEbeta, 
                                    unusedTopicPrior=gamma*remBeta)
  newLP = LStep.update_ElogPi(newLP, gamma*remBeta)
  return newLP

def propose_model_and_SS_with_comps_removed__viaLP(Data, model, compIDs=0, 
                                                      newLP=None, LP=None,
                                                      **kwargs):
  '''
     Return
     --------
     model
     SS
  '''
  if newLP is None:
    newLP = construct_LP_with_comps_removed(Data, model, 
                                                  compIDs=compIDs, LP=LP,
                                                  **kwargs)

  newModel = model.copy()  
  newSS = newModel.get_global_suff_stats(Data, newLP, doPrecompEntropy=True)
  newModel.update_global_params(newSS)
  newModel.update_global_params(newSS)
  return newModel, newSS
