import numpy as np
import KMeansRex

def create_expanded_suff_stats(Data, curModel, **kwargs):
  ''' Create new suff stats that have useful new topics
      
      Returns
      --------
        expandSS : SuffStatBag with K + Kfresh components
                   will have scale of the provided target dataset Data
  '''
  curLP = curModel.calc_local_params(Data)
  curSS = curModel.get_global_suff_stats(Data, curLP, doPrecompEntropy=True)
  curELBO = curModel.calc_evidence(SS=curSS)

  expandModel = create_expanded_model_with_critical_need_topics(
                      Data, curModel, curLP, **kwargs)
  Kexpand = expandModel.obsModel.K
  Korig = curModel.obsModel.K

  for lap in xrange(kwargs['nFreshLap']):
    xLP = expandModel.calc_local_params(Data)
    xSS = expandModel.get_global_suff_stats(Data, xLP)
    expandModel.partial_update_global_params(xSS, range(Korig, Kexpand))

  # Perform some merges in the expanded model
  xSS, xELBO = cleanup_mergenewcompsintoexisting(Data, expandModel, xLP,
                                                    Korig=Korig, **kwargs)
  if xSS.K == Korig:
    msg = 'BIRTH failed. unable to create useful new comps'
    raise BirthProposalError(msg)

  # Now assess the ELBO of the expanded model
  if xELBO < curELBO:
    msg = 'BIRTH failed. adding %d new comps no better than existing model'
    raise BirthProposalError(msg % (Kexpand - Korig))

  return xSS

def cleanup_mergenewcompsintoexisting(Data, expandModel, xLP, 
                                            Korig=0, **kwargs):
  import MergeMove
  
  xLP = expandModel.calc_local_params(Data, xLP)
  xSS = expandModel.get_global_suff_stats(Data, xLP,
                  doPrecompEntropy=True, doPrecompMergeEntropy=True)
  Kexpand = xSS.K

  mPairIDs = MergeMove.preselect_all_merge_candidates(
              expandModel, expandSS, randstate=kwargs['randstate'],
              preselectroutine=kwargs['cleanuppreselectroutine'], 
              mergePerLap=kwargs['cleanupNumMergeTrials']*(Kexpand-Korig),
              compIDs=range(Korig, Kexpand))
  mPairIDsOrig = [x for x in mPairIDs]

  mergexModel, mergexSS, mergexEv, MTracker = MergeMove.run_many_merge_moves(
                               expandModel, Data, xSS,
                               nMergeTrials=xSS.K**2, 
                               mPairIDs=mPairIDs,
                               randstate=randstate, **kwargs)

  for x in MTracker.acceptedOrigIDs:
    assert x in mPairIDsOrig
  
  targetSS = xSS
  targetSS.setELBOFieldsToZero()
  targetSS.setMergeFieldsToZero()

  return mergexSS, mergexEv


def create_expanded_model_with_critical_need_topics(Data, curModel, curLP,
                                              fracKeep=0.5, 
                                              Kfresh=10, **kwargs):
  '''
  '''
  K = curModel.obsModel.K

  
  Lik = np.exp(curModel.obsModel.getElogphiMatrix())
  Prior = np.exp(LP['E_logPi'][:,:K])

  # DocWordFreq : vocab_size x nDoc
  DocWordFreq_model = np.dot( Lik.T, Prior.T )
  DocWordFreq_model = DocWordFreq_model / DocWordFreq_model.sum(axis=0)

  DocWordFreq_empirical = Data.to_sparse_docword_matrix().toarray().T
  DocWordFreq_empirical = DocWordFreq_empirical + 1e-100
  DocWordFreq_empirical /= DocWordFreq_empirical.sum(axis=0)

  KLperDoc = calcKLdivergence_discrete(DocWordFreq_empirical, DocWordFreq_model)

  sortedDocIDs = np.argsort( -1 * KLperDoc )

  Nkeep = int(fracKeep * len(sortedDocIDs))
  DocWordFreq_missing = DocWordFreq_empirical[:, sortedDocIDs[:Nkeep]] \
                         - DocWordFreq_model[:, sortedDocIDs[:Nkeep]] 
  DocWordFreq_missing = np.maximum(1e-50, DocWordFreq_missing)
  DocWordFreq_missing /= DocWordFreq_missing.sum(axis=0)
  DocWordFreq_missing = DocWordFreq_missing.T.copy(order='F')

  DocWordFreq_clusterctrs, Z = KMeansRex.RunKMeans(DocWordFreq_missing, Kfresh,
                               initname='plusplus',
                               Niter=10, seed=0)
  DocWordFreq_clusterctrs /= DocWordFreq_clusterctrs.sum(axis=1)

  Korig = curModel.obsModel.K
  expandModel = curModel.copy()
  expandModel.insert_global_params( beta=np.ones(Kfresh)/Kfresh,
                                    topics=DocWordFreq_clusterctrs
                                   )
  assert expandModel.allocModel.K == Korig + Kfresh
  assert expandModel.obsModel.K == Korig + Kfresh
  return expandModel


def calcKLdivergence_discrete(P1, P2, axis=0):
  KL = np.log(P1) - np.log(P2)
  KL *= P1
  KL = KL.sum(axis=axis)
  return KL

def viz_docwordfreq_sidebyside(P1, P2, title1='', title2='', 
                                vmax=None, aspect=None):
  from matplotlib import pylab
  pylab.figure()

  if vmax is None:
    vmax = 1.0
    P1limit = np.percentile(P1.flatten(), 97)
    P2limit = np.percentile(P2.flatten(), 97)
    while vmax > P1limit and vmax > P2limit:
      vmax = 0.8 * vmax

  if aspect is None:
    aspect = float(P1.shape[1])/P1.shape[0]
  pylab.subplot(1, 2, 1)
  pylab.imshow(P1, aspect=aspect, interpolation='nearest', vmin=0, vmax=vmax)
  if len(title1) > 0:
    pylab.title(title1)
  pylab.subplot(1, 2, 2)
  pylab.imshow(P2, aspect=aspect, interpolation='nearest', vmin=0, vmax=vmax)
  if len(title2) > 0:
    pylab.title(title2)
  pylab.show(block=False)


  