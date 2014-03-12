import numpy as np
import logging
Log = logging.getLogger('bnpy')
Log.setLevel(logging.DEBUG)
import KMeansRex
from BirthMove import BirthProposalError

def create_expanded_suff_stats(Data, curModel, allSS, **kwargs):
  ''' Create new suff stats that have useful new topics
      
      Returns
      --------
        expandSS : SuffStatBag with K + Kfresh components
                   will have scale of the provided target dataset Data
  '''
  Kfresh = np.minimum(kwargs['Kfresh'], kwargs['Kmax'] - curModel.obsModel.K)
  if Kfresh < 2:
    msg = 'BIRTH: Skipped to avoid exceeding specified limit Kmax=%d'
    raise BirthProposalError(msg % (kwargs['Kmax']))
  kwargs['Kfresh'] = Kfresh

  curLP = curModel.calc_local_params(Data)
  curSS = curModel.get_global_suff_stats(Data, curLP, doPrecompEntropy=True)
  curELBO = curModel.calc_evidence(SS=curSS)

  expandModel = create_expanded_model_with_critical_need_topics(
                      Data, curModel, curLP, allSS=allSS, **kwargs)
  Korig = curModel.obsModel.K
  Kexpand = expandModel.obsModel.K

  # TODO: should we remember the xLP from previous laps?
  for lap in xrange(kwargs['nFreshLap']):
    xLP = expandModel.calc_local_params(Data)
    xSS = expandModel.get_global_suff_stats(Data, xLP)
    unusedRatio = np.abs(xSS.sumLogPiUnused) / np.abs(allSS.sumLogPiUnused)
    if unusedRatio > 100:
      msg = 'BIRTH failed. proposed suff stats invalid. %.2f' % (unusedRatio)
      raise BirthProposalError(msg)
    expandModel.obsModel.update_global_params(xSS, comps=range(Korig, Kexpand))
    expandModel.allocModel.update_global_params(xSS)

  if kwargs['birthVerbose']:
    Log.info("K= %3d | BRAND NEW FRESH MODEL" % (xSS.K))

  # Remove empty topics (assigned to less than 10 words in the target set
  for k in reversed(xrange(Korig, xSS.K)):
    if xSS.N[k] < 10:
      xSS.removeComp(k)
      del expandModel.obsModel.comp[k]
      expandModel.obsModel.K = xSS.K
  if xSS.K < expandModel.allocModel.K:
    xLP = None
    expandModel.allocModel.update_global_params(xSS)
    if kwargs['birthVerbose']:
      Log.info("K= %3d | after removal of empties" % (xSS.K))

  unusedRatio = np.abs(xSS.sumLogPiUnused) / np.abs(allSS.sumLogPiUnused)
  if unusedRatio > 100:
    msg = 'BIRTH failed. proposed suff stats invalid. %.4f' % (unusedRatio)
    raise BirthProposalError(msg)

  assert xSS.K == expandModel.allocModel.K
  Ebeta = expandModel.allocModel.Ebeta
  if np.allclose(Ebeta[:3], [1./2, 1./4, 1./8]):
    msg = 'BIRTH failed. new topic probabilities invalid.'
    raise BirthProposalError(msg)

  # Double-check that we aren't modifying the params for original topics
  diffInputSSandInputParams = curModel.obsModel.comp[0].lamvec[:4] \
                              - allSS.WordCounts[0, :4]
  isSynced = np.allclose(diffInputSSandInputParams,          
                         curModel.obsModel.obsPrior.lamvec[:4])
  if isSynced:
    assert np.allclose(curModel.obsModel.comp[0].lamvec,
                     expandModel.obsModel.comp[0].lamvec)

  if kwargs['cleanupByDeletion']:
    expandModel, xLP, xSS, xELBO = delete_comps_that_improve_ELBO(
                                                      Data, expandModel,
                                                      Korig=Korig, **kwargs)
    if kwargs['birthVerbose']:
      Log.info( "K= %3d | %.3e | after deletion" % (xSS.K, xELBO))

  if expandModel.obsModel.K == Korig:
    msg = 'BIRTH failed. unable to create useful new comps'
    raise BirthProposalError(msg)


  # Merge within new comps only
  expandModel, xSS, xLP, xELBO = cleanup_mergenewcompsonly(Data, expandModel,
                                                    LP=xLP,  
                                                    Korig=Korig, **kwargs)
  if kwargs['birthVerbose']:
    Log.info("K= %3d | %.3e | after merges new-new" % (xSS.K, xELBO))

  if hasattr(Data, 'nDoc') and xELBO > 0:
    msg = 'BIRTH failed. proposed model ELBO invalid.'
    raise BirthProposalError(msg)

  if expandModel.obsModel.K == Korig:
    msg = 'BIRTH failed. unable to create useful new comps'
    raise BirthProposalError(msg)

  if kwargs['cleanupModifyOrigComps']:
    # Merge between new comps and orig comps
    xSS, xELBO = cleanup_mergenewcompsintoexisting(Data, expandModel, xSS, xLP,
                                                    Korig=Korig, **kwargs)
    if kwargs['birthVerbose']:
      Log.info( "K= %3d | %.3e | after merges old-new" % (xSS.K, xELBO))

  if hasattr(Data, 'nDoc') and xELBO > 0:
    msg = 'BIRTH failed. proposed model ELBO invalid.'
    raise BirthProposalError(msg)

  if xSS.K == Korig:
    msg = 'BIRTH failed. unable to create useful new comps'
    raise BirthProposalError(msg)

  # Verify expanded model preferred over current model
  improveEvBound = xELBO - curELBO
  if improveEvBound <= 0 or improveEvBound < 0.00001 * abs(curELBO):
    msg = "BIRTH terminated. Not better than current model on target data."
    msg += "\n  expanded  | K=%3d | %.7e" % (xSS.K, xELBO)
    msg += "\n  current   | K=%3d | %.7e" % (curSS.K, curELBO)
    raise BirthProposalError(msg)

  xSS.setELBOFieldsToZero()
  xSS.setMergeFieldsToZero()
  return xSS

def delete_comps_that_improve_ELBO(Data, model, Korig=0, LP=None,
                                   SS=None, ELBO=None, **kwargs):
  if LP is None:
    LP = model.calc_local_params(Data)
  if SS is None:
    SS = model.get_global_suff_stats(Data, LP, doPrecompEntropy=True)
  if ELBO is None:
    ELBO = model.calc_evidence(SS=SS)

  ''' Iteratively attempt deleting comps K, K-1, K-2, ... Korig
        going in this order makes it easiest to remove components
  '''
  K = SS.K
  for k in reversed(range(Korig, K)):
    rmodel = model.copy()
    rSS = SS.copy()
    rSS.removeComp(k)
    rmodel.obsModel.K = rSS.K
    rmodel.allocModel.update_global_params(rSS, mergeCompB=k)
    del rmodel.obsModel.comp[k]

    rLP = rmodel.calc_local_params(Data)
    rSS = rmodel.get_global_suff_stats(Data, rLP, doPrecompEntropy=True)
    rELBO = rmodel.calc_evidence(SS=rSS)

    if kwargs['doVizBirth'] == 2:
      viz_deletion_sidebyside(model, rmodel, ELBO, rELBO)

    if rELBO >= ELBO:
      SS = rSS
      LP = rLP
      model = rmodel
      ELBO = rELBO      
  return model, LP, SS, ELBO

def calc_ELBO_for_data_under_just_one_topic(Data, curModel, anySS):
  singleModel = curModel.copy()
  singleSS = anySS.getComp(0, doCollapseK1=False)
  singleModel.update_global_params(singleSS)

  singleLP = singleModel.calc_local_params(Data)
  singleSS = singleModel.get_global_suff_stats(Data, singleLP,
                  doPrecompEntropy=True)
  singleModel.update_global_params(singleSS)

  singleELBO = singleModel.calc_evidence(SS=singleSS)
  return singleELBO

def cleanup_mergenewcompsintoexisting(Data, expandModel, xSS, xLP,
                                            Korig=0, **kwargs):
  import MergeMove

  Kexpand = xSS.K
  mPairIDs = MergeMove.preselect_all_merge_candidates(
              expandModel, xSS, randstate=kwargs['randstate'],
              preselectroutine=kwargs['cleanuppreselectroutine'], 
              mergePerLap=kwargs['cleanupNumMergeTrials']*(Kexpand-Korig),
              compIDs=range(Korig, Kexpand))
  mPairIDsOrig = [x for x in mPairIDs]  

  if xLP['K'] != xSS.K:
    # Provided local params are stale, so need to recompute!
    xLP = expandModel.calc_local_params(Data)
  xSS = expandModel.get_global_suff_stats(Data, xLP,
                  doPrecompEntropy=True, doPrecompMergeEntropy=True,
                  mPairIDs=mPairIDs)

  assert 'randstate' in kwargs
  mergexModel, mergexSS, mergexEv, MTracker = MergeMove.run_many_merge_moves(
                               expandModel, Data, xSS,
                               nMergeTrials=xSS.K**2, 
                               mPairIDs=mPairIDs,
                               **kwargs)

  for x in MTracker.acceptedOrigIDs:
    assert x in mPairIDsOrig
  
  targetSS = xSS
  targetSS.setELBOFieldsToZero()
  targetSS.setMergeFieldsToZero()

  return mergexSS, mergexEv

def cleanup_mergenewcompsonly(Data, expandModel, LP=None, 
                                    Korig=0, **kwargs):
  import MergeMove

  mergeModel = expandModel
  Ktotal = mergeModel.obsModel.K

  # Perform many merges among the fresh components
  for trial in xrange(10):
    mPairIDs = list()
    for kA in xrange(Korig, Ktotal):
      for kB in xrange(kA+1, Ktotal):
        mPairIDs.append( (kA,kB) )

    if trial == 0 and LP is not None:
      mLP = LP
    else:
      mLP = mergeModel.calc_local_params(Data)
    mLP['K'] = mergeModel.allocModel.K
    mSS = mergeModel.get_global_suff_stats(Data, mLP,
                    doPrecompEntropy=True, doPrecompMergeEntropy=True,
                    mPairIDs=mPairIDs)

    assert 'randstate' in kwargs
    mergeModel, mergeSS, mergeEv, MTracker = MergeMove.run_many_merge_moves(
                               mergeModel, Data, mSS, 
                               nMergeTrials=len(mPairIDs),
                               mPairIDs=mPairIDs, 
                               **kwargs)
    if mergeSS.K == Ktotal:
      break # no merges happened, so quit trying
    Ktotal = mergeSS.K


  return mergeModel, mergeSS, mLP, mergeEv


def create_expanded_model_with_critical_need_topics(Data, curModel, curLP,
                                              fracKeep=0.5, 
                                              allSS=None,
                                              Kfresh=10, **kwargs):
  '''
  '''
  K = curModel.obsModel.K

  Lik = np.exp(curModel.obsModel.getElogphiMatrix())
  Prior = np.exp(curLP['E_logPi'][:,:K])

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
  DocWordFreq_missing = np.maximum(1e-7, DocWordFreq_missing)
  DocWordFreq_missing /= DocWordFreq_missing.sum(axis=0)
  DocWordFreq_missing = DocWordFreq_missing.T.copy(order='F')

  DocWordFreq_clusterctrs, Z = KMeansRex.RunKMeans(DocWordFreq_missing, Kfresh,
                               initname='plusplus',
                               Niter=10, seed=0)
  DocWordFreq_clusterctrs /= DocWordFreq_clusterctrs.sum(axis=1)[:,np.newaxis]

  Korig = curModel.obsModel.K
  expandModel = curModel.copy()

  """ Here is an old, very bad way to construct the expanded model
      it is bad because we basically keep all old global topic probabilities
      so that the new ones are only allowed to split the mass that "remains"
        which can be very very very small (like 1e-8),
        and results in numerical issues

     expandModel.insert_global_params( beta=np.ones(Kfresh)/Kfresh, K=Kfresh,
                                    topics=DocWordFreq_clusterctrs
                                   )
  """

  freshModel = curModel.copy()
  freshModel.set_global_params(beta=np.ones(Kfresh)/Kfresh, K=Kfresh,
                                    topics=DocWordFreq_clusterctrs
                                   )
  freshLP = freshModel.calc_local_params(Data)
  freshSS = freshModel.get_global_suff_stats(Data, freshLP)
  expandSS = allSS.copy()
  expandSS.insertComps(freshSS)
  expandModel.update_global_params(expandSS)

  if kwargs['doVizBirth'] == 2:
    viz_docwordfreq_sidebyside(DocWordFreq_missing, 
                               DocWordFreq_empirical.T[sortedDocIDs[:Nkeep]],
                               block=True)

  assert expandModel.allocModel.K == Korig + Kfresh
  assert expandModel.obsModel.K == Korig + Kfresh
  return expandModel


def calcKLdivergence_discrete(P1, P2, axis=0):
  KL = np.log(P1) - np.log(P2)
  KL *= P1
  KL = KL.sum(axis=axis)
  return KL

def viz_deletion_sidebyside(model, rmodel, ELBO, rELBO, block=False):
  from ..viz import BarsViz
  from matplotlib import pylab
  pylab.figure()
  h=pylab.subplot(1,2,1)
  BarsViz.plotBarsFromHModel(model, figH=h)
  h=pylab.subplot(1,2,2)
  BarsViz.plotBarsFromHModel(rmodel, figH=h)
  pylab.xlabel("%.3e" % (rELBO - ELBO))
  pylab.show(block=block)

def viz_docwordfreq_sidebyside(P1, P2, title1='', title2='', 
                                vmax=None, aspect=None, block=False):
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
  pylab.show(block=block)
