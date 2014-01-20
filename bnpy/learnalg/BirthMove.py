'''
BirthMove.py

Create new components for a bnpy model.

Usage
--------
Inside a LearnAlg, to try a birth move on a particular model and dataset.
>>> hmodel, SS, evBound, MoveInfo = run_birth_move(hmodel, BirthData, SS)

To force a birth targeted at component "7"
>>> hmodel, SS, evBound, MoveInfo = run_birth_move(hmodel, BirthData, SS, kbirth=7)
'''

import numpy as np
from collections import defaultdict
from .VBLearnAlg import VBLearnAlg
from ..util import EPS, discrete_single_draw
import logging
Log = logging.getLogger('bnpy')
Log.setLevel(logging.DEBUG)

class BirthProposalError( ValueError):
  def __init__( self, *args, **kwargs):
    super(type(self), self).__init__( *args, **kwargs )


###########################################################
###########################################################
def subsample_data(DataObj, LP, targetCompID, targetProbThr=0.1,
                    maxTargetObs=100, maxTargetSize=100, randstate=np.random, 
                    subsampleroutine='keepentiredocuments', **kwargs):
  ''' 
    Select a subsample of the given dataset
      which is primarily associated with component "targetCompID"
      via a simple thresholding procedure
    
    Args
    -------
    DataObj : bnpy dataset object, with nObs observations
    LP : local param dict, containing fields
          resp : nObs x K matrix
    targetCompID : integer within {0, 1, ... K-1}
    ...
    TODO
    
    Returns
    -------
    new DataObj that contains a subset of the data
  '''
  if 'word_variational' in LP:
    # -------------------------------------- WordsData code
    if subsampleroutine == 'keeponlymatchingwords':
      mask = LP['word_variational'][: , targetCompID] > targetProbThr
      TargetData = DataObj.select_subset_by_mask(wordMask=np.flatnonzero(mask),
                                                doTrackFullSize=False)    
    elif subsampleroutine == 'keepentiredocuments':
      # Ndk : empirical doc-topic distribution
      #       rows sum to one
      Ndk = LP['DocTopicCount'].copy()
      Ndk /= np.sum(Ndk,axis=1)[:,np.newaxis]
      docmask = Ndk[:, targetCompID] > targetProbThr
      docIDs = np.flatnonzero(docmask)
      if docIDs.size == 0:
        return None
      # Select at most maxTargetSize of these docs, uniformly at random
      randstate.shuffle(docIDs)
      docIDs = docIDs[:maxTargetSize]
  
      TargetData = DataObj.select_subset_by_mask(docMask=docIDs,
                                                doTrackFullSize=False)    
  else:
    # -------------------------------------- XData code
    mask = LP['resp'][: , targetCompID] > targetProbThr
    objIDs = np.flatnonzero(mask)
    randstate.shuffle(objIDs)
    targetObjIDs = objIDs[:maxTargetObs]
    TargetData = DataObj.select_subset_by_mask(targetObjIDs, 
                                                doTrackFullSize=False)
  return TargetData
  
###########################################################
###########################################################
def run_birth_move(curModel, targetData, SS, randstate=np.random, 
                   doVizBirth=False, ktarget=None, **kwargs):
  ''' Create new model from curModel
        with up to Kbirth new components
  '''
  try:
    freshModel = curModel.copy()
    freshSS = learn_fresh_model(freshModel, targetData, 
                              randstate=randstate, curModel=curModel, **kwargs)
    Kfresh = freshSS.K
    Kold = curModel.obsModel.K
    newSS = SS.copy()
    newSS.insertComps(freshSS)
    newModel = curModel.copy()
    newModel.update_global_params(newSS)

    birthCompIDs = range(Kold, Kold+Kfresh)
    MoveInfo = dict(didAddNew=True,
                    msg='BIRTH: %d fresh comps' % (Kfresh),
                    Kfresh=Kfresh,
                    birthCompIDs=birthCompIDs,
                    freshSS=freshSS)
    if doVizBirth:
      viz_birth_proposal_2D(curModel, newModel, ktarget, birthCompIDs)

    return newModel, newSS, MoveInfo
  except BirthProposalError, e:
    MoveInfo = dict(didAddNew=False, msg=str(e),
                    Kfresh=0, birthCompIDs=[])
    return curModel, SS, MoveInfo

def learn_fresh_model(freshModel, targetData, Kmax=500, Kfresh=10,
                      freshInitName='randexamples', freshAlgName='VB',
                      nFreshLap=50, randstate=np.random,
                      doSimpleThrTest=False, curModel=None, **kwargs):
  ''' Learn a new model with Kfresh components
      Enforces an "upper limit" on number of components Kmax,
        so if Kexisting + Kfresh would exceed Kmax,
          we only consider Kmax-Kexisting components

      Returns
      -------
      freshSS : bnpy SuffStatDict with Kfresh components
  '''
  Kfresh = np.minimum(Kfresh, Kmax - freshModel.obsModel.K)

  if Kfresh < 2:
    raise BirthProposalError('BIRTH: Skipped to avoid exceeding user-specified limit of Kmax=%d components. ' % (Kmax))

  seed = randstate.choice(xrange(100000))
  freshModel.init_global_params(targetData, K=Kfresh, 
                                seed=seed, initname=freshInitName)
 
  LearnAlgConstructor = dict()
  LearnAlgConstructor['VB'] = VBLearnAlg
  algP = dict(nLap=nFreshLap, convergeSigFig=6, startLap=0)
  outP = dict(saveEvery=-1, traceEvery=-1, printEvery=-1)
  learnAlg = LearnAlgConstructor[freshAlgName](savedir=None, 
                    algParams=algP, outputParams=outP, seed=seed)

  targetLP, evBound = learnAlg.fit(freshModel, targetData)
  targetSS = freshModel.get_global_suff_stats(targetData, targetLP)
  
  if doSimpleThrTest:
    Nthr = np.maximum(100, 0.05 * targetData.nObs)
    rejectIDs = np.flatnonzero(targetSS.N < Nthr)
    rejectIDs = np.sort(rejectIDs)[::-1]
    for kreject in rejectIDs:
      targetSS.removeComp(kreject)
    
  else:
    targetSS = clean_up_fresh_model(targetData, curModel, freshModel, 
                            randstate=randstate, **kwargs)
  
  if targetSS.K < 2:
    msg = 'BIRTH: Didnt create >1 useful comp from Data of size %d'
    raise BirthProposalError(msg % (targetData.nObs))
  return targetSS
  
def clean_up_fresh_model(targetData, curModel, freshModel, 
                            randstate=np.random, **mergeKwArgs):
  ''' Returns a "cleaned" fresh model,  
      1) verifies fresh model improves over default (single component) model
      2) perform merges within fresh, requiring improvement on target data
      3) perform merges within full (combined) model,
            aiming only to remove the new/fresh comps
  '''
  import MergeMove

  # Verify fresh model preferred over single comp model
  targetLP = freshModel.calc_local_params(targetData)
  targetSS = freshModel.get_global_suff_stats(targetData, targetLP,
                  doPrecompEntropy=True, doPrecompMergeEntropy=True)
  freshEvBound = freshModel.calc_evidence(SS=targetSS)

  singleModel = curModel.copy()
  singleSS = targetSS.getComp(0, doCollapseK1=False)
  singleModel.update_global_params(singleSS)

  singleLP = singleModel.calc_local_params(targetData)
  singleSS = singleModel.get_global_suff_stats(targetData, singleLP,
                  doPrecompEntropy=True)
  singleModel.update_global_params(singleSS) # make it reflect targetData

  singleEvBound = singleModel.calc_evidence(SS=singleSS)
 
  improveEvBound = freshEvBound - singleEvBound
  if improveEvBound <= 0 or improveEvBound < 0.00001 * abs(singleEvBound):
    msg = "BIRTH terminated. Not better than single component on target data."
    msg += "\n  fresh  | K=%3d | %.7e" % (targetSS.K, freshEvBound)
    msg += "\n  single | K=%3d | %.7e" % (singleSS.K, singleEvBound)
    raise BirthProposalError(msg)

  # Verify fresh model improves over current model 
  curLP = curModel.calc_local_params(targetData)
  curSS = curModel.get_global_suff_stats(targetData, curLP, doPrecompEntropy=True)
  curEvBound = curModel.calc_evidence(SS=curSS)
  improveEvBound = freshEvBound - curEvBound
  if improveEvBound <= 0 or improveEvBound < 0.00001 * abs(curEvBound):
    msg = "BIRTH terminated. Not better than current model on target data."
    msg += "\n  fresh | K=%3d | %.7e" % (targetSS.K, freshEvBound)
    msg += "\n  cur   | K=%3d | %.7e" % (curSS.K, curEvBound)
    raise BirthProposalError(msg)


  # Perform many merges among the fresh components
  for trial in xrange(5):
    if trial > 0:
      targetLP = freshModel.calc_local_params(targetData)
      targetSS = freshModel.get_global_suff_stats(targetData, targetLP,
                    doPrecompEntropy=True, doPrecompMergeEntropy=True)
    prevK = targetSS.K
    freshModel, targetSS, newEv, MTracker = MergeMove.run_many_merge_moves(
                               freshModel, targetData, targetSS,
                               nMergeTrials=targetSS.K**2, 
                               randstate=randstate, 
                               **mergeKwArgs)
    if targetSS.K == prevK:
      break # no merges happened
    else:
      Log.info("  **** Merged away %d fresh comps" % (prevK - targetSS.K))
  """
  # Step 3: create expanded model,
  #          and try merging fresh comps with existing ones
  origLP = curModel.calc_local_params(targetData)
  expandSS = curModel.get_global_suff_stats(targetData, origLP) 
  expandSS.insertComps(targetSS)
  expandModel = curModel.copy()
  expandModel.update_global_params(expandSS)

  compList = list(curModel.obsModel.K + np.arange(targetSS.K))
  expandLP = expandModel.calc_local_params(targetData)
  expandSS = expandModel.get_global_suff_stats(targetData, expandLP,
                  doPrecompEntropy=True, doPrecompMergeEntropy=True)
  prevK = expandSS.K
  expandModel, expandSS, Info = MergeMove.run_many_merge_moves(
                               expandModel, targetData, expandSS,
                               nMergeTrials=expandSS.K**2, 
                               compList=compList,
                               doExcludePairsInList=True,
                               randstate=randstate, 
                               **mergeKwArgs)

  for k in reversed(sorted(Info['acceptedIDs'])):
    ktarget = k - curModel.obsModel.K 
    if ktarget >= 0:
      targetSS.removeComp(ktarget)
  """

  return targetSS

########################################################### Select birth comps
###########################################################
def select_birth_component(SS, targetSelectName='sizebiased', 
                           randstate=np.random, emptyTHR=100,
                           lapsSinceLastBirth=defaultdict(int),
                           excludeList=list(), doVerbose=False, **kwargs):
  ''' Choose a single component among indices {1,2,3, ... K-1, K}
      to target with a birth proposal.
  '''
  K = SS.K
  if len(excludeList) >= K:
    raise BirthProposalError('BIRTH not possible. All possible K=%d targets used or excluded.' % (K))  
  ps = np.zeros(K)
  if targetSelectName == 'uniform':
    ps = np.ones(K)
  elif targetSelectName == 'sizebiased':
    ps = SS.N.copy()
    ps[SS.N < emptyTHR] = 0
  elif targetSelectName == 'delaybiased':
    # Bias choice towards components that have not been selected in a long time
    lapDist = np.asarray([lapsSinceLastBirth[kk] for kk in range(K)])
    ps = np.maximum(lapDist + 1e-5, 0)
    ps = ps * ps
  elif targetSelectName == 'delayandsizebiased':
    # Bias choice towards components that have not been selected in a long time
    #  *and* which have many members
    lapDist = np.asarray([lapsSinceLastBirth[kk] for kk in range(K)])
    ps = np.maximum(lapDist + 1e-5, 0)
    ps = ps * ps * SS.N
    ps[SS.N < emptyTHR] = 0
  else:
    raise NotImplementedError('Unrecognized procedure: ' + targetSelectName)
  # Make final selection at random
  ps[excludeList] = 0
  if np.sum(ps) < EPS:
    raise BirthProposalError('BIRTH not possible. All possible target comps have zero probability.')
  sortIDs = np.argsort(ps)[::-1]
  if doVerbose:
    for kk in sortIDs[:6]:
      print "comp %3d : %.2f prob | %3d delay | %8d size" % (kk, ps[kk]/sum(ps), lapsSinceLastBirth[kk], SS.N[kk])
  kbirth = discrete_single_draw(ps, randstate)
  return kbirth





###########################################################  Visualization
###########################################################
def viz_birth_proposal_2D(curModel, newModel, ktarget, freshCompIDs):
  ''' Create before/after visualization of a birth move (in 2D)
  '''
  from ..viz import GaussViz, BarsViz
  from matplotlib import pylab

  fig = pylab.figure()
  h1 = pylab.subplot(1,2,1)

  if curModel.obsModel.__class__.__name__.count('Gauss'):
    GaussViz.plotGauss2DFromHModel(curModel, compsToHighlight=ktarget)
  else:
    BarsViz.plotBarsFromHModel(curModel, compsToHighlight=ktarget, figH=h1)
  pylab.title( 'Before Birth' )
    
  h2 = pylab.subplot(1,2,2)
  if curModel.obsModel.__class__.__name__.count('Gauss'):
    GaussViz.plotGauss2DFromHModel(newModel, compsToHighlight=freshCompIDs)
  else:
    BarsViz.plotBarsFromHModel(newModel, compsToHighlight=freshCompIDs, figH=h2)
  pylab.title( 'After Birth' )
  pylab.show(block=False)
  try: 
    x = raw_input('Press any key to continue >>')
  except KeyboardInterrupt:
    import sys
    sys.exit(-1)
  pylab.close()

