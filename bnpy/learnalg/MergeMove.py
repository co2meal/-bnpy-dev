'''
MergeMove.py

Merge components of a bnpy model.

Usage
--------
Inside a LearnAlg, to try a merge move on a particular model and dataset.
>>> hmodel, SS, curEv, MoveInfo = run_merge_move(hmodel, Data, SS)

To force a merge of components "2" and "4"
>>> hmodel, SS, curEv, MoveInfo = run_merge_move(hmodel, Data, SS, kA=2, kB=4)
'''

import numpy as np
import logging
import os

from MergePairSelector import MergePairSelector
from MergeTracker import MergeTracker

Log = logging.getLogger('bnpy')
Log.setLevel(logging.DEBUG)

############################################################ Many Merge Moves
############################################################
def run_many_merge_moves(hmodel, Data, SS, evBound=None,
                               nMergeTrials=1, compList=list(), 
                               randstate=np.random, 
                              **mergeKwArgs):
  ''' Run (potentially many) merge move on hmodel

      Args
      -------
      hmodel
      Data
      SS
      nMergeTrials : number of merges to try
      compList : list of components to include in attempted merges
      randstate : random number generator

      Returns
      -------
      hmodel
      SS
      Info
  '''
  nMergeTrials = np.maximum(nMergeTrials, len(compList))

  MTracker = MergeTracker(SS.K)
  MSelector = MergePairSelector()

  if evBound is None:
    newEv = hmodel.calc_evidence(SS=SS)
  else:
    newEv = evBound

  trialID = 0
  while trialID < nMergeTrials and MTracker.hasAvailablePairs():
    oldEv = newEv  
        
    if len(compList) > 0:
      kA = compList.pop()
      if kA not in MTracker.getAvailableComps():
        continue
    else:
      kA = None

    hmodel, SS, newEv, MoveInfo = run_merge_move(
                 hmodel, Data, SS, oldEv, kA=kA, randstate=randstate,
                 MSelector=MSelector, MTracker=MTracker,
                 **mergeKwArgs)
    if MoveInfo['didAccept']:
      assert newEv > oldEv
    trialID += 1
    MTracker.recordResult(**MoveInfo)

  return hmodel, SS, newEv, MTracker

def run_merge_move(curModel, Data, SS=None, curEv=None, doVizMerge=False,
                   kA=None, kB=None, MTracker=None, MSelector=None,
                   mergename='marglik', randstate=np.random.RandomState(),
                   doUpdateAllComps=1, savedir=None, doVerbose=False, **kwargs):
  ''' Creates candidate model with two components merged,
      and returns either candidate or current model,
      whichever has higher log probability (ELBO).

      Args
      --------
       curModel : bnpy model whose components will be merged
       Data : bnpy Data object 
       SS : bnpy SuffStatDict object for Data under curModel
            must contain precomputed merge entropy in order to try a merge.
       curEv : current evidence bound, provided to save re-computation.
                curEv = curModel.calc_evidence(SS=SS)
       kA, kB : (optional) integer ids for which specific components to merge
       excludeList : (optional) list of integer ids excluded when selecting
                      which components to merge. useful when doing multiple 
                      rounds of merges, since precomputed merge terms are 
                      valid for one merge only.
      Returns
      --------
      hmodel, SS, evBound, MoveInfo

      hmodel := candidate or current model (bnpy HModel object)
      SS := suff stats for Data under hmodel
      evBound := log evidence (ELBO) of Data under hmodel
      MoveInfo := dict of info about this merge move, with fields
            didAccept := boolean flag, true if candidate accepted
            msg := human-readable string about this move
            kA, kB := indices of the components to be merged.
  ''' 
  if SS is None:
    LP = curModel.calc_local_params(Data)
    SS = curModel.get_global_suff_stats(Data, LP,
                                        doPrecompEntropy=True,
                                        doPrecompMerge=True)
  if curEv is None:
    curEv = curModel.calc_evidence(SS=SS)
  if MTracker is None:
    MTracker = MergeTracker(SS.K)
  if MSelector is None:
    MSelector = MergePairSelector()

  # Need at least two components to merge!
  if curModel.allocModel.K == 1:
    MoveInfo = dict(didAccept=0, msg="need >= 2 comps to merge")    
    return curModel, SS, curEv, MoveInfo  
  
  if not SS.hasMergeTerms():
    MoveInfo = dict(didAccept=0, msg="suff stats did not have merge terms")    
    return curModel, SS, curEv, MoveInfo  

  if kA is not None and kA not in MTracker.getAvailableComps():
    MoveInfo = dict(didAccept=0, msg="target comp kA must be excluded.")    
    return curModel, SS, curEv, MoveInfo  
    
  # Select which 2 components kA, kB in {1, 2, ... K} to merge
  if kA is None or kB is None:
    kA, kB = select_merge_components(curModel, Data, SS,
                                     kA=kA, MTracker=MTracker,
                                     MSelector=MSelector,
                                     mergename=mergename, 
                                     randstate=randstate)
  if doVerbose:
    print "  merging %3d+%3d" % (kA, kB)
  # Create candidate merged model
  propModel, propSS = propose_merge_candidate(curModel, SS, kA, kB, doUpdateAllComps=doUpdateAllComps)

  # Decide whether to accept the merge
  propEv = propModel.calc_evidence(SS=propSS)

  if np.isnan(propEv) or np.isinf(propEv):
    raise ValueError('propEv should never be nan/inf')
    
  if doVizMerge:
    viz_merge_proposal(curModel, propModel, kA, kB, curEv, propEv)

  evDiff = propEv - curEv
  if doVerbose:
    s = ''
    if evDiff > 0:
      if propEv < 0:
        s = '***'
      else:
        s = '!!!!!!!!!!!!!!!!'
    print "    new ev %.3e | diff %.3e |  %s" % (propEv, evDiff, s)

  if propEv > 0 and curEv < 0:
    MoveInfo = dict(didAccept=0, kA=kA, kB=kB, msg="CRAP. bad proposed evidence.")
    return curModel, SS, curEv, MoveInfo
  if propEv > curEv:
    MSelector.reindexAfterMerge(kA, kB)
    msg = "merge %3d & %3d | ev +%.3e ****" % (kA, kB, propEv - curEv)
    MoveInfo = dict(didAccept=1, kA=kA, kB=kB, msg=msg, evDiff=evDiff)
    log_merge_move(MoveInfo, MSelector, curModel, SS, savedir)
    return propModel, propSS, propEv, MoveInfo
  else:
    msg = "merge %3d & %3d | ev -%.3e" % (kA, kB, curEv - propEv)
    MoveInfo = dict(didAccept=0, kA=kA, kB=kB, msg=msg, evDiff=evDiff)
    log_merge_move(MoveInfo, MSelector, curModel, SS, savedir)
    return curModel, SS, curEv, MoveInfo

########################################################## Log info to file
##########################################################
def log_merge_move(MoveInfo, MSelector, hmodel, SS, savedir):
  if 'kA' not in MoveInfo or savedir is None:
    return
  assert os.path.exists(savedir)
  headerfile = os.path.join(savedir, 'mergelog_marglik_header.csv')
  savefile = os.path.join(savedir, 'mergelog_marglik.csv')
  if not os.path.exists(savefile):
    headerstring = "didAccept,evDiff,"
    headerstring += "logmRatio,logmBoth,logmA,logmB,"
    headerstring += "K,NA,NB,fracA,fracB"
    with open(headerfile, 'w') as f:
      f.write(headerstring + '\n')

  kA = MoveInfo['kA']
  kB = MoveInfo['kB']
  mA = MSelector._calcLogMargLikForComp(hmodel, SS, kA)
  mB = MSelector._calcLogMargLikForComp(hmodel, SS, kB)
  mBoth = MSelector._calcLogMargLikForPair(hmodel, SS, kA, kB)
  mRatio = mBoth - mA - mB
  NA = SS.N[kA]
  NB = SS.N[kB]
  fracA = NA/np.sum(SS.N)
  fracB = NB/np.sum(SS.N)
  csvstring = "%d,%.3e," % (MoveInfo['didAccept'], MoveInfo['evDiff'])
  csvstring += "%.3e,%.3e,%.3e,%.3e," % (mRatio, mBoth, mA, mB)
  csvstring += "%d,%d,%d,%.3f,%.3f" % (SS.K, NA, NB, fracA, fracB)
  with open(savefile, 'a') as f:
    f.write( csvstring + '\n')

########################################################## Select kA,kB to merge
##########################################################
def select_merge_components(curModel, Data, SS, MTracker=None,
                            MSelector=None,
                            mergename='marglik', randstate=None,
                            kA=None, **kwargs):
  ''' Select which two existing components to merge when constructing
      a candidate "merged" model from curModel, which has K components.
      We select components kA, kB by their integer ID, in {1, 2, ... K}

      Args
      --------
      curModel : bnpy model whose components we should merge
      Data : data object 
      SS : suff stats object for Data under curModel
      LP : local params dictionary (not required except for 'overlap')
      mergename : string specifying routine for how to select kA, kB
                  options include
                  'random' : select comps at random, without using data.
                  'marglik' : select comps by marginal likelihood ratio.
      Returns
      --------
      kA : integer id of the first component to merge
      kB : integer id of the 2nd component to merge

      This method guarantees that kA < kB.
  '''
  if MTracker is None:
    MTracker = MergeTracker(SS.K)
  if MSelector is None:
    MSelector = MergePairSelector()
  kA, kB = MSelector.select_merge_components(curModel, SS, MTracker,
                                    mergename=mergename, 
                                    kA=kA, randstate=randstate)
  return kA, kB

############################################################ Construct new model
############################################################
def propose_merge_candidate(curModel, SS, kA=None, kB=None, 
                             doUpdateAllComps=1):
  ''' Propose new bnpy model from the provided current model (with K comps),
      where components kA, kB are combined into one "merged" component

      Args
      --------
      curModel := bnpy HModel object
      SS := bnpy SuffStatBag object
      kA := integer id of comp to merge (will be index of kA+kB in new model)
      kB := integer id of comp to merge (will be removed from new model)
      doUpdateAllComps := integer flag. if 1, all K obsModel global params updated. if 0, only the relevant component (kA) is updated. always, all allocModel comps are updated.

      Returns
      --------
      propModel := bnpy HModel object
      propSS := bnpy sufficient statistic object.

      Both propSS and propModel have K-1 components.
  '''
  propModel = curModel.copy()
  
  # Rewrite candidate's kA component to be the merger of kA+kB
  # For now, **all* components get updated.
  # TODO: smartly avoid updating obsModel comps except related to kA/kB
  propSS = SS.copy()
  propSS.mergeComps(kA, kB)
  assert propSS.K == SS.K - 1
  if doUpdateAllComps:
    propModel.update_global_params(propSS)
  else:
    propModel.update_global_params(propSS, mergeCompA=kA, mergeCompB=kB)

  # Remember, after calling update_global_params
  #  propModel's components must exactly match propSS's.
  # So kB has effectively been deleted here. It is already gone.
  return propModel, propSS


############################################################ Visualization
############################################################
def viz_merge_proposal(curModel, propModel, kA, kB, curEv, propEv):
  ''' Visualize merge proposal (in 2D)
  '''
  from ..viz import GaussViz, BarsViz
  from matplotlib import pylab
  
  fig = pylab.figure()
  h1 = pylab.subplot(1,2,1)
  if curModel.obsModel.__class__.__name__.count('Gauss'):
    GaussViz.plotGauss2DFromHModel(curModel, compsToHighlight=[kA, kB])
  else:
    BarsViz.plotBarsFromHModel(curModel, compsToHighlight=[kA, kB], figH=h1)
  pylab.title( 'Before Merge' )
  pylab.xlabel( 'ELBO=  %.2e' % (curEv) )
    
  h2 = pylab.subplot(1,2,2)
  if curModel.obsModel.__class__.__name__.count('Gauss'):
    GaussViz.plotGauss2DFromHModel(propModel, compsToHighlight=[kA])
  else:
    BarsViz.plotBarsFromHModel(propModel, compsToHighlight=[kA], figH=h2)
  pylab.title( 'After Merge' )
  pylab.xlabel( 'ELBO=  %.2e \n %d' % (propEv, propEv > curEv))
  pylab.show(block=False)
  try: 
    x = raw_input('Press any key to continue / Ctrl-C to quit >>')
  except KeyboardInterrupt:
    import sys
    sys.exit(-1)
  pylab.close()

