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
                  maxTargetObs=100, randstate=np.random, **kwargs):
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
    mask = LP['word_variational'][: , targetCompID] > targetProbThr
  else:
    mask = LP['resp'][: , targetCompID] > targetProbThr
  objIDs = np.flatnonzero(mask)
  if 'word_variational' in LP:
    # TODO: better control for the size of the subset??
    TargetData = DataObj.select_subset_by_mask(wordMask=objIDs,
                                                doTrackFullSize=False)
  else:
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
                              randstate=randstate, **kwargs)
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
                      nFreshLap=50, randstate=np.random, **kwargs):
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
  
  Nthr = np.maximum(100, 0.05 * targetData.nObs)
  rejectIDs = np.flatnonzero(targetSS.N < Nthr)
  rejectIDs = np.sort(rejectIDs)[::-1]
  for kreject in rejectIDs:
    targetSS.removeComp(kreject)
    
  if targetSS.K < 2:
    raise BirthProposalError( 'BIRTH: Did not create more than one comp of size %d from Data of size %d' % (Nthr, targetData.nObs) )
  return targetSS
  
  
###########################################################
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
    raise BirthProposalError('All comps excluded. Selection failed.')
  
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
    raise BirthProposalError('All comps have zero probability. Selection failed.');
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

