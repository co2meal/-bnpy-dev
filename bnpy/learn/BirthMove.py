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
                  nTargetObs=100, randstate=np.random, **kwargs):
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
  mask = LP['resp'][: , targetCompID] > targetProbThr
  objIDs = np.flatnonzero(mask)
  randstate.shuffle(objIDs)
  targetObjIDs = objIDs[:nTargetObs]
  TargetData = DataObj.select_subset_by_mask(targetObjIDs)
  return TargetData
  
###########################################################
###########################################################
def run_birth_move(curModel, targetData, SS, randstate=np.random, **kwargs):
  ''' Create new model from curModel
        with up to Kbirth new components
  '''
  Log.debug("nObs: %d" % (targetData.nObs))  
  try:
    freshModel = curModel.copy()
    freshSS = learn_fresh_model(freshModel, targetData, 
                              randstate=randstate, **kwargs)
    Kfresh = freshSS.K
    Kold = curModel.obsModel.K
    newSS = SS.copy()
    newSS.insertComponents(freshSS)
  
    newModel = curModel.copy()
    newModel.update_global_params(newSS)
    MoveInfo = dict(msg='newModel has %d fresh comps' % (Kfresh),
                    birthCompIDs=range(Kold, Kold+Kfresh),
                    freshSS=freshSS)
    return newModel, newSS, MoveInfo
  except BirthProposalError, e:
    MoveInfo = dict(msg=str(e),
                    birthCompIDs=[])
    return curModel, SS, MoveInfo

def learn_fresh_model(freshModel, targetData, Kfresh=10,
                      freshInitName='randexamples', freshAlgName='VB',
                      nFreshLap=50, randstate=np.random, **kwargs):
  ''' Learn a new model with Kfresh components
      Returns
      -------
      freshSS : bnpy SuffStatDict with Kfresh components
  '''
  seed = randstate.choice(xrange(100000))
  freshModel.init_global_params(targetData, K=Kfresh, 
                                seed=seed, initname=freshInitName)
 
  LearnAlgConstructor = dict()
  LearnAlgConstructor['VB'] = VBLearnAlg
  algP = dict(nLap=nFreshLap, convergeTHR=1e-6)
  outP = dict(saveEvery=-1, traceEvery=-1, printEvery=-1)
  learnAlg = LearnAlgConstructor[freshAlgName](savedir=None, 
                    algParams=algP, outputParams=outP, seed=seed)

  targetLP, evBound = learnAlg.fit(freshModel, targetData)
  targetSS = freshModel.get_global_suff_stats(targetData, targetLP)
  
  Nthr = np.maximum(100, 0.05 * targetData.nObs)
  rejectIDs = np.flatnonzero(targetSS.N < Nthr)
  rejectIDs = np.sort(rejectIDs)[::-1]
  for kreject in rejectIDs:
    targetSS.removeComponent(kreject)
    
  if targetSS.K < 2:
    raise BirthProposalError( 'HALT. Failed to create more than 1 component with more than %d members' % (Nthr) )
  #TODO: need to precompute entropy here???
  return targetSS
  
  
###########################################################
###########################################################
def select_birth_component(SS, targetselectname='sizebiased', 
                           randstate=np.random, emptyTHR=100, **kwargs):
  ''' Choose a single component among indices {1,2,3, ... K-1, K}
      to target with a birth proposal.
  '''
  K = SS.K
  ps = np.zeros(K)
  if targetselectname == 'uniform':
    ps = np.ones(K)
  elif targetselectname == 'sizebiased':
    ps = SS['N']
    ps[SS['N'] < emptyTHR] = 0
  if ps.sum() < EPS:
    ps = np.ones(K)
  kbirth = discrete_single_draw(ps, randstate)
  return kbirth





###########################################################  Visualization
###########################################################
def viz_birth_proposal_2D( curmodel, newmodel, infoDict, origEv, newEv):
  ''' Create before/after visualization of a split proposal
  '''
  from ..viz import GaussViz
  from matplotlib import pylab
  
  X = infoDict['Dchunk']['X']
  fig = pylab.figure()
  
  hA = pylab.subplot( 1, 2, 1)
  pylab.hold('on')
  GaussViz.plotGauss2DFromModel( curmodel, Hrange=[infoDict['kbirth']] )
  pylab.plot( X[:2000,0], X[:2000,1], 'k.')
  pylab.title( 'Before Birth' )
  if origEv != -1:
    pylab.xlabel( 'ELBO=  %.5e' % (origEv) )
  
  hB=pylab.subplot(1,2,2)
  pylab.hold('on')
  newIDs = infoDict['newIDs'].tolist()
  newIDs.append( infoDict['kbirth'] )
  GaussViz.plotGauss2DFromHModel( newmodel, Hrange=newIDs )
  pylab.plot( X[:1000,0], X[:1000,1], 'k.')
  pylab.title( 'After Birth' )
  if newEv != -1:
    pylab.xlabel( 'ELBO=  %.5e \n %d' % (newEv, newEv > origEv) )
  pylab.show(block=False)
  fig.canvas.draw()
  
  try: 
    x = raw_input('Press any key to continue >>')
  except KeyboardInterrupt:
    doViz = False
  pylab.close()


