'''
  Fast birth move.

'''
from IPython import embed
from collections import defaultdict
import numpy as np
import copy
import time

from ..util import discrete_single_draw, np2flatstr
from .GibbsSamplerAlg import GibbsSamplerAlg
from .VBLearnAlg import VBLearnAlg

class ProposalError( ValueError):
  def __init__( self, *args, **kwargs):
    super(type(self), self).__init__( *args, **kwargs )

'''  NOTES ON SETTING BIRTH PARAMETERS
      --------------------------------
      splitTHR should tend to be small, but may need to be stochastic [look at the ManualInspectBirth script for an example]
      doDeleteExtra should probably always be on.
      do NOT do the "forward lookahead" computation of accept/reject.  it's much too strict. Compare nova+K4+THR0.1+pp50+JustRefine1 to nova+K4+THR0.1+pp50+Refine1Fwd
      DO set nPropIter quite large for better proposals, especially with random initialization or plusplus
      DO set nRefineIter=1 *always*  if it is too large, we can accept births just because pre-existing features are improving the bound
'''
SInfoDEF = dict( Kextra=8, splitTHR=0.1, doDeleteExtra=True, nPropIter=1, nIterInit=1, splitpropname='seq+batchVB', nRefineIter=1, nTrial=1)

def target_subsample_data( Data, LP, kbirth, nObsMax=10000, splitTHR=None, PRNG=np.random):
  origResp =  LP['resp'][:,kbirth]
  rawIDs = np.flatnonzero( origResp > splitTHR )
  Xchunk   =  Data['X'][ rawIDs ]
  nChunk   = Xchunk.shape[0]
  D = Xchunk.shape[1]
  if nChunk < 1:
    raise ProposalError( 'HALT. Not enough data to make a good subsample! nChunk=%d'%(nChunk) )
  nObsMax = np.minimum( nObsMax, nChunk)
  allIDs = PRNG.permutation( nChunk ).tolist()
  selectIDs = allIDs[:nObsMax]
  Xselect = Xchunk[selectIDs]
  return Xselect, dict( selectIDs=rawIDs[allIDs] )

def select_birth_component( K, SS, iterid, MoveLog, PRNG=np.random, emptyTHR=100, excludeList=None):
  ''' Choose a single component from hmodel to attempt to split
  '''
  if 'lasttryiter' not in MoveLog:
    MoveLog['lasttryiter'] = defaultdict(lambda: -100)
  ps = np.zeros( K)
  for kk in xrange( K):
    lastiter = MoveLog['lasttryiter'][kk]
    ps[kk] = 1e-5 + np.abs(iterid - lastiter)
  if 'N' in SS:
    ps = ps**3 * SS['N']    # make distribution much more peaky, and use component sizes to "break ties"
    rareIDs =  SS['N'] < emptyTHR
    if np.sum( rareIDs) == K:
      print '*********************************************************'
      print '                  WARNING:  emptyTHR too HIGH for selecting birth component. No component eligible!'
      print '*********************************************************'
    else:
      ps[ rareIDs] = 0
  else:
    ps = ps**3
  if excludeList is not None:
    ps[excludeList] = 0
  print ps
  if ps.sum() == 0:
    raise ProposalError( 'HALT. Bad ps calculation') # extremely unlikely!
  return discrete_single_draw( ps, PRNG ), dict(ps=ps)


def run_targetbirth_move( curmodel, SS, Dtarget, kbirth, MoveLog=dict(), iterid=0, seed=42, doViz=False, SInfo=SInfoDEF):
  ''' Input

      Returns
      -------
      hmodel : new (or current) HModel, if proposal is accepted (rejected)
      SS
      SSextra  (can be None)
      MoveInfo : dict of information about this proposal
      MoveLog  : dict of info about all accepted proposals
  '''
  for key in SInfoDEF:
    if key not in SInfo:
      SInfo[key] = SInfoDEF[key]

  if 'lasttryiter' not in MoveLog:
    MoveLog['lasttryiter'] = defaultdict(lambda: -100)
  if 'Hz' not in SS:
    raise ValueError( 'Need whole data suff stats SS to have entropy precalculated')

  try:  
    print '>>>>>>>-----------  BIRTH move on kbirth = %3d' % (kbirth)
    print 'Nsample %d' % (Dtarget['nObs'])
    MoveLog['lasttryiter'][kbirth] = iterid

    # Create candidate model for targeted dataset
    SSextra, MoveInfo = train_fresh_model_on_subsample( curmodel, Dtarget, Ntotal=SS['N'][kbirth], seed=seed, Mflag=False, **SInfo)
    #newSS = create_combined_suff_stats( SS, SSextra, MoveInfo['Kextra'], doSkip=True )
    newSS = get_expanded_suff_stats( SS, MoveInfo['Kextra'])

    candidate = curmodel.copy()
    candidate.reset_K( curmodel.K + MoveInfo['Kextra'] )
    candidate.update_global_params( newSS )

  except ProposalError as e:
    print e.message
    MoveInfo = dict(didAccept=0)
    return curmodel, SS, MoveLog, MoveInfo

  ###################################################################   Decide whether to Accept/Reject
  if doViz:
    vc = curmodel.copy()
    vc.reset_K( curmodel.K + MoveInfo['Kextra'] )
    vc.update_global_params(  create_combined_suff_stats( SS, SSextra, MoveInfo['Kextra'], doSkip=True )  )
    MoveInfo['Dchunk'] = Dtarget
    MoveInfo['kbirth']=kbirth
    userResp = viz_birth_proposal( curmodel, vc, MoveInfo, -1, -1)
    if userResp == 'p':
      embed()
 
  ############################################################################  Return appropriate stuff
  '''
  origEv = curmodel.calc_evidence( Dtarget )
  newEv  = candidate.calc_evidence( Dtarget )
  print '  origEv %.5e' % (origEv)
  print '  newEv  %.5e' % (newEv)
  '''
  doAccept = True
  if doAccept:
    MoveInfo['msg'] = 'birth'
    MoveInfo['didAccept'] = 1
    MoveInfo['Ktotal'] = candidate.K
    MoveInfo['kbirth'] = kbirth
    SSbirth = expand_suff_stats_for_birth( SSextra, curmodel.K)
    MoveLog['SSbirth'] = SSbirth

    for kk in MoveInfo['newIDs']:
      MoveLog['lasttryiter'][kk] = iterid
    MoveLog['lasttryiter'][kbirth] = iterid    
    MoveLog['AInfo'].append( MoveInfo )
    MoveLog['AIters'].append( iterid )
            
    return candidate, newSS, MoveLog, MoveInfo
  else:
    MoveInfo['didAccept'] = 0
  return curmodel, SS, MoveLog, MoveInfo


###################################################################
###################################################################  Fresh Model Creation
###################################################################  subsample targeted data from minibatch, train fresh model with Kextra new components, output suff stats
def train_fresh_model_on_subsample( hmodel, Dsub, seed=0, Ntotal=None, nPropIter=SInfoDEF['nPropIter'], splitpropname=SInfoDEF['splitpropname'], Kextra=SInfoDEF['Kextra'], doDeleteExtra=SInfoDEF['doDeleteExtra'], Mflag=False, nIterInit=SInfoDEF['nIterInit'], nTrial=SInfoDEF['nTrial'], **kwargs ):
  '''
     Given subsampled data, train a completely fresh model with (at most) Kextra components.  
     Returns
     -------
     SSsub : Suff statistics for this new model, with at most Kextra components
  '''
  PRNG = np.random.RandomState(seed)
  Kmax = 0
  if Kextra < 0:
    Kmax = int(np.abs(Kextra ))
  
  newmodel = hmodel.copy()
  newmodel.set_qType( 'VB' ) # make sure to do batch-style updates!

  for trial in range( nTrial):
    seed = seed+trial

    newmodel.reset_K( Kextra )
    if Kmax > 0:
      Kextra = PRNG.randint( 3, Kmax)
      newmodel.reset_K( Kextra )
    
    if splitpropname == 'pp+batchVB':
      infer = VBLearnAlg( printEvery=-1, saveEvery=-1, nIter=nPropIter,initname='plusplus')
      infer.init_global_params( newmodel, Dsub, seed=seed )
      LPsub = infer.fit( newmodel, Dsub )
    elif splitpropname == 'stick+batchVB':
      infer = VBLearnAlg( printEvery=-1, saveEvery=-1, nIter=nPropIter,initname='randpartstick')
      alph = PRNG.uniform( hmodel.allocModel.alpha0, 10*hmodel.allocModel.alpha0 )
      infer.init_global_params( newmodel, Dsub, seed=seed, alpha0=alph )
      LPsub = infer.fit( newmodel, Dsub )
    elif splitpropname == 'rp+batchVB':
      infer = VBLearnAlg( printEvery=-1, saveEvery=-1, nIter=nPropIter,initname='randpart')
      infer.init_global_params( newmodel, Dsub, seed=seed )
      LPsub = infer.fit( newmodel, Dsub )
    elif splitpropname == 'rpdiverse+batchVB':
      infer = VBLearnAlg( printEvery=-1, saveEvery=-1, nIter=nPropIter,initname='randpartdiverse')
      infer.init_global_params( newmodel, Dsub, seed=seed )
      LPsub = infer.fit( newmodel, Dsub )
    elif splitpropname == 'rx+batchVB':
      infer = VBLearnAlg( printEvery=-1, saveEvery=-1, nIter=nPropIter,initname='randexamples')
      infer.init_global_params( newmodel, Dsub, seed=seed )
      LPsub = infer.fit( newmodel, Dsub )
    elif splitpropname == 'rs+batchVB':
      infer = VBLearnAlg( printEvery=-1, saveEvery=-1, nIter=nPropIter,initname='randsample')
      infer.init_global_params( newmodel, Dsub, seed=seed )
      LPsub = infer.fit( newmodel, Dsub )
    elif splitpropname == 'seq+batchVB':
      print '---------------------------------------------------------- one pass of sequential gibbs'
      infer = VBLearnAlg( printEvery=-1, saveEvery=-1, nIter=nPropIter, initname='seqCGS')
      infer.init_global_params( newmodel, Dsub, seed=seed, nIterInit=nIterInit, printEvery=-1, Kmax=Kextra )
      LPsub = infer.fit( newmodel, Dsub )
      newmodel.set_qType( 'VB' )
    elif splitpropname == 'pca+batchVB':
      infer = VBLearnAlg( printEvery=-1, saveEvery=-1, nIter=nPropIter, initname='pca')
      infer.init_global_params( newmodel, Dsub, seed=seed )
      LPsub = infer.fit( newmodel, Dsub )
      newmodel.set_qType( 'VB' )
    elif splitpropname == 'seqCGS':
      infer = VBLearnAlg( printEvery=-1, saveEvery=-1, nIter=1, initname='seqCGS')
      infer.init_global_params( newmodel, Dsub, seed=seed, nIterInit=2, printEvery=-1 )
      LPsub = infer.fit( newmodel, Dsub )
      newmodel.set_qType( 'VB' )
    else:
      raise ValueError('splitpropname %s not recognised.' % (splitpropname) )
    if doDeleteExtra:
      SSsub = newmodel.get_global_suff_stats( Dsub, LPsub)
      emptyTHR=np.maximum( 100,0.05*Dsub['nObs'] )
      newmodel = delete_empty_extra_components( newmodel, SSsub, emptyTHR=emptyTHR )
      print '---------------------------------------------------------- after %d iters of batch vb | trial %d | Kextra %d' % (nPropIter, trial, Kextra)
      if newmodel.K >= 2:
        break
  if newmodel.K < 2:
    raise ProposalError( 'HALT. Failed to create more than 1 component with more than %d members' % (emptyTHR) )
  LPsub = newmodel.calc_local_params( Dsub )
  #newmodel, SSsub, LPsub, ev, Info = run_delete_move( newmodel, Dsub, LP=LPsub, emptyFRAC=0.05, doDecide=False)
  Kextra = newmodel.K
  SSsub = newmodel.get_global_suff_stats( Dsub, LPsub, Ntotal=Ntotal, Eflag=True, Mflag=Mflag)
  del SSsub['ampF']
  return SSsub, dict( Kextra=Kextra, newIDs=hmodel.K+np.arange(Kextra)  )
  
###################################################################
###################################################################  Visualization
###################################################################
def viz_birth_proposal( curmodel, newmodel, infoDict, origEv, newEv):
  if curmodel.obsModel.D == 2:
    viz_birth_proposal_2D( curmodel, newmodel, infoDict, origEv, newEv)
    return None
  from matplotlib import pylab
  #fig = pylab.figure()
  nRow=2
  nCol=8
  fig, axes = pylab.subplots(nrows=nRow, ncols=nCol)
  fig.tight_layout()
  
  pylab.hold('on')
  kbirth=infoDict['kbirth']
  oldids = range( np.maximum(0,kbirth-nCol+1), kbirth+1)
  remK = nCol-len(oldids)
  if remK > 0:
    oldids.extend( range(kbirth+1, np.minimum(curmodel.K, kbirth+remK)) )

  for nn in range( len(oldids) ):  
    hA = pylab.subplot( nRow, nCol, nn+1 )
    pylab.imshow( curmodel.obsModel.get_covar_mat(oldids[nn]), interpolation='nearest', vmin=-2, vmax=2 )
    pylab.ylabel( '%.2f'%( np.exp(curmodel.allocModel.Elogw[ oldids[nn] ]))  )
    pylab.xticks([])
    pylab.yticks([])
    if oldids[nn] == kbirth:
      pylab.title( 'Before Birth' )
      pylab.xlabel( 'ELBO=  %.5e' % (origEv) )

  for nn in range( len(oldids)+1, nCol+1):
    pylab.subplot( nRow, nCol, nn)
    pylab.imshow( 0*curmodel.obsModel.get_covar_mat(0), interpolation='nearest' )
    pylab.xticks([])
    pylab.yticks([])

  '''hA = pylab.subplot( nRow, nCol, 1)
  pylab.hold('on')
  pylab.imshow( curmodel.obsModel.get_covar_mat( ), interpolation='nearest' )  
  pylab.title( 'Before Birth' )
  pylab.xlabel( 'ELBO=  %.5e' % (origEv) )
  '''
  newIDs = infoDict['newIDs']
  for nn in range( np.minimum( len(newIDs), nCol) ):  
    hB=pylab.subplot( nRow, nCol, nn+nCol+1 )
    pylab.imshow( newmodel.obsModel.get_covar_mat(newIDs[nn]), interpolation='nearest', vmin=-2, vmax=2 )
    pylab.ylabel( '%.2f'%( np.exp(newmodel.allocModel.Elogw[ newIDs[nn] ]))  )
    pylab.xticks([])
    pylab.yticks([])
  pylab.title( 'After Birth' )
  pylab.xlabel( 'ELBO=  %.5e \n %d' % (newEv, newEv > origEv) )

  xtraids=range( len(newIDs)+1, nCol+1)
  for nn in xtraids:
    pylab.subplot( nRow, nCol, nCol+nn)
    pylab.imshow( 0*curmodel.obsModel.get_covar_mat(0), interpolation='nearest' )
    pylab.xticks([])
    pylab.yticks([])

  pylab.show(block=False)
  fig.canvas.draw()
  
  x = None
  try: 
    x = raw_input('Press any key to continue >>')
    print '>>>>>>>>>>>>>>>>>>>>>Raw Input:', x
  except KeyboardInterrupt:
    doViz = False
  pylab.close()
  return x

def viz_cov_mats( hmodel ):
  from matplotlib import pylab
  pylab.figure()
  pID=0
  for kk in range( hmodel.K):
    pID+=1
    pylab.subplot( 1, hmodel.K, pID)
    pylab.imshow( hmodel.obsModel.get_covar_mat(kk), interpolation='nearest', vmin=-0.25, vmax=2 )
    pylab.xticks( [] )
    pylab.yticks( [] )
  pylab.draw()

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
  GaussViz.plotGauss2DFromModel( newmodel, Hrange=newIDs )
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

###################################################################
###################################################################  Suff Stat Utility Calculations
###################################################################
def delete_empty_extra_components( hmodel, SS, emptyTHR, Krange=None ):
  ''' Produces a NEW hmodel
        with some components removed.
  '''
  if Krange is None:
    Krange = range(hmodel.K)
  Korig = hmodel.K
  delIDs = [kk for kk in Krange if SS['N'][kk] < emptyTHR ]
  keepIDs  = list( set( xrange(hmodel.K) ).difference( delIDs ) )
  if len(keepIDs) == hmodel.K:
    return hmodel
  hmodel.allocModel.delete_components( keepIDs)  
  hmodel.obsModel.delete_components( keepIDs)
  hmodel.K = len(keepIDs)
  return hmodel

def inc_suff_stats_from_chunk( SS, SSchunk):
  for key in SS:
    if type( SS[key] ) is not np.ndarray and type( SS[key] ) is not np.float64:
      continue
    if SS[key].size >= 1:
      SS[key] += SSchunk[key]

def dec_suff_stats_from_chunk( SS, SSchunk):
  for key in SS:
    if type( SS[key] ) is not np.ndarray and type( SS[key] ) is not np.float64:
      continue
    if SS[key].size >= 1:
      SS[key] -= SSchunk[key]

def remove_suff_stat_mass_from_comp( SS, kk, remFrac=None):
  for key in SS:
    if key == 'Hmerge':
      continue
    if type( SS[key] ) is np.ndarray:
      if remFrac is not None:
        SS[key][kk] = remFrac*SS[key][kk]
  return SS  
  
def get_expanded_suff_stats( SS, Kextra):
  SS = copy.deepcopy( SS )
  for key in SS:
    if type( SS[key] ) is np.ndarray:
      DimVec = SS[key].shape
      if key == 'Hmerge':
        SS[key] = np.vstack( [ SS[key], np.zeros((Kextra, DimVec[1])) ] )
        SS[key] = np.hstack( [ SS[key], np.zeros( (Kextra+DimVec[1], Kextra))])
      elif SS[key].ndim == 1:
        SS[key] = np.hstack( [ SS[key], np.zeros(Kextra)] )
      elif SS[key].ndim == 2:
        SS[key] = np.vstack( [ SS[key], np.zeros((Kextra, DimVec[1])) ] )
      elif SS[key].ndim == 3:
        SS[key] = np.vstack( [ SS[key], np.zeros((Kextra, DimVec[1], DimVec[2]))] )
  return SS 

def expand_suff_stats_for_birth( SSbirth, K):
  '''  Return suff stats with K +Kbirth components,
          where only the last Kbirth components are non-zero
  '''
  SSall = dict()
  Kextra = len( SSbirth['N'] )
  Knew = K + Kextra
  for key in SSbirth:
    if type( SSbirth[key]) is np.float64:
      SSall[key] = SSbirth[key]
    if type( SSbirth[key]) is np.ndarray:
      if SSbirth[key].ndim == 1:
        SSall[key] = np.zeros( Knew )
      if SSbirth[key].ndim == 2:
        SSall[key] = np.zeros( (Knew, SSbirth[key].shape[1]) )
      if SSbirth[key].ndim == 3:
        SSall[key] = np.zeros( (Knew, SSbirth[key].shape[1], SSbirth[key].shape[2]) )
      SSall[key][ K: ] = SSbirth[key]
  return SSall

def create_combined_suff_stats( SS, SSextra, Kextra, doSkip=False):
  '''  Create combined sufficient stat structure
      Returns
      --------
      SSall : suff stat dictionary with K + Kextra components
  '''
  SSall = get_expanded_suff_stats( SS, Kextra) 
  
  for key in SSall:
    if not doSkip and key == 'Hmerge':
      SSall['Hmerge'][ -Kextra:, -Kextra:] = SSextra[key]
    if doSkip and key == 'Hz':
      continue
    if doSkip and key == 'Hmerge':
      continue
    elif type( SSall[key] ) is np.ndarray and SSall[key].size > 1:
      SSall[key][ -Kextra: ] = SSextra[key]
  return SSall

