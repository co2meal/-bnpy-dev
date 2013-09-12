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

def run_fastbirth_move( curmodel, SS, Dchunk, SSchunk, LPchunk, origEv=None, MoveLog=dict(), iterid=0, seed=42, acceptname='chunk', SInfo=SInfoDEF, kbirth=None, doViz=False, doForward=False, doAmp=False, doInflate=False):
  ''' Input
      -------
      SS : suff stats *including the current chunk*!!

      Returns
      -------
      hmodel : new (or current) HModel, if proposal is accepted (rejected)
      SS 
      SSchunk
      LPchunk
      evBound   
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
  if origEv is None:
    origEv = curmodel.calc_evidence( SS=SS )

  assert LPchunk['resp'].shape[1] == curmodel.K

  try:  
    print '>>>>>>>----------- Attempting BIRTH move on batchID= %d' % (Dchunk['bID'] )

    # Select which component to alter by ID
    if kbirth is None:
      kbirth, SelectInfo = select_birth_component( curmodel, Dchunk, SSchunk, LPchunk, MoveLog, iterid=iterid, splitTHR=SInfo['splitTHR'], seed=seed)
      kvec = np.asarray([k for k in range(curmodel.K)])
      tryvec = np.asarray( [MoveLog['lasttryiter'][kk] for kk in range(curmodel.K)] )
      printIDs = np.arange( 0, curmodel.K)
      if curmodel.K > 10:
        printIDs = np.arange( kbirth-5, kbirth+5)
        printIDs = printIDs[ printIDs >= 0]
        printIDs = printIDs[ printIDs < curmodel.K ]
        kbuffer = np.sum( printIDs < kbirth)
      else:
        kbuffer = 1*kbirth
      print '   candidates ', np2flatstr( kvec[printIDs], '%4d' )
      print '     last try ', np2flatstr( tryvec[printIDs], '%4d')
      if 'ps' in SelectInfo:
        pvec = SelectInfo['ps']/np.sum(SelectInfo['ps'])
        print '         prob ', np2flatstr( pvec[printIDs], '%.2f')

      print '  kbirth=%3d'%(kbirth), '     '*(kbuffer), '  *^*'
    else:
      print ' PREDEFINED kbirth= %3d' % (kbirth)

    MoveLog['lasttryiter'][kbirth] = iterid

    # Create candidate model for current *chunk*
    remSSchunk, freshSSchunk, PropInfo = propose_fresh_suff_stats( curmodel, Dchunk, SSchunk, LPchunk, kbirth, SInfo, seed=seed, Mflag=MoveLog['Mflag'])
    candidate, newSSchunk, newLPchunk, MoveInfo = create_integrated_proposal_for_chunk(  curmodel, Dchunk, remSSchunk, freshSSchunk, Mflag=MoveLog['Mflag'], nRefineIter=SInfo['nRefineIter'], PropInfo=PropInfo )

  except ProposalError as e:
    print e.message
    MoveInfo = dict(didAccept=0)
    return curmodel, SS, SSchunk, LPchunk, origEv, MoveLog, MoveInfo

  ###################################################################   Decide whether to Accept/Reject
  if acceptname is 'chunk':
    chunknewEv = candidate.calc_evidence( SS=newSSchunk )

    # Advance current state "one step forward"
    #   so that it has same number of Msteps, Esteps as the candidate
    '''
    fwdmodel = curmodel.copy()
    fwdmodel.update_global_params( SS=SSchunk)
    chunkfwdEv = fwdmodel.calc_evidence( SS=SSchunk )    
    #fwdoldLPchunk = fwdmodel.calc_local_params( Dchunk)
    #fwdoldSSchunk = fwdmodel.get_global_suff_stats( Dchunk, fwdoldLPchunk, Eflag=True)
    #chunkfwdEv = fwdmodel.calc_evidence( SS=fwdoldSSchunk )
    '''

    chunkorigEv = curmodel.calc_evidence( SS=SSchunk )
    if doForward:
      doAccept = chunknewEv > chunkfwdEv
      msg = 'birth ev +%4.2e' % (chunknewEv-chunkfwdEv)
    else:      
      doAccept = chunknewEv > chunkorigEv
      msg = 'birth ev +%4.2e' % (chunknewEv-chunkorigEv)

    #print '        fwdEv   %.4e' % (chunkfwdEv)
    print '       origEv   %.4e' % (chunkorigEv)
    print '        newEv   %.4e' % (chunknewEv)

    newEv = origEv
    if doViz:
      MoveInfo['Dchunk'] = PropInfo['Dsub']; MoveInfo['kbirth']=kbirth
      if doForward:
        userResp = viz_birth_proposal( fwdmodel, candidate, MoveInfo, chunkfwdEv, chunknewEv)
      else:
        userResp = viz_birth_proposal( curmodel, candidate, MoveInfo, chunkorigEv, chunknewEv)
      if userResp == 'p':
        embed()
  else:
    newEv  = candidate.calc_evidence( SS=SSnew )
    doAccept = newEv > origEv
    msg = 'birth ev +%4.2e' % (newEv-origEv)
    if doViz:
      userResp = viz_birth_proposal( curmodel, candidate, MoveInfo, origEv, newEv)  
      if userResp == 'p':
        embed()
 
  ############################################################################  Return appropriate stuff
  if doAccept:
    MoveInfo['msg'] = msg
    MoveInfo['didAccept'] = 1
    MoveInfo['Ktotal'] = candidate.K
    MoveInfo['kbirth'] = kbirth

    for kk in MoveInfo['newIDs']:
      MoveLog['lasttryiter'][kk] = iterid
    MoveLog['lasttryiter'][kbirth] = iterid    
    MoveLog['AInfo'].append( MoveInfo )
    MoveLog['AIters'].append( iterid )

    assert newLPchunk['resp'].shape[1] == candidate.K

    # Increment global suff stats with new stuff
    Kextra = MoveInfo['Kextra']
    SSnew = copy.deepcopy(SS)
    dec_suff_stats_from_chunk( SSnew, SSchunk )
    SSnew = get_expanded_suff_stats( SSnew, Kextra)

    if doInflate:
      print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  INFLATING PARAMETERS ONLY'
      MoveLog['inflateIDs'] = MoveInfo['newIDs']

    if doAmp:
      newIDs = MoveInfo['newIDs']
      ampF = SS['N'][kbirth]/np.sum( newSSchunk['N'][newIDs] )
      print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  AMPLIFYING x%.1f' % (ampF)
      newSSchunk['N'][newIDs] *= ampF
      if 'x' in newSSchunk:
        newSSchunk['x'][newIDs] *= ampF
      if 'xxT' in newSSchunk:
        newSSchunk['xxT'][newIDs] *= ampF
      MoveLog['didAmp'] = 1

    inc_suff_stats_from_chunk( SSnew, newSSchunk )

    if not doAmp:
      assert np.allclose( SS['N'].sum(), SSnew['N'].sum() )

    return candidate, SSnew, newSSchunk, newLPchunk, newEv, MoveLog, MoveInfo
  else:
    MoveInfo['didAccept'] = 0
  return curmodel, SS, SSchunk, LPchunk, origEv, MoveLog, MoveInfo


###################################################################
###################################################################  Fresh Model Creation
###################################################################  subsample targeted data from minibatch, train fresh model with Kextra new components, output suff stats
def propose_fresh_suff_stats( hmodel, Dchunk, SSchunk, LPchunk, kbirth, SInfo, seed=42, Mflag=False):
  '''
    Returns
    -------
      remSSchunk : suff stats for pre-existing components, with subsample mass removed
      subSSchunk : suff stats for fresh new components, with total mass = subsample mass
    both remSS and subSS should together have the same mass as the original SSchunk
  '''
  Dsub = subsample_data( Dchunk, LPchunk, kbirth, splitTHR=SInfo['splitTHR'], seed=seed)
  Nswitch = np.sum(  LPchunk['resp'][ Dsub['selectIDs'],:], axis=0 )

  print 'Subsample: Ntotal=%5d' % ( Dsub['nObs'] ) 
  print '   N[kbirth] = %5d | N[others] = %5d' % ( Nswitch[kbirth], np.sum( Nswitch[:kbirth]) + np.sum(Nswitch[kbirth+1:]) )

  # Adjust current "remainder" suff stats for this batch
  #  by removing equivalent mass of the targeted subsampled data
  remSSchunk = copy.deepcopy( SSchunk)
  for kk in range(hmodel.K):
    remFrac = (SSchunk['N'][kk] - Nswitch[kk])/(SSchunk['N'][kk] + 1e-9)
    remSSchunk = remove_suff_stat_mass_from_comp( remSSchunk, kk, remFrac )

  # Get suff stats for some brand new components
  #    scaled so that all removed mass is given to them
  SSsub = train_fresh_model_on_subsample( hmodel, Dsub, Ntotal=np.sum(Nswitch), seed=seed, Mflag=Mflag, **SInfo)
  assert np.allclose( SSchunk['N'].sum(), remSSchunk['N'].sum()+SSsub['N'].sum() )
  assert np.allclose( Dchunk['nObs'], remSSchunk['N'].sum()+SSsub['N'].sum() )
  return remSSchunk, SSsub, dict( Nswitch=Nswitch, fracSwitch=Nswitch/(SSchunk['N']+1e-9), Dsub=Dsub )

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
      emptyTHR=np.maximum(25,0.005*Dsub['nObs'])
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
  print ' '.join( ['%4d'%(x) for x in SSsub['N']/SSsub['ampF'] ] )
  return SSsub

def subsample_data( Data, LP, kbirth, nObsMax=10000, splitTHR=None, seed=42 ):
  origResp =  LP['resp'][:,kbirth]
  rawIDs = np.flatnonzero( origResp > splitTHR )
  Xchunk   =  Data['X'][ rawIDs ]
  nChunk   = Xchunk.shape[0]
  D = Xchunk.shape[1]
  if nChunk <= np.maximum(50, 0.05*D**2):
    raise ProposalError( 'HALT. Not enough data to make a good subsample! nChunk=%d'%(nChunk) )
  nObsMax = np.minimum( nObsMax, nChunk)
  PRNG = np.random.RandomState(seed)
  allIDs = PRNG.permutation( nChunk ).tolist()
  selectIDs = allIDs[:nObsMax]
  Dsub   = dict( X=Xchunk[ selectIDs ], nObs=len(selectIDs), selectIDs=rawIDs[allIDs] )
  return Dsub

def select_birth_component( hmodel, Data, SS, LP, MoveLog, iterid=None, seed=42, splitTHR=None, excludeList=None ):
  ''' Choose a single component from hmodel to attempt to split
  '''
  if 'lasttryiter' not in MoveLog:
    MoveLog['lasttryiter'] = defaultdict(lambda: -100)
  ps = np.zeros( hmodel.K)
  for kk in xrange( hmodel.K):
    lastiter = MoveLog['lasttryiter'][kk]
    ps[kk] = 1e-5 + np.abs(iterid - lastiter)
  Ncand = np.sum(LP['resp']>splitTHR, axis=0)
  ps[ Ncand <= 10  ] = 0
  ps = ps**3  # make distribution much more peaky
  if excludeList is not None:
    ps[excludeList] = 0
  if ps.sum() == 0:
    raise ProposalError( 'HALT. Bad ps calculation') # extremely unlikely!
  return discrete_single_draw( ps, np.random.RandomState(seed) ), dict( ps=ps)

###################################################################
###################################################################  Integration and Refinement [within current chunk]
###################################################################  integrate the fresh components with existing ones, to form suff stats coherent with entire local chunk
def create_integrated_proposal_for_chunk(  hmodel, Dchunk, remSSchunk, freshSSchunk, Mflag=False, doDeleteExtra=True,  nRefineIter=1, PropInfo=dict() ):
  '''
    Returns
    -------
    candidate  : new model with K+Kextra comps, with params at Dchunk scale
    bigSSchunk : new suff stats for chunk, with K+Kextra comps
  '''
  assert len(remSSchunk['N'])==hmodel.K
  Kextra = len( freshSSchunk['N'] )
  Knew = Kextra + hmodel.K
  newIDs = range( hmodel.K, Knew)

  if Kextra == 1:
    raise ProposalError('HALT. Proposal created only one new component.')

  bigSSchunk = create_combined_suff_stats( remSSchunk, freshSSchunk, Kextra )
  #print 'bigSS[N]=', np2flatstr( bigSSchunk['N'], '%5.0f')

  # Finally, create a candidate model with K+Knew components
  candidate = hmodel.copy()
  candidate.set_K( Knew )
  candidate.set_qType( 'VB')

  if 'Nswitch' in PropInfo:
    rankIDs = np.argsort( PropInfo['Nswitch'] )[::-1][ :3]
    reloldIDs = rankIDs[ PropInfo['Nswitch'][rankIDs] > 0.05 ]
    if len( reloldIDs ) == 0:
      reloldIDs = rankIDs[:2]
    print 'reloldIDs:', reloldIDs
    print '          %25s  |  %25s' % ('N[ old ids ]', 'N[ new ids ]')
    print 'before    %25s  |  %25s' % ( np2flatstr(bigSSchunk['N'][reloldIDs],fmt='%6.0f'), np2flatstr(bigSSchunk['N'][newIDs],fmt='%6.0f') )

  emptyTHR=np.maximum(25,0.001*Dchunk['nObs'])  
  # Do a quick E-step, M-step to quickly make the configuration *coherent*
  #   that is, we can be sure there exists a setting of the responsibilities (local params) that produces the current global params/suff stats exactly
  for rr in range( nRefineIter):
    candidate.update_global_params( bigSSchunk )
    # Prune off empties!
    if doDeleteExtra:
      candidate = delete_empty_extra_components( candidate, bigSSchunk, emptyTHR=emptyTHR, Krange=range(hmodel.K,candidate.K)  )
      if candidate.K < Knew:
        print 'removed %d fresh components as too small' % (Knew-candidate.K)
        newIDs = range( hmodel.K, candidate.K)
        Knew = candidate.K
      if candidate.K == hmodel.K + 1:
        raise ProposalError( 'HALT. Only created one new component (after refinement).' )

    bigLPchunk = candidate.calc_local_params( Dchunk)
    if rr == nRefineIter-1:
      bigSSchunk = candidate.get_global_suff_stats( Dchunk, bigLPchunk, Eflag=True, Mflag=Mflag )
    else:
      bigSSchunk = candidate.get_global_suff_stats( Dchunk, bigLPchunk, Eflag=True )
    print 'refine %1d  %25s  |  %25s' % (rr, np2flatstr(bigSSchunk['N'][reloldIDs],fmt='%6.0f'), np2flatstr(bigSSchunk['N'][newIDs],fmt='%6.0f') )
  
  delIDs = [kk for kk in newIDs if bigSSchunk['N'][kk] < emptyTHR ]
  if len(delIDs) == len(newIDs):
    raise ProposalError( 'HALT. No new component larger than %.0f (after refinement). Aborting.'% (emptyTHR) )

  candidate.set_qType( hmodel.qType)
  return candidate, bigSSchunk, bigLPchunk, dict(newIDs=newIDs, Kextra=len(newIDs))

def create_integrated_proposal_for_whole_data( hmodel, SS, Dchunk, oldSSchunk, newSSchunk, newLPchunk, nRefineIter=0, Mflag=False):
  '''
    Returns
    -------
    candidate  : new model with K+Kextra comps, with params at WHOLE DATA scale
    SSnew : new suff stats for WHOLE DATA
  '''
  Kextra = len( newSSchunk['N'] ) - len( oldSSchunk['N'] )
  SSnew = copy.deepcopy(SS)
  dec_suff_stats_from_chunk( SSnew, oldSSchunk )
  #print '     Nrest :', np2flatstr( SSnew['N'], fmt='%9.2f' )
  #print ' cur Nchunk:', np2flatstr( newSSchunk['N'], fmt='%9.2f' )
  #print 'prev Nchunk:', np2flatstr( oldSSchunk['N'] , fmt='%9.2f' )

  SSnew = get_expanded_suff_stats( SSnew, Kextra)
  inc_suff_stats_from_chunk( SSnew, newSSchunk )

  candidate = hmodel.copy()

  '''
  candidate.set_qType( 'VB')
  for iterid in xrange( nRefineIter):
    # This loop is bad since it tries to use stale suff stats from other batches to derail good moves.  Don't use it!
    candidate.update_global_params( SSnew )
    newLPchunk = candidate.calc_local_params( Dchunk)

    dec_suff_stats_from_chunk( SSnew, newSSchunk )
    newSSchunk = candidate.get_global_suff_stats( Dchunk, newLPchunk, Eflag=True, Mflag=Mflag)
    inc_suff_stats_from_chunk( SSnew, newSSchunk )
  candidate.set_qType( hmodel.qType)
  '''
  return candidate, SSnew, newSSchunk, newLPchunk
  
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
  pylab.xlabel( 'ELBO=  %.5e' % (origEv) )
  
  hB=pylab.subplot(1,2,2)
  pylab.hold('on')
  newIDs = infoDict['newIDs']
  newIDs.append( infoDict['kbirth'] )
  GaussViz.plotGauss2DFromModel( newmodel, Hrange=newIDs )
  pylab.plot( X[:1000,0], X[:1000,1], 'k.')
  pylab.title( 'After Birth' )
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

def create_combined_suff_stats( SS, SSextra, Kextra):
  '''  Create combined sufficient stat structure
      Returns
      --------
      SSall : suff stat dictionary with K + Kextra components
  '''
  SSall = get_expanded_suff_stats( SS, Kextra) 
  
  for key in SSall:
    if key == 'Hmerge':
      SSall['Hmerge'][ -Kextra:, -Kextra:] = SSextra[key]
    elif type( SSall[key] ) is np.ndarray and SSall[key].size > 1:
      SSall[key][ -Kextra: ] = SSextra[key]
  assert np.allclose( SSall['Ntotal'], SSall['N'].sum() )
  assert np.allclose( SSall['Ntotal'], SS['Ntotal'] )
  return SSall
    
# Checks!
#origvsum = np.sum( [curmodel.obsModel.comp[kk].v - curmodel.obsModel.obsPrior.v for kk in range(curmodel.K)])
#newvsum = np.sum( [candidate.obsModel.comp[kk].v - curmodel.obsModel.obsPrior.v for kk in range(candidate.K)])
#assert np.allclose( origvsum, newvsum)
#if not np.allclose( origvsum, newvsum):
#  print 'Warning: new params are not at the same scale as the original ones (this can be ok)'

'''
def propose_birth_candidate( hmodel, Dchunk, SSchunk, LPchunk, kbirth, seed=42, SInfo=SInfoDEF):
  # Propose new candidate model with some new components
  #    Returns
  #    -------
  #    candidate : model with K+Kextra comps, but only last Kextra are used
  #    SSchunk   : new suff stats for chunk of data.  has K+Kextra comps, but only last Kextra used
  #    infoDict  : dict object with fields
  #                'newIDs' : ids of new components in candidate,
  Dsub = subsample_data( Dchunk, LPchunk, kbirth, splitTHR=SInfo['splitTHR'], seed=seed)

  # Subsample data assigned to target comp
  if kbirth is None or Dsub is None:
    return None, None, None
  
  print 'kbirth=', kbirth  
  print 'Effective mass stolen:', np.sum(  LPchunk['resp'][ Dsub['selectIDs'],:], axis=0 )

  newFrac = 1.00  
  xSSchunk = copy.deepcopy( SSchunk)
  xSSchunk = remove_suff_stat_mass_from_comp( xSSchunk, kbirth, 1-newFrac)
  
  # Get suff stats for some brand new components
  #    scaled so that all mass of this chunk is given to them
  Ntotal = newFrac*SSchunk['N'][kbirth]
  SSsub = train_new_model_on_subsample( hmodel, Dsub, Ntotal=Ntotal, seed=seed, **SInfo)
  Kextra = len( SSsub['N'] )
      
  #Now modify SSchunk so that it's for a model with K+Kextra comps
  #  where first K comps are like before, and last few equal to SSsub, and kbirth is empty    
  Knew = Kextra + hmodel.K
  newIDs = range( hmodel.K, Knew)  
  SSall = create_combined_suff_stats( xSSchunk, SSsub, Kextra )
  assert np.allclose( SSall['N'].sum(), SSchunk['N'].sum() )
  
  # Finally, create a candidate model with K+Knew components
  candidate = hmodel.copy()
  candidate.set_K( Knew )
  candidate.set_qType( 'VB')
  candidate.update_global_params( SSall )
  candidate.set_qType( hmodel.qType)
  
  infoDict = dict( kbirth=kbirth, newIDs=newIDs, Kextra=len(newIDs), Dchunk=Dchunk)
  return candidate, SSall, infoDict         
  # Now modify SSsub so that it's for model with K+Kextra components
  #   where first K components are empty, last few are exactly equal to SSsub
  #SSall = expand_suff_stats_for_birth( SSsub, hmodel.K )

  for rr in xrange( 20):
    # Create candidate model with new global parameters
    candidate.update_global_params( SSnew )

    dec_suff_stats_from_chunk( SSnew, xSSchunk )

    # Compute new LP and SS for the chunk, using new global params
    xLPchunk = candidate.calc_local_params( Dchunk )

    # Artificially force new components to have more mass!
    xSSchunk = candidate.get_global_suff_stats( Dchunk, xLPchunk, Eflag=True)
    print '     Nrest :', SSnew['N']
    print ' cur Nchunk:', xSSchunk['N']
    #print 'prev Nchunk:', SSchunk['N']
    
    inc_suff_stats_from_chunk( SSnew, xSSchunk )
    
    assert np.allclose( SS['Ntotal'] , SSnew['Ntotal'] )
    assert np.allclose( SS['N'].sum(), SSnew['N'].sum() )
    assert np.allclose( xSSchunk['N'].sum(), Dchunk['nObs'])
    np.set_printoptions( precision=2, suppress=True, linewidth=150)
    #print '     OLD N:', SS['N']
    #print '     NEW N:', SSnew['N']
    #print 'NEW Nchunk:', xSSchunk['N']
    
    newEv = candidate.calc_evidence( SS=SSnew )
    if doViz and rr==0:
      viz_birth_proposal( curmodel, candidate, infoDict, origEv, newEv)
    
    print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>               %.6e' % (newEv)
    if newEv > acceptFactor*origEv:
      break
'''
# Create candidate model for *whole dataset*
#candidate, SSnew, newSSchunk, newLPchunk = create_integrated_proposal_for_whole_data( candidate, SS, Dchunk, SSchunk, xSSchunk, xLPchunk, nRefineIter=0, Mflag=MoveLog['Mflag'])

