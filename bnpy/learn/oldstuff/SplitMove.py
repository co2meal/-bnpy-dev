'''
'''
from IPython import embed
from collections import defaultdict
import numpy as np
import copy

from ..util import discrete_single_draw
from .GibbsSamplerAlg import GibbsSamplerAlg
from .VBLearnAlg import VBLearnAlg

from .DeleteMove import run_delete_move, delete_empty_components

def run_split_move( curmodel, Data, SS=None, LP=None, origEv=None, MoveInfo=None, iterid=0, ksplit=None, doViz=False, doDeleteExtra=False, splitpropname='rs+batchVB', splitTHR=0.25, acceptFactor=1.0, nPropIter=None):
  ''' Returns
      -------
      hmodel : new (or current) HModel, if proposal is accepted (rejected)
      SS
      LP
      evBound   :
      MoveInfo : dict of information about this proposal
  '''
  if SS is None or LP is None:
    LP = curmodel.calc_local_params( Data )
    SS = curmodel.get_global_suff_stats( Data, LP)
  if origEv is None:
    origEv = curmodel.calc_evidence( Data, SS, LP )
  if MoveInfo is None:
    MoveInfo = dict()
  MoveInfo['didAccept'] = 0
  if 'msg' in MoveInfo:
    del MoveInfo['msg']
  if 'lasttryiter' not in MoveInfo:
    MoveInfo['lasttryiter'] = defaultdict(int)
    
  # Select which component to alter by ID
  if ksplit is None:
    ksplit = select_split_component( curmodel, Data, SS, LP, MoveInfo, iterid=iterid, splitTHR=splitTHR )

  # Create candidate model
  candidate, infoDict = propose_split_candidate( curmodel, Data, SS, LP, ksplit, splitpropname=splitpropname, splitTHR=splitTHR, nPropIter=nPropIter, doDeleteExtra=doDeleteExtra)
  MoveInfo.update( infoDict )
  if candidate is None:
    MoveInfo['didAccept'] = 0
    return curmodel, SS, LP, origEv, MoveInfo

  # Decide whether to accept/reject
  xLP = candidate.calc_local_params( Data )
  if 'Ntotal' in SS:
    xSS = candidate.get_global_suff_stats( Data, xLP, Ntotal=SS['Ntotal'])
  else:
    xSS = candidate.get_global_suff_stats( Data, xLP)
  newEv = candidate.calc_evidence( Data, xSS, xLP)

  if doViz:
    viz_split_proposal( curmodel, candidate, infoDict, origEv, newEv)
 
  if newEv > acceptFactor * origEv:
    msg = 'split ev +%4.2e' % (newEv-origEv)
    MoveInfo['msg'] = msg
    for kk in infoDict['newIDs']:
      MoveInfo['lasttryiter'][kk] = iterid
    MoveInfo['didAccept'] = 1
    return candidate, xSS, xLP, newEv, MoveInfo
  else:
    MoveInfo['lasttryiter'][ksplit] = iterid
    MoveInfo['didAccept'] = 0

  return curmodel, SS, LP, origEv, MoveInfo


def select_split_component( hmodel, Data, SS, LP, MoveInfo, iterid=None, splitTHR=None):
  ''' Choose a single component from hmodel to attempt to split
  '''
  ps = np.zeros( hmodel.K)
  for kk in xrange( hmodel.K):
    lastiter = MoveInfo['lasttryiter'][kk]
    ps[kk] = 1e-5 + np.abs(iterid - lastiter)
  Ncand = np.sum(LP['resp']>splitTHR, axis=0)
  ps[ Ncand <= 10  ] = 0
  ps = ps**3  # make distribution much more peaky
  if ps.sum() == 0:
    return None # extremely unlikely!
  return discrete_single_draw( ps )

    
def propose_split_candidate( hmodel, Data, SS, LP, ksplit, Kextra=8, splitTHR=None, doDeleteExtra=False, nPropIter=50, splitpropname='rs+batchVB'):
  ''' Propose new candidate model with some new components
      Returns
      -------
      candidate : ExpFamModel
      infoDict  : dict object with fields
                  'newIDs' : ids of new components in candidate,
  '''
  infoDict = dict( ksplit=ksplit )

  # Subsample data assigned to target comp
  if ksplit is None:
    return None, infoDict
  origResp =  LP['resp'][:,ksplit]
  Xchunk   =  Data['X'][ origResp > splitTHR ]
  nChunk   = Xchunk.shape[0]
  D = Xchunk.shape[1]
  if nChunk <= 0.5*D**2:
    print nChunk
    return None, infoDict
  nSelect = 10000
  nSelect = np.minimum( nSelect, nChunk)
  selectIDs = np.random.permutation( nChunk )[:nSelect] 

  Dchunk   = dict( X=Xchunk[ selectIDs ], nObs=len(selectIDs) )
  infoDict['Dchunk'] = Dchunk

  candidate = hmodel.copy()
  candidate.reset_K( Kextra )
  candidate.set_qType( 'VB' ) # make sure to do batch-style updates!

  if splitpropname == 'pp+batchVB':
    infer = VBLearnAlg( printEvery=-1, saveEvery=-1, nIter=nPropIter,initname='plusplus')
    infer.init_global_params( candidate, Dchunk, seed=selectIDs[0] )
    LPchunk = infer.fit( candidate, Dchunk )
  elif splitpropname == 'rs+batchVB':
    infer = VBLearnAlg( printEvery=-1, saveEvery=-1, nIter=nPropIter,initname='randsample')
    infer.init_global_params( candidate, Dchunk, seed=selectIDs[0] )
    LPchunk = infer.fit( candidate, Dchunk )
  elif splitpropname == 'seq+batchVB':
    infer = VBLearnAlg( printEvery=-1, saveEvery=-1, nIter=nPropIter, initname='seqCGS')
    infer.init_global_params( candidate, Dchunk, seed=selectIDs[0], nIterInit=1, printEvery=-1, Kmax=Kextra )
    LPchunk = infer.fit( candidate, Dchunk )
    candidate.set_qType( 'VB' )
  elif splitpropname == 'pca+batchVB':
    infer = VBLearnAlg( printEvery=-1, saveEvery=-1, nIter=nPropIter, initname='pca')
    infer.init_global_params( candidate, Dchunk, seed=selectIDs[0] )
    LPchunk = infer.fit( candidate, Dchunk )
    candidate.set_qType( 'VB' )
  elif splitpropname == 'seqCGS':
    infer = VBLearnAlg( printEvery=-1, saveEvery=-1, nIter=1, initname='seqCGS')
    infer.init_global_params( candidate, Dchunk, seed=selectIDs[0], nIterInit=2, printEvery=-1 )
    LPchunk = infer.fit( candidate, Dchunk )
    candidate.set_qType( 'VB' )

  # Clean up
  if doDeleteExtra:
    candidate, SSchunk, LPchunk, ev, Info = run_delete_move( candidate, Dchunk, LP=LPchunk, emptyFRAC=0.05, doDecide=False)
    SSchunk = candidate.get_global_suff_stats( Dchunk, LPchunk, Ntotal=SS['N'][ksplit] )
  else:
    SSchunk = candidate.get_global_suff_stats( Dchunk, LPchunk, Ntotal=SS['N'][ksplit] )
  
  Kextra = candidate.K
  if Kextra <= 1:
    return None, infoDict

  SSall, Knew, newIDs = create_combined_suff_stats( hmodel, SS, SSchunk, Kextra, ksplit )
  infoDict['newIDs'] = newIDs

  # Create new candidate model that uses all the available components
  candidate = hmodel.copy()
  candidate.set_K( Knew )
  candidate.set_qType( 'VB' ) # make sure to do batch-style updates!

  candidate.update_global_params( SSall, Krange=newIDs )
  candidate.set_qType( hmodel.qType )
  return candidate, infoDict

def create_combined_suff_stats( hmodel, SS, SSchunk, Kextra, ksplit ):
  '''  Create combined sufficient stat structure
      Returns
      --------
      SSall : suff stat dictionary with hmodel.K + Kextra-1 components
                replacing ksplit with a new component
  '''
  Knew = hmodel.K + Kextra -1
  NewIDs = np.hstack( [ksplit, np.arange( hmodel.K, Knew)] )
  SSall = copy.deepcopy( SS ) 
       
  hmodel.expand_suff_stats_in_place( SSall, Kextra-1 )
  for key in SSall:
    if type( SSall[key] ) is np.ndarray and SSall[key].size > 1:
      SSall[key][ NewIDs ] = SSchunk[key]
  assert np.allclose( SSall['Ntotal'], SSall['N'].sum() )
  return SSall, Knew, NewIDs
    
def viz_split_proposal( curmodel, newmodel, infoDict, origEv, newEv):
  if curmodel.obsModel.D == 2:
    viz_split_proposal_2D( curmodel, newmodel, infoDict, origEv, newEv)
    return None
  from matplotlib import pylab
  fig = pylab.figure()
  
  hA = pylab.subplot( 1, 4, 1)
  pylab.hold('on')
  pylab.imshow( curmodel.obsModel.get_covar_mat( infoDict['ksplit']), interpolation='nearest' )  
  pylab.title( 'Before Split' )
  pylab.xlabel( 'ELBO=  %.5e' % (origEv) )
  
  pylab.hold('on')

  newIDs = infoDict['newIDs'].tolist()
  w =  np.exp(newmodel.allocModel.Elogw[newIDs] )
  myOrder = np.argsort( w )
  w = sorted(w)[::-1]
  newIDs = np.asarray(newIDs)[myOrder[::-1]]
  for nn in range(3):  
    hB=pylab.subplot(1, 4, nn+2)
    pylab.imshow( newmodel.obsModel.get_covar_mat(newIDs[nn]), interpolation='nearest' )
    pylab.title( '%.2f'%(w[nn]) )

  pylab.xlabel( 'ELBO=  %.5e \n %d' % (newEv, newEv > origEv) )
  pylab.show(block=False)
  fig.canvas.draw()
  
  try: 
    x = raw_input('Press any key to continue >>')
  except KeyboardInterrupt:
    doViz = False
  pylab.close()  


def viz_split_proposal_2D( curmodel, newmodel, infoDict, origEv, newEv):
  ''' Create before/after visualization of a split proposal
  '''
  from ..viz import GaussViz
  from matplotlib import pylab
  X = infoDict['Dchunk']['X']
  fig = pylab.figure()
  hA = pylab.subplot( 1, 2, 1)
  pylab.plot( X[:,0], X[:,1], 'k.')
  GaussViz.plotGauss2DFromModel( curmodel, Hrange=[infoDict['ksplit']] )
  pylab.title( 'Before Split K=%d' % (curmodel.K) )
  pylab.xlabel( 'ELBO=  %.2e' % (origEv) )
    
  hB=pylab.subplot(1,2,2)
  pylab.plot( X[:,0], X[:,1], 'k.')
  GaussViz.plotGauss2DFromModel( newmodel, Hrange=infoDict['newIDs'] )

  pylab.title( 'After Split K=%d' % (newmodel.K) )
  pylab.xlabel( 'ELBO=  %.2e \n %d' % (newEv, newEv > origEv) )
  pylab.show(block=False)
  fig.canvas.draw()

  try: 
    x = raw_input('Press any key to continue >>')
  except KeyboardInterrupt:
    doViz = False
  pylab.close()
          
