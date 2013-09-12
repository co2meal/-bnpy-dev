import numpy as np
from collections import defaultdict
import copy

from ..util import EPS, discrete_single_draw

def run_merge_move( curmodel, Data, SS=None, LP=None, origEv=None, iterid=None, MoveInfo=dict(), doViz=False ):
  MoveInfo['didAccept'] = 0
  if 'msg' in MoveInfo:
    del MoveInfo['msg']
  
  if SS is None or LP is None:
    LP = curmodel.calc_local_params( Data )
    SS = curmodel.get_global_suff_stats( Data, LP)
  if origEv is None:
    origEv = curmodel.calc_evidence( Data, SS, LP)
    
  # Need at least two components to merge!
  if curmodel.K == 1:
    return curmodel, SS, LP, origEv, MoveInfo  
      
  # Select components to merge by ID
  kA, kB = select_merge_components( curmodel, Data, SS, LP )

  # Create candidate merged model
  candidate,MoveInfo = propose_merge_candidate( curmodel, Data, SS, LP, kA, kB,MoveInfo)
  
  # Decide whether to accept the merge
  mLP = candidate.calc_local_params( Data )
  if 'Ntotal' in SS:
    mSS = candidate.get_global_suff_stats( Data, mLP, Ntotal=SS['Ntotal'])
  else:
    mSS = candidate.get_global_suff_stats( Data, mLP)
  newEv = candidate.calc_evidence( Data, mSS, mLP)
  assert np.allclose( mSS['Ntotal'] , SS['Ntotal'] )
  assert np.allclose( np.sum(mSS['N']) , mSS['Ntotal'] )
    
  if doViz:
    viz_merge_proposal( curmodel, candidate, kA, kB, origEv, newEv)
    
  if newEv > origEv:
    msg = 'merge ev +%4.2e' % (newEv-origEv)
    MoveInfo['msg'] = msg
    MoveInfo['didAccept'] = 1
    if 'lasttryiter' in MoveInfo:
      keepVals = [ MoveInfo['lasttryiter'][kk] for kk in MoveInfo['keepIDs'] ]
      MoveInfo['lasttryiter'] = defaultdict(int)
      for key in range(candidate.K):
        MoveInfo['lasttryiter'][key] = keepVals[key]
    return candidate, mSS, mLP, newEv, MoveInfo
  else:
    MoveInfo['didAccept'] = 0    
  return curmodel, SS, LP, origEv, MoveInfo

def select_merge_components( curmodel, Data, SS, LP):
  Overlap = np.dot( LP['resp'].T, LP['resp'] )/np.outer( SS['N']+EPS,SS['N']+EPS)
  for kk in xrange(curmodel.K):
    Overlap[kk,:kk+1]=0
  if np.random.rand() < 0.5:
    sourcename = 'Argmax!'
    maxID = np.argmax( Overlap )
  else:
    sourcename = 'Sampled!'
    maxID = discrete_single_draw( Overlap.flatten() )
  kA = int( maxID / curmodel.K ) #row ID.  kA always < kB so stickbreak order is good
  kB = int( maxID % curmodel.K ) #col ID. 
  return kA, kB    

def propose_merge_candidate( curmodel, Data, SS, LP, kA, kB, MoveInfo):
  candidate = curmodel.copy()
  
  # Rewrite candidate's kA component to be the merger of kA+kB  
  mSS = get_merge_suff_stats( SS, kA, kB)
  candidate.update_global_params( mSS, Krange=[kA] )
  
  # Remove component kB
  beforeList = np.arange(kB)
  afterList = np.arange(kB+1, curmodel.K)
  keepIDs = np.hstack( [beforeList, afterList] ).tolist()
  candidate.allocModel.delete_components( keepIDs )
  candidate.obsModel.delete_components( keepIDs )
  candidate.K = candidate.K -1

  assert candidate.K == candidate.allocModel.K
  assert candidate.K == candidate.obsModel.K

  MoveInfo['keepIDs'] = keepIDs
  return candidate, MoveInfo

def get_merge_suff_stats( SS, kA, kB ):
  mSS = copy.deepcopy( SS ) 
       
  for key in mSS:
    if type( mSS[key] ) is np.ndarray and mSS[key].size > 1:
      mSS[key][ kA ] = SS[key][kA] + SS[key][kB]
  return mSS

def viz_merge_proposal( curmodel, candidate, kA, kB, origEv, newEv ):
  from ..viz import GaussViz
  from matplotlib import pylab
    
  fig = pylab.figure()
  hA = pylab.subplot(1,2,1)
  GaussViz.plotGauss2DFromModel( curmodel, Hrange=[kA, kB] )
  pylab.title( 'Before Merge' )
  pylab.xlabel( 'ELBO=  %.2e' % (origEv) )
    
  hB = pylab.subplot(1,2,2)
  GaussViz.plotGauss2DFromModel( candidate, Hrange=[kA] )
  pylab.title( 'After Merge' )
  pylab.xlabel( 'ELBO=  %.2e \n %d' % (newEv, newEv > origEv) )

  pylab.show(block=False)

  try: 
    x = raw_input('Press any key to continue >>')
  except KeyboardInterrupt:
    doViz = False
  pylab.close()
