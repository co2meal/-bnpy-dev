from IPython import embed
import numpy as np
from collections import defaultdict
import copy

from ..util import EPS, discrete_single_draw, np2flatstr

def get_adjust_factor( SS, kA, kB, adjustname):
  #if 'Hmerge' in SS:
  #  return SS['Hmerge'][kA,kB] - SS['Hz'][kA] - SS['Hz'][kB]
  if adjustname == 'log2':
    return (SS['N'][kA] + SS['N'][kB])*np.log(2.0)
  else:
    return SS['N'][kA] + SS['N'][kB]

def run_many_combine_moves( curmodel, Data, SS, LP=None, origEv=None, iterid=None, MoveLog=dict(), randstate=None, \
                        kA=None, kB=None, mname='marglik', adjustname='log2', AllData=None, doViz=False, Ntrial=20, doonlyAobs=False, doAMPLIFYTERRIBLE=False, verbosity=1):
  if origEv is None:
    origEv = curmodel.calc_evidence( SS=SS )
  MoveInfo = dict( didAccept=0, nAttempt=0)
  # Need at least two components to merge!
  if curmodel.K == 1:
    return curmodel, SS, origEv, MoveLog, MoveInfo

  if 'knext' in MoveLog:
    excludeList = [int(k) for k in MoveLog['knext'].tolist() if k >=0]
  else:
    excludeList = []
  compList = [ k for k in range( curmodel.K) if k not in excludeList ]
  randstate.shuffle( compList )

  for trial in range(Ntrial):
    if verbosity > 1:
      print 'compList', compList[:10]

    MoveInfo['nAttempt']+= 1
    if len(compList) < 2 or len(excludeList) >= curmodel.K-1:
      break
    kA = compList[0]
    compList.remove( kA)

    kA, kB = select_combine_components( curmodel, Data, SS, LP, mname, kA, randstate, excludeList=excludeList )
    if kA >= kB:
      if verbosity > 1:
        print '???????????????? bad pick.  Selected bad combine candidates %d,%d' % (kA, kB)
      continue

    if kA in excludeList or kB in excludeList:
      if verbosity > 1:
        print '???????????????? BADNESS.  Selected bad combine candidates %d,%d' % (kA, kB)
        print '  skipping this attempt (reverting back to state where everything is fine)'
      continue

    curmodel, SS, origEv, MoveLog, MInfo = run_combine_move( curmodel, Data, SS, LP, origEv, iterid, MoveLog, randstate, kA, kB, mname, adjustname, AllData, doViz, verbosity=verbosity, doAMPLIFYTERRIBLE=doAMPLIFYTERRIBLE)

    if MInfo['didAccept']:
      excludeList.append( kA)

      MoveInfo['didAccept'] += 1
      if 'msg' not in MoveInfo:
        MoveInfo['msg'] = ''
      MoveInfo['msg'] += MInfo['msg']+'\n'
      if curmodel.K == 1:
        break
      # we can remove kB from consideration, since we merged it away!
      try:
        compList.remove( kB)
      except:
        pass

      if mname == 'overlap':
        temp = LP['resp']
        LP['resp'] = np.zeros( (temp.shape[0], curmodel.K) )
        LP['resp'][:, :kB] = temp[:, :kB]
        LP['resp'][:, kB:curmodel.K] = temp[:, kB+1:]
        del temp
        # dont need to combine kA,kB, since its on exclude list

      if len(compList) < 2:
        break
      # Need to adjust our component indices down by one!
      compList = np.asarray( compList )
      excludeList = np.asarray( excludeList)
      compList[ compList > kB] -= 1
      excludeList[ excludeList > kB] -= 1
      compList = compList.tolist()
      excludeList = excludeList.tolist()

      assert max( compList ) < curmodel.K
      assert max( excludeList ) < curmodel.K

  return curmodel, SS, origEv, MoveLog, MoveInfo

def run_combine_move( curmodel, Data, SS, LP=None, origEv=None, iterid=None, MoveLog=dict(), randstate=None, \
                        kA=None, kB=None, mname='marglik', adjustname='log2', AllData=None, doViz=False, doonlyAobs=False, verbosity=1, doAMPLIFYTERRIBLE=False):
  if origEv is None:
    origEv = curmodel.calc_evidence( SS=SS )
  MoveInfo = dict( didAccept=0)
  # Need at least two components to merge!
  if curmodel.K == 1:
    return curmodel, SS, origEv, MoveLog, MoveInfo

  # Select components to merge by ID
  if kB is None:
    kA, kB = select_combine_components( curmodel, Data, SS, LP, mname=mname, kA=kA, randstate=randstate )
  else:
    kMin = np.minimum(kA,kB)
    kB  = np.maximum(kA,kB)
    kA = kMin

  if verbosity > 1:
    print '>>>>>>>----------- Attempting combine for   kA=%3d | kB=%3d' % (kA, kB)
  if verbosity > 1:
    print '                                      N=  %8.2f | %8.2f' % (SS['N'][kA], SS['N'][kB])

  # Create candidate merged model
  candidate, fastmSS, MoveInfo = propose_combined_candidate( curmodel, Data, SS, kA, kB, doonlyAobs=doonlyAobs, doAMPLIFYTERRIBLE=doAMPLIFYTERRIBLE)

  if 'Hmerge' not in SS:
    if 'Hz_adjust' not in fastmSS:
      fastmSS['Hz_adjust'] = dict()
    fastmSS['Hz_adjust'][iterid] = get_adjust_factor( SS, kA, kB, adjustname)

  # Decide whether to accept the merge
  newEv = candidate.calc_evidence( SS=fastmSS )

  if AllData is not None:
    aLP = candidate.calc_local_params( AllData)
    aSS = candidate.get_global_suff_stats( AllData, aLP)
    aEv = candidate.calc_evidence( AllData, aSS, aLP)
    oEv2 = curmodel.calc_evidence( AllData )
    print '            Orig approx Ev %.6e' % (origEv)
    print '          Merged approx Ev %.6e | %d' % (newEv, newEv > origEv)
    print '[expensive]   Orig True Ev %.6e' % (oEv2)
    print '[expensive] Merged True Ev %.6e | %d' % (aEv, aEv > oEv2)
    assert oEv2 >= origEv

  if verbosity > 1:
    print '       origEv   %.4e' % (origEv)
    print '        newEv   %.4e' % (newEv)
  '''
  print '  AllocModel'
  acur = curmodel.allocModel.calc_evidence( None, SS,None )
  anew = candidate.allocModel.calc_evidence( None, fastmSS,None )
  print '       origEv   %.4e' % (acur)
  print '        newEv   %.4e' % (anew )
  print '                                 diff   % 8.2f' % (anew-acur)
  print '  ObsModel'
  ocur = curmodel.obsModel.calc_evidence( None, SS,None  ) 
  onew = candidate.obsModel.calc_evidence( None, fastmSS,None ) 
  print '       origEv   %.4e' % (ocur)
  print '        newEv   %.4e' % (onew)
  print '                                 diff   % 8.2f' % (onew-ocur)
  '''
  if doViz:
    MoveInfo['Dchunk'] = Data
    MoveInfo['iterid'] = iterid # for saving fig to file
    viz_proposal( curmodel, candidate, MoveInfo, origEv, newEv)

  if newEv > origEv:
    msg = 'combine ev +%4.2e' % (newEv-origEv)
    if verbosity > 1:
      print '******************* ACCEPTED'
    MoveInfo['msg'] = msg
    MoveInfo['didAccept'] = 1
    # Record permanent info needed to replay this move again later
    MoveLog['AIters'].append( iterid )
    MoveLog['AInfo'].append( dict( Knew=candidate.K, kA=kA, kB=kB) )
    
    if 'knext' in MoveLog:
      assert kA not in MoveLog['knext'] 
      assert kB not in MoveLog['knext'] 
      adjINDS= MoveLog['knext'] > kB
      oldval = MoveLog['knext']
      if np.sum(adjINDS) > 0:
        MoveLog['knext'][adjINDS] -= 1
        print '**** corrected on deck kbirth'
        print '   ', oldval
        print '   ', MoveLog['knext']

    if 'lasttryiter' in MoveLog:
      keepVals = [ MoveLog['lasttryiter'][kk] for kk in MoveInfo['keepIDs'] ]
      MoveLog['lasttryiter'] = defaultdict(int)
      for key in range(candidate.K):
        MoveLog['lasttryiter'][key] = keepVals[key]
    
    return candidate, fastmSS, newEv, MoveLog, MoveInfo
  else:
    MoveInfo['didAccept'] = 0
  return curmodel, SS, origEv, MoveLog, MoveInfo

def get_merge_suff_stats( SS, kA, kB ):
  mSS = copy.deepcopy( SS )       
  doCorrect = False
  for key in mSS:
    if key is 'Hmerge':
      mSS[key] = np.delete( mSS[key], kB, axis=0 )
      mSS[key] = np.delete( mSS[key], kB, axis=1 )
      doCorrect = True      
      continue      
    if type( mSS[key] ) is np.ndarray and mSS[key].size > 1:
      mSS[key][ kA ] = SS[key][kA] + SS[key][kB]
      mSS[key] = np.delete( mSS[key], kB, axis=0 )
  if doCorrect:
    mSS['Hz'][kA] = SS['Hmerge'][kA,kB]
  return mSS

def select_combine_components( curmodel, Data, SS, LP, mname='marglik', kA=None, randstate=None, excludeList=[], emptyTHR=0.0001 ):
  if mname == 'random':
    ps = np.ones( curmodel.K )
    ps[ excludeList] = 0
    if kA is None:
      kA = discrete_single_draw( ps, randstate )
    ps[kA] = 0
    kB = discrete_single_draw( ps, randstate )
  elif mname == 'overlap' and LP is not None:
    R = LP['resp'].copy()
    N = R.sum(axis=0)
    Overlap = np.dot( R.T, R)/np.outer( N+EPS, N+EPS)
    for kk in xrange(curmodel.K):
      Overlap[kk,:kk+1]=0
      if kk in excludeList:
        Overlap[kk, :] = 0
        Overlap[:,kk]  = 0
    if kA is not None:
      Overlap[ :kA, :] = 0
      Overlap[ kA+1:, :] = 0
    Overlap = Overlap/(EPS+np.max(Overlap) )
    ps = Overlap.flatten()    
    if np.sum( ps ) < EPS:
      ps = np.ones( curmodel.K )
      ps[ kA] = 0
      ps[ excludeList] = 0
      kB = discrete_single_draw(  ps, randstate )
    else:
      maxID = discrete_single_draw(  ps, randstate )
      kA = int( maxID / curmodel.K ) #row ID.  kA always < kB so stickbreak order is good
      kB = int( maxID % curmodel.K ) #col ID.
  elif mname == 'marglik':
    ps = np.ones( curmodel.K )
    if kA is None:
      kA = discrete_single_draw(  ps, randstate )
    '''
    if SS['N'][kA] < emptyTHR*SS['Ntotal']:
      print 'IN EMPTY ZONE'
      ps = np.ones(curmodel.K)
      ps[kA] = 0
      ps[excludeList]= 0
    else:
    '''
    logmA = curmodel.obsModel.calc_log_marg_lik_combo( SS, kA)  
    logscore = -1*np.inf*np.ones( curmodel.K)    
    for kB in xrange( curmodel.K):
      if kB == kA or kB in excludeList:
        continue
      logmB = curmodel.obsModel.calc_log_marg_lik_combo( SS, kB)
      logmCombo = curmodel.obsModel.calc_log_marg_lik_combo( SS, kA, kB)
      logscore[kB] = logmCombo - logmA - logmB
    ps = np.exp( logscore - np.max(logscore))
    if np.sum( ps ) < EPS:
      ps = np.ones( curmodel.K )
      ps[ kA] = 0
      ps[ excludeList] = 0
    #print randstate.rand(4)
    kB = discrete_single_draw(  ps, randstate )
  kMin = np.minimum(kA,kB)
  kB  = np.maximum(kA,kB)
  kA = kMin
  assert kB < curmodel.K
  return kA, kB

def propose_combined_candidate( curmodel, Data, SS, kA, kB, doonlyAobs, doAMPLIFYTERRIBLE=False):
  candidate = curmodel.copy()  
  # Rewrite candidate's kA component to be the merger of kA+kB  
  mSS = get_merge_suff_stats( SS, kA, kB)
  # Remove component kB
  beforeList = np.arange(kB)
  afterList = np.arange(kB+1, curmodel.K)
  keepIDs = np.hstack( [beforeList, afterList] ).tolist()
  candidate.allocModel.delete_components( keepIDs )
  candidate.obsModel.delete_components( keepIDs )
  candidate.K = candidate.K -1
  # Update global parameters with combined suff stats
  if doonlyAobs:
    candidate.update_global_params( mSS, Krange=[kA] )
  elif doAMPLIFYTERRIBLE:
    # amplify just the params of the candidate
    aSS = copy.deepcopy(mSS)
    aSS['N'] *= doAMPLIFYTERRIBLE
    aSS['Ntotal'] *= doAMPLIFYTERRIBLE    
    if 'x' in aSS:  
      aSS['x'] *= doAMPLIFYTERRIBLE
    if 'xxT' in aSS:
      aSS['xxT'] *= doAMPLIFYTERRIBLE
    candidate.update_global_params( aSS)
  else:
    candidate.update_global_params( mSS)
  assert candidate.K == candidate.allocModel.K
  assert candidate.K == candidate.obsModel.K
  #print 'Inspecting deg-of-freedom, before and after'
  #for kk in range( candidate.K):
  #  print '%8.2f | %8.2f' % (curmodel.obsModel.comp[kk].dF, candidate.obsModel.comp[kk].dF)
  return candidate, mSS, dict( keepIDs = keepIDs, kA=kA, kB=kB)

###################################################################
###################################################################  Visualization
###################################################################
def viz_proposal( curmodel, newmodel, infoDict, origEv, newEv):
  if curmodel.obsModel.D == 2:
    viz_proposal_2D( curmodel, newmodel, infoDict, origEv, newEv)
    return None
  from matplotlib import pylab
  fig = pylab.figure()
  
  kA = infoDict['kA']
  kB = infoDict['kB']
  hA = pylab.subplot( 2, 2, 1)
  pylab.imshow( curmodel.obsModel.get_covar_mat( kA), interpolation='nearest' )  
  w = np.exp( curmodel.allocModel.Elogw )
  pylab.title( 'Before kA=%d | w[%d] = %.3f' %(kA, kA, w[kA]) )
  hB = pylab.subplot( 2, 2, 2)
  pylab.imshow( curmodel.obsModel.get_covar_mat( kB), interpolation='nearest' )  
  pylab.title( 'Before kB=%d| w[%d] = %.3f' %(kB, kB, w[kB]) )
  pylab.xlabel( 'ELBO=  %.5e' % (origEv) )
  
  neww = np.exp( newmodel.allocModel.Elogw )
  hC = pylab.subplot(2, 2, 3)
  pylab.imshow( newmodel.obsModel.get_covar_mat( kA), interpolation='nearest' )
  pylab.title( 'After | w[%d] = %.3f' %(kA, neww[kA]) )

  pylab.xlabel( 'ELBO=  %.5e \n %d' % (newEv, newEv > origEv) )
  pylab.show(block=False)
  fig.canvas.draw()
  #pylab.savefig( 'mergefig_iterid%d.png' % (infoDict['iterid']) )
  try: 
    x = raw_input('Press any key to continue >>')
  except KeyboardInterrupt:
    doViz = False
  pylab.close()  


def viz_proposal_2D( curmodel, newmodel, infoDict, origEv, newEv):
  ''' Create before/after visualization of a split proposal
  '''
  from ..viz import GaussViz
  from matplotlib import pylab
  X = infoDict['Dchunk']['X']
  s = np.random.RandomState( hash(newEv)%10000 )
  pIDs = s.permutation( X.shape[0])[:5000]
  X = X[ pIDs ]

  kA = infoDict['kA']
  kB = infoDict['kB']

  w = np.exp( curmodel.allocModel.Elogw )
  fig = pylab.figure()
  hA = pylab.subplot( 1, 2, 1)
  pylab.plot( X[:,0], X[:,1], 'k.')
  GaussViz.plotGauss2DFromModel( curmodel, Krange=[kA, kB], wTHR=0 )
  #pylab.title( 'Before Merge K=%d' % (curmodel.K) )
  pylab.title( 'Before | w[a] = %.3f w[b] = %.3f' %(w[kA], w[kB]) )
  pylab.xlabel( 'ELBO=  %.4e' % (origEv) )
    
  neww = np.exp( newmodel.allocModel.Elogw )
  hB=pylab.subplot(1,2,2)
  pylab.plot( X[:,0], X[:,1], 'k.')
  GaussViz.plotGauss2DFromModel( newmodel, Krange=[kA], coffset=3, wTHR=0 )

  pylab.title( 'After Merge | w[new]=%.3f' % (neww[kA] ) )
  pylab.xlabel( 'ELBO=  %.4e \n %d' % (newEv, newEv > origEv) )
  pylab.show(block=False)
  fig.canvas.draw()
  #pylab.savefig( 'mergefig2D_iterid%d.png' % (infoDict['iterid']) )
  try: 
    x = raw_input('Press any key to continue >>')
  except KeyboardInterrupt:
    doViz = False
  pylab.close()
