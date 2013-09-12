from collections import defaultdict
from IPython import embed
import numpy as np
import copy

def get_adjust_factor( SS, delIDs ):
  ep = SS['N'][delIDs].max()
  Kdel = len(delIDs)
  return np.sum( SS['N'][delIDs] )*np.log( Kdel*ep ), np.log( Kdel*ep)

def run_fastdeath_move( hmodel, SS, origEv=None, MoveLog=dict(), emptyFRAC=0.05, iterid=0, doDecide=True):
  MoveInfo = dict()
  MoveInfo['didAccept'] = 0
  if 'msg' in MoveInfo:
    del MoveInfo['msg']

  if origEv is None:
    origEv = hmodel.calc_evidence( SS=SS )
   
  keepIDs = [kk for kk in xrange(hmodel.K) if SS['N'][kk]> emptyFRAC ]
  if len(keepIDs) >= hmodel.K-1:
    return hmodel, SS, origEv, MoveLog, MoveInfo  

  delIDs  = set( xrange(hmodel.K) ).difference( keepIDs )
  delIDs = list(delIDs)
  assert len( delIDs ) > 0
  consolID = np.min( delIDs )
  keepIDs.append( consolID )
  

  MoveInfo['delIDs'] = delIDs
  MoveInfo['keepIDs'] = keepIDs
  MoveInfo['consolID'] = consolID
  candidate, mSS, MoveInfo = delete_empty_components( hmodel, SS, keepIDs, consolID, MoveInfo)

  if 'Hz_adjust' not in mSS:
    mSS['Hz_adjust'] = dict()
  mSS['Hz'][ consolID ] = 0
  Nlogadj, logadj = get_adjust_factor( SS, delIDs)
  mSS['Hz_adjust'][iterid] = Nlogadj
  MoveInfo['logadj'] = logadj

  if not doDecide:
    MoveInfo['didAccept'] = 1    
    return candidate, xSS, origEv, MoveLog, MoveInfo

  newEv = candidate.calc_evidence( SS=mSS )

  if newEv > origEv:
    msg = 'delete ev +%4.2e' % (newEv-origEv)
    MoveInfo['msg'] = msg
    MoveInfo['didAccept'] = 1
    MoveLog['AInfo'].append( MoveInfo )
    MoveLog['AIters'].append( iterid )
    return candidate, mSS, newEv, MoveLog, MoveInfo
  else:
    MoveInfo['didAccept'] = 0
  return hmodel, SS, origEv, MoveLog, MoveInfo

def get_consolidated_suff_stats( SS, keepIDs, consolID ):
  mSS = copy.deepcopy( SS )       
  K = len( SS['N'] )  
  for key in mSS:
    if type( mSS[key] ) is np.ndarray and mSS[key].size > 1:
      for k in range( K )[::-1]:
        if k not in keepIDs:
          mSS[key][ consolID ] += SS[key][k]
          mSS[key] = np.delete( mSS[key], k, axis=0 )
  return mSS

def delete_empty_components( hmodel, SS, keepIDs, consolID, MoveInfo ):
  Korig = hmodel.K

  candidate = hmodel.copy()
  candidate.allocModel.delete_components( keepIDs)  
  candidate.obsModel.delete_components( keepIDs)
  candidate.K = len(keepIDs)

  mSS = get_consolidated_suff_stats( SS, keepIDs, consolID )
  candidate.update_global_params( mSS )

  if MoveInfo is not None:
    if 'lasttryiter' in MoveInfo:
      tmpdict = MoveInfo['lasttryiter']
      MoveInfo['lasttryiter'] = defaultdict(int)
      jj = 0
      for kk in xrange( Korig ):
        if kk not in keepIDs:
          continue
        MoveInfo['lasttryiter'][jj] = tmpdict[kk]
        jj += 1

  return candidate, mSS, MoveInfo
