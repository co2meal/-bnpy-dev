from collections import defaultdict

def run_delete_move( hmodel, Data, SS=None, LP=None, origEv=None, MoveInfo=dict(), emptyFRAC=0.005, doDecide=True):
  if LP is None:
    LP = hmodel.calc_local_params( Data )
  if SS is None:
    SS = hmodel.get_global_suff_stats( Data, LP)
  if doDecide and origEv is None:
    origEv = hmodel.calc_evidence( Data, SS, LP )
  MoveInfo['didAccept'] = 0
  if 'msg' in MoveInfo:
    del MoveInfo['msg']
   
  candidate = hmodel.copy()
  MoveInfo = delete_empty_components( candidate, Data, SS, emptyFRAC= emptyFRAC*SS['N'].sum(), MoveInfo=MoveInfo)
  if candidate.K == hmodel.K:
    MoveInfo['didAccept'] = 0
    return hmodel, SS, LP, origEv, MoveInfo
    
  xLP = candidate.calc_local_params(Data)
  if 'nTotal' in Data:
    xSS = candidate.get_global_suff_stats( Data, xLP, Ntotal=Data['nTotal'])
  else:
    xSS = candidate.get_global_suff_stats( Data, xLP)

  if not doDecide:
    MoveInfo['didAccept'] = 1    
    return candidate, xSS, xLP, 0, MoveInfo

  newEv = candidate.calc_evidence(Data, xSS, xLP)

  if newEv > origEv:
    msg = 'delete ev +%4.2e' % (newEv-origEv)
    MoveInfo['msg'] = msg
    MoveInfo['didAccept'] = 1
    return candidate, xSS, xLP, newEv, MoveInfo
  else:
    MoveInfo['didAccept'] = 0
    
  return hmodel, SS, LP, origEv, MoveInfo

def delete_empty_components( hmodel, Data, SS, emptyFRAC, MoveInfo=None):
  Korig = hmodel.K
  keepIDs = [kk for kk in xrange(hmodel.K) if SS['N'][kk]> emptyFRAC ]
  if len(keepIDs) == hmodel.K:
    MoveInfo['delIDs'] = []
    return MoveInfo
  delIDs  = set( xrange(hmodel.K) ).difference( keepIDs )
  hmodel.allocModel.delete_components( keepIDs)  
  hmodel.obsModel.delete_components( keepIDs)
  if MoveInfo is not None and 'lasttryiter' in MoveInfo:
    tmpdict = MoveInfo['lasttryiter']
    MoveInfo['lasttryiter'] = defaultdict(int)
    jj = 0
    for kk in xrange( Korig ):
      if kk in delIDs:
        continue
      MoveInfo['lasttryiter'][jj] = tmpdict[kk]
      jj += 1
  hmodel.K = len(keepIDs)
  MoveInfo['delIDs'] = delIDs
  return MoveInfo
