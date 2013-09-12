'''
 Variational Bayes learning algorithm

Author: Mike Hughes (mike@michaelchughes.com)
'''
from IPython import embed
import numpy as np
import time
import copy
from collections import defaultdict

from CombineMove import run_combine_move
from .LearnAlg import LearnAlg

class IncrementalVBLearnAlg( LearnAlg ):

  def __init__( self, ibatch_size=5000, **kwargs ):
    super(type(self), self).__init__( **kwargs )
    self.ibatch_size= ibatch_size
    print '*****************  ibatch_size', ibatch_size

  def get_next_inc_batch( self, Data ):
    '''  Grab a contiguous chunk of data for the next incremental update
            of local responsibilities and sufficient statistics
    '''
    if 'nGroup' in Data:
      return self.get_next_inc_batch_by_group( Data)
    else:
      return self.get_next_inc_batch_iid( Data)

  def get_next_inc_batch_iid( self, Data):
    if not hasattr(self, 'offset'):
      self.offset = 0
      self.mbID   = 0
    batchIDs = self.offset + np.arange(self.ibatch_size)
    batchIDs = batchIDs[ batchIDs < Data['nObs'] ]
    curX = Data['X'][batchIDs]
    nObs = curX.shape[0]
    mbID = int(self.mbID)

    if self.offset + nObs < Data['nObs']:
      self.offset += nObs
      self.mbID   += 1
    else:
      self.offset = 0
      self.mbID   = 0
    return mbID, batchIDs, dict( X=curX, nObs=nObs )

  def get_next_inc_batch_by_group( self, Data):
    '''  In the group setting, ibatch_size says how many groups to do per batch
    '''
    if not hasattr(self, 'offset'):
      self.offset = 0
      self.mbID   = 0

    gIDs = self.offset + np.arange(self.ibatch_size)
    gIDs = gIDs[ gIDs < Data['nGroup'] ]

    GroupIDs = Data['GroupIDs']
    curX = np.vstack( [ Data['X'][ GroupIDs[gg][0]:GroupIDs[gg][1]] for gg in gIDs ] )

    curGroupIDs = list()
    obsIDs = list()
    start=0
    for gg in gIDs:
      stop = start + GroupIDs[gg][1]-GroupIDs[gg][0]
      curGroupIDs.append( (start,stop) )
      obsIDs.extend( np.arange(GroupIDs[gg][0],GroupIDs[gg][1]) )
      start = stop
    assert curGroupIDs[-1][1] == curX.shape[0]
    assert len(obsIDs) == curX.shape[0]

    mbID = int(self.mbID)
    if self.offset + len(curGroupIDs) < Data['nGroup']:
      self.offset += len(curGroupIDs)
      self.mbID   += 1
    else:
      self.offset = 0
      self.mbID   = 0
    return mbID, dict(obsIDs=obsIDs, gIDs=gIDs, G=Data['nGroup']), dict( X=curX, nObs=curX.shape[0], nGroup=len(curGroupIDs), GroupIDs=curGroupIDs )

  #####################################################################
  #####################################################################
  #####################################################################

  def update_local_params_from_chunk( self, LP, bID, ChunkInfo, LPbyChunk ):
    if type( ChunkInfo) is dict:
      self.update_group_local_params_from_chunk( LP, bID, ChunkInfo, LPbyChunk)
      return None
    obsIDs = ChunkInfo
    for key in LP:
      if type( LP[key] ) is not np.ndarray:
        continue
      if LP[key].size > 0:
        LP[key][ obsIDs,:] = LPbyChunk[ bID][key]

  def update_group_local_params_from_chunk( self, LP, bID, ChunkInfo, LPbyChunk):
    for key in LP:
      if type( LP[key] ) is not np.ndarray:
        continue
      if LP[key].shape[0] == 0:
        continue
      if LP[key].shape[0] == ChunkInfo['G']:
        LP[key][ ChunkInfo['gIDs'],:] = LPbyChunk[ bID][key]
      else:
        LP[key][ ChunkInfo['obsIDs'],:] = LPbyChunk[bID][key]

  def dec_suff_stats_from_chunk( self, SS, bID, SSbyChunk ):
    for key in SS:
      if type( SS[key] ) is not np.ndarray:
        continue
      if SS[key].size > 0:
        SS[key] -= SSbyChunk[ bID][key]

  def inc_suff_stats_from_chunk( self, SS, bID, SSbyChunk ):
    for key in SS:
      if type( SS[key] ) is not np.ndarray:
        continue
      if SS[key].size > 0:
        SS[key] += SSbyChunk[ bID][key]

  def fit( self, hmodel, Data ):
    '''
        Notes
        -------
        *order* of Mstep, Estep, and ev calculation is very important
    '''
    
    self.start_time = time.time()
    status = "max iters reached."
    prevBound = -np.inf
    evBound = -1
    #LP = hmodel.calc_local_params( Data )
    #SS = hmodel.get_global_suff_stats( Data, LP, Eflag=True )

    print 'Scanning through all data once to init suff stats'
    LPbyChunk = defaultdict( lambda: None )
    SSbyChunk = defaultdict( lambda: None )
    if 'nGroup' in Data:
      nBatch = Data['nGroup']/self.ibatch_size
    else:
      nBatch = Data['nObs']/self.ibatch_size
    for batch_name in range( 2*nBatch ):
      bID, ChunkInfo, Dchunk = self.get_next_inc_batch( Data )
      if bID == 0 and batch_name > 0:
        break
      LPbyChunk[bID] = hmodel.calc_local_params( Dchunk )            
      SSbyChunk[bID] = hmodel.get_global_suff_stats( Dchunk, LPbyChunk[bID], Eflag=True )
      if bID == 0:
        SS = copy.deepcopy( SSbyChunk[bID] )
      else:
        self.inc_suff_stats_from_chunk( SS, bID, SSbyChunk)

    print 'Now running updates'
    for iterid in xrange(self.Niter):
      if iterid > 0:
        # M-step
        hmodel.update_global_params( SS ) 
      
      bID, ChunkInfo, Dchunk = self.get_next_inc_batch( Data )

      # E-step 
      LPbyChunk[bID] = hmodel.calc_local_params( Dchunk, LPbyChunk[bID] )
      #self.update_local_params_from_chunk( LP, bID, ChunkInfo, LPbyChunk )

      self.dec_suff_stats_from_chunk( SS, bID, SSbyChunk )
      SSbyChunk[bID] = hmodel.get_global_suff_stats( Dchunk, LPbyChunk[bID], Eflag=True )
      self.inc_suff_stats_from_chunk( SS, bID, SSbyChunk )

      evBound = hmodel.calc_evidence( Data, SS)
      #evBound = hmodel.calc_evidence( Data, SS, LP )

      # Save and display progress
      self.save_state(hmodel, iterid+1, evBound, Dchunk['nObs'])
      self.print_state(hmodel, iterid+1, evBound)

      # Check for Convergence!
      #  throw error if our bound calculation isn't working properly
      #    but only if the gap is greater than some tolerance
      isConverged = self.verify_evidence( evBound, prevBound )

      if isConverged:
        status = 'converged.'
        break
      prevBound = evBound

    #Finally, save, print and exit 
    self.save_state(hmodel, iterid+1, evBound, Dchunk['nObs'], doFinal=True) 
    self.print_state(hmodel, iterid+1, evBound, doFinal=True, status=status)
    
