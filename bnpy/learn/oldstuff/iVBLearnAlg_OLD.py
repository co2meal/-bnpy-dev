'''
 incremental Variational Bayes learning algorithm

Author: Mike Hughes (mike@michaelchughes.com)

Attempts to (smartly) use disk caching to 

'''
from IPython import embed
import numpy as np
import time
import os
import copy
from collections import defaultdict
from distutils.dir_util import mkpath  #mk_dir functionality
from ..util import np2flatstr
from .LearnAlg import LearnAlg

import glob
import shelve
import cPickle

doShelve = False
doCPickle = False


def inflate_in_place( SS, inflateIDs, passID, nBatch):
  for key in SS:
    if type( SS[key] ) is not np.ndarray:
      continue
    if key == 'x' or key == 'xxT' or key == 'N':
      SS[key][inflateIDs] *= nBatch/float(passID)

def deflate_in_place( SS, inflateIDs, passID, nBatch):
  for key in SS:
    if type( SS[key] ) is not np.ndarray:
      continue
    if key == 'x' or key == 'xxT' or key == 'N':
      SS[key][inflateIDs] /= nBatch/float(passID)

class iVBLearnAlg( LearnAlg ):

  def __init__( self, cachesize=250, cachepath=os.environ['BNPYCACHEPATH'], doWaitFullPass=False, doViz=False, **kwargs ):
    LearnAlg.__init__( self, **kwargs )
    self.Niter = '' # empty
    self.cachepath = cachepath
    self.doViz = doViz
    cachedir, ext = os.path.split( cachepath )
    mkpath( cachedir )
    self.doWaitFullPass = doWaitFullPass
    self.cachesize = cachesize
    self.clean_up_cache()

  def clean_up_cache(self):
    try:
      os.unlink( self.cachepath )
    except Exception:
      flist = glob.glob( self.cachepath+'*' )
      for f in flist:
        os.unlink( f )

  #####################################################################
  #####################################################################
  #####################################################################
  def dec_suff_stats_from_chunk( self, SS, SSchunk ):
    for key in SS:
      if type( SS[key] ) is not np.ndarray and type( SS[key] ) is not np.float64:
        continue
      if SS[key].size >= 1:
        SS[key] -= SSchunk[key]

  def inc_suff_stats_from_chunk( self, SS, SSchunk ):
    for key in SSchunk:
      if key not in SS:
        SS[key] = 1.0*SSchunk[key]
        continue
      if type( SS[key] ) is not np.ndarray and type( SS[key] ) is not np.float64:
        continue
      if SS[key].size >= 1:
        SS[key] += SSchunk[key]

  def fit( self, hmodel, DataGenerator, AllData=None ):
    self.start_time = time.time()
    self.seenIDs = dict()
    prevBound = -np.inf
    SS = dict()
    MoveLog = dict( AIters=list(), AInfo=list() )
    if hasattr(self, 'moves') and 'M' in self.moves:
      MoveLog['Mflag'] = True
    else:
      MoveLog['Mflag'] = False
    self.RandState = np.random.RandomState( self.seed ) #seed inherited from LearnAlg
    status = 'all data gone.'
    canEscape = False
    for iterid, Dchunk in enumerate(DataGenerator):
      
      bID = Dchunk['bID']

      #if Dchunk['passID'] == 1:
      #  embed()

      if 'inflateIDs' in MoveLog:
        if Dchunk['passID'] > 2.0*self.nBatch/3.0:
          print '...................................................................... no longer inflating!'
          del MoveLog['inflateIDs']

      if 'inflateIDs' in MoveLog:
        #print '...................................................................... inflating on pass %d' % (Dchunk['passID'])
        inflate_in_place( SS, MoveLog['inflateIDs'], Dchunk['passID'], self.nBatch )

      if self.doWaitFullPass:
        if bID in self.seenIDs:
          if iterid == len(self.seenIDs.keys()):
            print'************* Updating params now'
          hmodel.update_global_params( SS )
      else:
        if iterid > 0:
          hmodel.update_global_params( SS )

      if 'inflateIDs' in MoveLog:
        deflate_in_place( SS, MoveLog['inflateIDs'], Dchunk['passID'], self.nBatch )

      # Load previous chunk SS,LP from file
      #  may need to run a fast-fwd op if MoveLog says so
      LPchunk = None
      if bID in self.seenIDs:
        SSchunk = self.load_suff_stat_for_chunk( bID )

        if bID == 0:
          beforeVec = SSchunk['N']
          if not np.allclose( self.afterVec, beforeVec):
            print '>>>>>>>>>>>>??????????????################@@@@@@@@@@@@@@@@@%%%%%%%%%%%%%%%%%%%%%'
            print '>>>>>>>>>>>>??????????????################@@@@@@@@@@@@@@@@@%%%%%%%%%%%%%%%%%%%%% WARNING. memoized suff stats not the same!!'
            print '>>>>>>>>>>>>??????????????################@@@@@@@@@@@@@@@@@%%%%%%%%%%%%%%%%%%%%%'
            embed()
          #print 'BEFORE bID %4d | N %40s' % (bID, np2flatstr( SSchunk['N'][:7], '%5.0f') )

        if MoveLog is not None:
          SSchunk = self.run_fast_forward( SS, SSchunk, bID, MoveLog)
        self.dec_suff_stats_from_chunk( SS, SSchunk )

        if bID == 0 and len(MoveLog['AIters'])>0:
          mySSchunk = dict( N=self.afterVec )
          dummySS = copy.deepcopy(mySSchunk)              
          mySSchunk = self.run_fast_forward( dummySS, mySSchunk, bID, MoveLog)
          if np.allclose( mySSchunk['N'], SSchunk['N'] ):
            print ' ????????-----------------------------------------------------------------------  Fast Forward check passed'
          else:
            print '>>>>>>>>>>>>??????????????################@@@@@@@@@@@@@@@@@%%%%%%%%%%%%%%%%%%%%%'
            print '????????>>>>>>>>>>>>>>>>>>##########&&&&&&&&&&@@@@@@@@@@@@@@@@^^^^^^^^^^^^^^^!!!!! Fast Forward check failed <<<<<<<<<<<<<<<<<<<<<<<<<<<<'
            print '>>>>>>>>>>>>??????????????################@@@@@@@@@@@@@@@@@%%%%%%%%%%%%%%%%%%%%%'
            print '      actual ffwd N %25s' % ( np2flatstr(SSchunk['N'], '%5.0f') )
            print '  replicated ffwd N %25s' % ( np2flatstr(mySSchunk['N'], '%5.0f') )

        if hmodel.allocModel.need_prev_local_params():  
          LPchunk = self.load_local_params_for_chunk( bID )
      
      # E-step
      LPchunk = hmodel.calc_local_params( Dchunk, LPchunk )
      SSchunk = hmodel.get_global_suff_stats( Dchunk, LPchunk, Eflag=True, Mflag=MoveLog['Mflag'] )
      self.inc_suff_stats_from_chunk( SS, SSchunk )

      # Write chunk SS,LP to file
      self.save_suff_stat_for_chunk( bID, SSchunk )
      if hmodel.allocModel.need_prev_local_params():  
        self.save_local_params_for_chunk( bID, LPchunk )
  
      if bID == 0:
        self.afterVec = SSchunk['N']
        #print 'AFTER  bID %4d | N %40s' % (bID, np2flatstr( SSchunk['N'][:7], '%5.0f') )

      # ELBO calc
      evBound = hmodel.calc_evidence( SS=SS )
      if AllData is not None:
        firstID = bID * Dchunk['nObs']
        x1 = Dchunk['X'][0][:4]
        x2 = AllData['X'][ firstID ][:4]
        if not np.allclose( x1, x2):
          print '---- data check failed!!!!! ----'
          print '      First row of this chunk', Dchunk['X'][0][:4]
          print 'Corresponding row in all data', AllData['X'][ firstID ][:4]
          print 'Cur Chunk Size :=', Dchunk['nObs']
          print 'All Data Size  :=', AllData['nObs']

      if hasattr(self, 'moves'):
        if (not self.doWaitFullPass or bID in self.seenIDs):
          canEscape = True
          hmodel, SS, evBound, MoveLog = self.run_moves( hmodel, SS, Dchunk, SSchunk, LPchunk, evBound, iterid, MoveLog, AllData=AllData)
          self.MoveLog = MoveLog  # making it a prop of self makes saving convenient

      # Check for Convergence!
      #  throw error if our bound calculation isn't working properly
      #    but only if the gap is greater than some tolerance
      if bID in self.seenIDs:
        isConverged = self.verify_evidence( evBound, prevBound )
        if isConverged and not canEscape:
          status = 'converged.'
          break
        prevBound = evBound

      self.seenIDs[bID] = iterid

      '''
      if hasattr( self, 'birthBatchIDs') and bID == 0 and (bID+1) in self.seenIDs:
        self.offset = (self.offset+1) % self.birthSpan
        self.curbirthBatchIDs = (self.offset+self.birthBatchIDs) % self.nBatch
        if len( self.curbirthBatchIDs)>5:
          print ' >>>>>>>>>>>>>>>>>>> births at batch IDs:', np2flatstr( self.curbirthBatchIDs[:4], '%.0f'), '...', np2flatstr( self.curbirthBatchIDs[-2:], '%.0f' )
        else:
          print ' >>>>>>>>>>>>>>>>>>> births at batch IDs:', np2flatstr( self.curbirthBatchIDs[:5], '%.0f')
      '''

      # Save and display progress
      self.save_state( hmodel, iterid+1, evBound, Dchunk['nObs'])
      self.print_state(hmodel, iterid+1, evBound )


    #Finally, save, print and exit 
    try:
      self.save_state(hmodel, iterid+1, evBound, Dchunk['nObs'], doFinal=True) 
      self.print_state(hmodel, iterid+1, evBound, doFinal=True, status=status)
    except UnboundLocalError:
      print 'No iters performed.  Perhaps DataGen empty. Rebuild DataGen and try again.'
      
  def run_fast_forward( self, SS, SSchunk, bID, MoveLog):
    return SSchunk
  
  #####################################################################
  #####################################################################
  #####################################################################
  def save_suff_stat_for_chunk(self, bID, SSchunk ):
    self.sync_SS( bID )
    self.SSbyChunk[bID] = SSchunk

  def load_suff_stat_for_chunk(self, bID ):
    try:
      bID in self.SSbyChunk
    except AttributeError:
      return None
    except KeyError:
      self.sync_SS( bID )
    SSchunk = self.SSbyChunk[bID]
    return SSchunk

  def sync_SS( self, bID ):
    if not hasattr(self, 'SSbyChunk'):
      self.SSbyChunk = dict()
      return
    if len( self.SSbyChunk.keys() ) > 0:
      oldbID = self.SSbyChunk.keys()[0]
      cIDprev = int(oldbID)/int(self.cachesize)
      cID = int(bID)/int(self.cachesize)
      if cIDprev == cID:
        return
      print '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< OH NO TO DISK!'
      newKey = 'SS%d' % (cID)
      oldKey = 'SS%d' % (cIDprev)
      if doShelve:
        Cache = shelve.open( self.cachepath )
        Cache['SS%d'%(cIDprev)] = self.SSbyChunk
        if newKey in Cache:
          self.SSbyChunk = Cache[ newKey ]
        else:
          self.SSbyChunk = dict()      
        Cache.close()
      elif doCPickle:
        prevpath = self.cachepath + '.' + oldKey
        newpath  = self.cachepath + '.' + newKey
        with open(prevpath, 'wb') as f:
          cPickle.dump( self.SSbyChunk, f )
        if os.path.exists( newpath ):
          with open(newpath, 'rb') as f:
            self.SSbyChunk = cPickle.load( f)
        else:
          self.SSbyChunk = dict()
      else:
        prevpath = self.cachepath + '.' + oldKey
        newpath  = self.cachepath + '.' + newKey
        for keyID in self.SSbyChunk:
          np.savez( prevpath+'+%d'%(keyID), **self.SSbyChunk[keyID] )        
        newlist = glob.glob( newpath+'*')
        self.SSbyChunk = dict()
        for newpath in newlist:
          cSS = np.load( newpath )
          bIDext = newpath[ newpath.index('+')+1: ] 
          bIDstr = bIDext[: bIDext.index('.')]
          bID = int(bIDstr )
          self.SSbyChunk[bID] = dict( **cSS )

  #####################################################################
  #####################################################################
  #####################################################################
  def save_local_params_for_chunk(self, bID, LPchunk ):
    self.sync_LP( bID )
    self.LPbyChunk[bID] = LPchunk

  def load_local_params_for_chunk(self, bID ):
    try:
      return self.LPbyChunk[bID]
    except AttributeError:
      return None
    except KeyError:
      self.sync_LP( bID )
      return self.LPbyChunk[bID]      

  def sync_LP( self, bID ):
    if not hasattr(self, 'LPbyChunk'):
      self.LPbyChunk = dict()
      return
    if len( self.LPbyChunk.keys() ) > 0:
      oldbID = self.LPbyChunk.keys()[0]
      cIDprev = int(oldbID)/int(self.cachesize)
      cID = int(bID)/int(self.cachesize)
      if cIDprev == cID:
        return
      newKey = 'LP%d' % (cID)
      oldKey = 'LP%d' % (cIDprev)
      if doShelve:
        Cache = shelve.open( self.cachepath )
        Cache['LP%d'%(cIDprev)] = self.LPbyChunk
        if newKey in Cache:
          self.LPbyChunk = Cache[ newKey ]
        else:
          self.LPbyChunk = dict()      
        Cache.close()
      elif doCPickle:
        prevpath = self.cachepath + '.' + oldKey
        newpath  = self.cachepath + '.' + newKey
        with open(prevpath, 'wb') as f:
          cPickle.dump( self.LPbyChunk, f )
        if os.path.exists( newpath ):
          with open(newpath, 'rb') as f:
            self.LPbyChunk = cPickle.load( f)
        else:
          self.LPbyChunk = dict()
      else:
        prevpath = self.cachepath + '.' + oldKey
        newpath  = self.cachepath + '.' + newKey
        for keyID in self.LPbyChunk:
          np.savez( prevpath+'+%d'%(keyID), **self.LPbyChunk[keyID] )        
        newlist = glob.glob( newpath+'*')
        self.LPbyChunk = dict()
        for newpath in newlist:
          cLP = np.load( newpath )
          bIDext = newpath[ newpath.index('+')+1: ] 
          bIDstr = bIDext[: bIDext.index('.')]
          bID = int(bIDstr )
          self.LPbyChunk[bID] = dict( **cLP )

