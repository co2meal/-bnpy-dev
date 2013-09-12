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
from FastDeathMove import run_fastdeath_move, get_consolidated_suff_stats
from CombineMove import run_combine_move, get_adjust_factor, get_merge_suff_stats, run_many_combine_moves
from FastBirthMove import run_fastbirth_move, get_expanded_suff_stats
from TargetBirthMove import target_subsample_data, run_targetbirth_move, select_birth_component, ProposalError

from .iVBLearnAlg import iVBLearnAlg

import glob
import shelve
import cPickle

doShelve = False
doCPickle = False

class iVBSMLearnAlg( iVBLearnAlg ):

  def __init__( self, moves='bmd', doAmpBirth=False, doInflateBirth=False, splitpropname='seq+batchVB', doonlyAobs=False, nPropIter=1, Kextra=10, splitTHR=0.1, mname='overlap', adjustname='log2', birthPerPass=1, mergePerPass=20, nBatch=None, nTarget=5000, nEpoch=2, fracBirth=0.5, nRep=None, doVerboseMerge=1, **kwargs ):
    super(type(self), self).__init__( **kwargs )
    self.nMerge = 0; self.nMergeTotal=0; self.nMergeTry=0; self.nMergeTotalTry=0;
    self.nBirth = 0; self.nBirthTotal=0; self.nBirthTry=0; self.nBirthTotalTry=0;
    self.nDeath = 0; self.nDeathTotal=0; self.nDeathTry=0; self.nDeathTotalTry=0;
    self.nSplit = 0; self.nSplitTotal=0; self.nSplitTry=0; self.nSplitTotalTry=0;
    self.nTarget = nTarget
    self.doonlyAobs = doonlyAobs
    self.adjustname = adjustname
    self.moves= moves
    self.birthPerPass = birthPerPass
    self.mergePerPass = mergePerPass
    self.mname = mname
    self.nBatch = nBatch
    self.doInflateBirth = doInflateBirth
    self.doAmpBirth = doAmpBirth
    self.nEpoch = nEpoch
    self.fracBirth = fracBirth
    self.nRep = nRep
    self.doVerboseMerge=doVerboseMerge
    self.SInfo = dict( splitpropname=splitpropname, nPropIter=nPropIter, splitTHR=splitTHR, Kextra=Kextra )
    print 'SINFO: ', [ '%s=%s'%(k,v) for k,v in self.SInfo.items() ]
    print 'MergeINFO: mname %s | mergePerPass %d | do only update one comp %d' % (mname, mergePerPass, doonlyAobs)
    epLen = nRep/nEpoch
    start = 0
    self.doTargetBirthAtRepID = dict()
    for ee in range( nEpoch ):
      med = start + int( np.floor( fracBirth*(epLen-1) ) )
      stop = start + epLen-1
      print '  Birth %d-%d / Rest %d-%d' % (start, med, med+1, stop)
      for nn in range(start, med+1):
        self.doTargetBirthAtRepID[nn] = True
      start = stop+1
    '''
    bFrac = float(1)/float(birthPerPass)
    if nBatch is None:
      raise ValueError( 'Need to specify nBatch!!!!!!!!!!!!!!!!!!')    
    if nBatch is not None:
      self.nBatch = nBatch
      self.offset = 0
      if bFrac >= 1:
        self.birthBatchIDs = np.asarray( [nBatch] )
      else:
        self.birthBatchIDs = np.unique( np.round( np.arange( bFrac, bFrac+1.0, bFrac)*nBatch ) )
      self.birthSpan = np.maximum( 1.0, self.birthBatchIDs[0] )

      self.curbirthBatchIDs = self.birthBatchIDs.copy()
      print 'Performing births at these batch IDs:', np2flatstr( self.birthBatchIDs[:4], '%.2f'), '...' 
    '''
    
  #####################################################################
  #####################################################################
  #####################################################################
  def run_fast_forward( self, SS, SSchunk, bID, MoveLog):
    for aa in xrange( len( MoveLog['AIters'])):
      moveiterid = MoveLog['AIters'][aa]
      if moveiterid >= self.seenIDs[bID]:
        if 'kA' in MoveLog['AInfo'][aa]:
          ########################################### fast fwd combine/merge move
          kA = MoveLog['AInfo'][aa]['kA']
          kB = MoveLog['AInfo'][aa]['kB']

          if 'Hz_adjust' in SS:
            adjF = get_adjust_factor( SSchunk, kA, kB, self.adjustname )
            SS['Hz_adjust'][moveiterid] -= adjF
            if np.allclose( SS['Hz_adjust'][moveiterid], 0):
              print 'No longer need to adjust for the merge at iter %d' % (moveiterid)
              del SS['Hz_adjust'][moveiterid]
          else:
            # get_merge_suff_stats automagically inserts correct merge Hz into SSchunk
            pass

          SSchunk = get_merge_suff_stats( SSchunk, kA, kB )
          assert len(SSchunk['N']) == MoveLog['AInfo'][aa]['Knew']

        elif 'consolID' in MoveLog['AInfo'][aa]:
          ########################################### fast fwd fastdelete move
          delIDs = MoveLog['AInfo'][aa]['delIDs']
          keepIDs = MoveLog['AInfo'][aa]['keepIDs']
          consolID = MoveLog['AInfo'][aa]['consolID']
          logadj = MoveLog['AInfo'][aa]['logadj']
          SS['Hz_adjust'][moveiterid] -= np.sum(SSchunk['N'][delIDs])*logadj
          SSchunk = get_consolidated_suff_stats( SSchunk, keepIDs, consolID )
        elif 'kbirth' in MoveLog['AInfo'][aa]:
          Ktotal = MoveLog['AInfo'][aa]['Ktotal']
          if len( SSchunk['N']) < Ktotal:  #may not need to do this 
            Kextra = MoveLog['AInfo'][aa]['Kextra']
            SSchunk = get_expanded_suff_stats( SSchunk, Kextra)
    return SSchunk

  def run_subsampling( self, Dchunk, LPchunk, MoveLog):
    if 'B' in self.moves and Dchunk['repID'] in self.doTargetBirthAtRepID and 'knext' in MoveLog:
      repID = Dchunk['repID']
      for kk in xrange( len(MoveLog['knext']) ):
        if MoveLog['knext'][kk] < 0:
          continue
        if self.tt[kk] == self.nTarget:
          continue
        try:
          Xsub, Info = target_subsample_data( Dchunk, LPchunk, MoveLog['knext'][kk], nObsMax=self.nTarget, splitTHR=self.SInfo['splitTHR'], PRNG=self.RandState)
        except ProposalError as e:
          return
        nObs = Xsub.shape[0]
        nTotal = self.tt[kk] + nObs
        if nTotal < self.nTarget:
          self.TargetX[kk,self.tt[kk]:nTotal ] = Xsub
          self.tt[kk] = nTotal 
        else:
          self.TargetX[kk,self.tt[kk]:self.nTarget ] = Xsub[:(self.nTarget-self.tt[kk])]
          self.tt[kk] = self.nTarget
        #nCheck = np.sum( self.TargetX[kk,:,0] != 0 )
        #print '                                                                                      kk %d Ncur %d Ntarget = %d | check %d' % (MoveLog['knext'][kk], nObs, self.tt[kk], nCheck)
    
  def run_first_moves(self, hmodel, SS, iterid, passID, repID, MoveLog):
    if 'B' not in self.moves:
      return hmodel, SS, MoveLog    

    if repID not in self.doTargetBirthAtRepID:
      if passID == 0:
        # Reset target counter
        self.tt.fill( 0 )
        self.TargetX.fill( 0 )
      if 'knext' in MoveLog:
        del MoveLog['knext']
      return hmodel, SS, MoveLog    
    if 'B' in self.moves and not hasattr(self, 'TargetX'):
      self.TargetX = np.zeros( (self.birthPerPass, self.nTarget, hmodel.obsModel.D ) )
      self.tt = np.zeros( (self.birthPerPass) )
    if 'B' in self.moves and passID == self.nBatch-1:
      if 'SSbirth' in MoveLog:
        del MoveLog['SSbirth']
        print "                                                                                     removing SSbirth"
    if 'B' in self.moves and passID ==0:
      # Run proposal based on TargetX
      if 'knext' in MoveLog:
        for kk, knext in enumerate(MoveLog['knext']):
          if knext==-1 or self.tt[kk]<=0:
            continue
          print '************ BIRTH. kbirth %d | nSamples=%d' % (MoveLog['knext'][kk], self.tt[kk])
          Dtarget = dict( X=self.TargetX[kk,:self.tt[kk] ], nObs=self.tt[kk] )
          hmodel, SS, MoveLog, MoveInfo = run_targetbirth_move( hmodel, SS, Dtarget, MoveLog['knext'][kk], iterid=iterid, MoveLog=MoveLog, seed=iterid, doViz=self.doViz, SInfo=self.SInfo)

      # Pick a new "on-deck" component
      if repID+1 in self.doTargetBirthAtRepID:
        MoveLog['knext'] = -1*np.ones( np.minimum(hmodel.K, self.birthPerPass) )
        excludeList=[]
        for kk in range( np.minimum(hmodel.K, self.birthPerPass) ):
          knext, Info = select_birth_component( hmodel.K, SS, iterid, MoveLog, PRNG=self.RandState, emptyTHR=0.05*self.nTarget, excludeList=excludeList)
          excludeList.append( knext )
          if 'N' in SS:
            print '                                                                                      ON DECK kbirth=%d. N = %5d' % (knext, SS['N'][knext])
          else:
            print '                                                                                      ON DECK kbirth=%d. SS[N] unknown.' % (knext)
          MoveLog['knext'][kk] = knext
          # Reset target counter
          self.tt[kk] = 0
          self.TargetX[kk].fill( 0 )
    return hmodel, SS, MoveLog

  '''  Classic "run moves"
  '''
  def run_moves( self, hmodel, SS, Dchunk, SSchunk, LPchunk, evBound, iterid, MoveLog, AllData=None):
    bID = Dchunk['bID']      
    if 'b' in self.moves and Dchunk['passID'] < self.birthPerPass:      
      hmodel, SS, SSchunk, LPchunk, evBound, MoveLog, curMInfo = run_fastbirth_move( hmodel, SS, Dchunk, SSchunk, LPchunk, origEv=evBound, iterid=iterid, MoveLog=MoveLog, doViz=self.doViz, SInfo=self.SInfo, seed=self.RandState.randint(0,10000), doAmp=self.doAmpBirth, doInflate=self.doInflateBirth)
      if 'msg' in curMInfo: print curMInfo['msg']
      self.nBirth += curMInfo['didAccept']
      self.nBirthTry +=1
      if curMInfo['didAccept']:
        self.save_suff_stat_for_chunk( bID, SSchunk )
        if bID == 0:
          self.afterVec = SSchunk['N'] # debugging
        if hmodel.allocModel.need_prev_local_params():  
          self.save_local_params_for_chunk( bID, LPchunk )

    if ('M' in self.moves or 'm' in self.moves) and Dchunk['passID'] == self.nBatch-1:
      if 'didAmp' in MoveLog:
        print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  skipping merge due to amplification'
        del MoveLog['didAmp']
      else:
        hmodel, SS, evBound, MoveLog, curMInfo = run_many_combine_moves( hmodel, Dchunk, SS, LPchunk, origEv=evBound, iterid=iterid, randstate=self.RandState, MoveLog=MoveLog, adjustname=self.adjustname, AllData=AllData, doonlyAobs=self.doonlyAobs, Ntrial=self.mergePerPass, mname=self.mname, verbosity=self.doVerboseMerge)
        if 'msg' in curMInfo: print curMInfo['msg']
        self.nMerge += curMInfo['didAccept']
        self.nMergeTry += curMInfo['nAttempt']

    if 'd' in self.moves and iterid > 10 and (iterid % 5 == 0):
      hmodel, SS, evBound, MoveLog, curMInfo = run_fastdeath_move( hmodel, SS, origEv=evBound, iterid=iterid, MoveLog=MoveLog )
      if 'msg' in curMInfo: print curMInfo['msg']
      self.nDeath += curMInfo['didAccept']
      self.nDeathTry +=1

    if 'p' in self.moves and Dchunk['passID'] == self.nBatch-1:
      doAMP = SS['Ntotal']/SSchunk['Ntotal']
      # DON'T PROVIDE "origEv", since we need to compute the evidence for CURRENT MINIBATCH ONLY
      hmodel, blah, doubleblah, MoveLog, curMInfo = run_many_combine_moves( hmodel, Dchunk, SSchunk, LPchunk, iterid=iterid, randstate=self.RandState, MoveLog=MoveLog, adjustname=self.adjustname, AllData=AllData, doonlyAobs=self.doonlyAobs, Ntrial=self.mergePerPass, mname=self.mname, verbosity=self.doVerboseMerge, doAMPLIFYTERRIBLE=doAMP)
      if 'msg' in curMInfo: print curMInfo['msg']
      self.nMerge += curMInfo['didAccept']
      self.nMergeTry += curMInfo['nAttempt']
      if curMInfo['didAccept']:
        for ii, miterid in enumerate(MoveLog['AIters'] ):
          if miterid == iterid:
            SS = get_merge_suff_stats( SS, MoveLog['AInfo'][ii]['kA'], MoveLog['AInfo'][ii]['kB'] )
        #self.save_suff_stat_for_chunk( bID, SSchunk )

    return hmodel, SS, evBound, MoveLog

