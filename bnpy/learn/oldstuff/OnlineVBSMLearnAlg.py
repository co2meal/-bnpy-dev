'''
 Online/Stochastic Variational Bayes learning algorithm

Author: Mike Hughes (mike@michaelchughes.com)
'''
from IPython import embed
import numpy as np
import time

from .LearnAlg import LearnAlg

from .SplitMove import run_split_move
from .MergeMove import run_merge_move
from .DeleteMove import run_delete_move
from .BirthMove import run_birth_move

class OnlineVBSMLearnAlg( LearnAlg ):

  def __init__( self, splitEvery=10, mergeEvery=5, pruneEvery=5, sortEvery=5, splitWait=20, mergeWait=20, \
                      doViz=False, splitTHR=0.333, splitpropname='rs+batchVB', moves='bsmd', doDeleteExtra=False, nPropIter=0, acceptFactor=1.0, **kwargs ):
    super(type(self), self).__init__( **kwargs )
    self.splitpropname = splitpropname
    self.splitEvery = splitEvery
    self.mergeEvery = mergeEvery
    self.sortEvery  = sortEvery
    self.pruneEvery = pruneEvery
    self.splitWait = splitWait
    self.mergeWait = mergeWait
    self.doViz = doViz
    self.splitTHR = splitTHR
    self.moves = moves
    self.doDeleteExtra = doDeleteExtra
    self.acceptFactor = acceptFactor
    self.nPropIter = nPropIter
    self.Niter = '' #empty

  def fit( self, hmodel, DataGenerator):
    self.start_time = time.time()
    rho = 1.0

    self.nMerge = 0; self.nMergeTotal=0; self.nMergeTry=0;
    self.nSplit = 0; self.nSplitTotal=0; self.nSplitTry=0;
    self.nBirth = 0; self.nBirthTotal=0; self.nBirthTry=0;
    self.nDeath = 0; self.nDeathTotal=0; self.nDeathTry=0;
    for iterid, Dchunk in enumerate(DataGenerator):
      # Mstep update with learning rate
      if iterid > 0:
        rho = ( iterid+1 + self.rhodelay )**(-1*self.rhoexp)
        hmodel.update_global_params( SS, rho )

      # E step
      LP = hmodel.calc_local_params( Dchunk )
      SS = hmodel.get_global_suff_stats( Dchunk, LP, Ntotal=Dchunk['nTotal'] )

      evBound = hmodel.calc_evidence( Dchunk, SS, LP)      

      # Split/Merge/Add/Delete moves
      hmodel, SS, LP, evBound = self.run_moves( iterid, Dchunk, hmodel, SS, LP, evBound)

      # Save and display progress
      self.save_state( hmodel, iterid+1, evBound, Dchunk['nObs'])
      self.print_state(hmodel, iterid+1, evBound, rho=rho)

    #Finally, save, print and exit 
    status = 'all data gone.'
    try:
      self.save_state(hmodel, iterid+1, evBound, Dchunk['nObs'], doFinal=True) 
      self.print_state(hmodel, iterid+1, evBound, doFinal=True, status=status, rho=rho)
      return LP
    except UnboundLocalError:
      print 'No iters performed.  Perhaps DataGen empty. Rebuild DataGen and try again.'


  ###################################################################### Moves to add/remove comps
  ######################################################################
  def run_moves( self, iterid, Data, hmodel, SS, LP, evBound ):
    try:
      MInfo = self.MInfo
    except AttributeError:
      MInfo = dict()
    splitArgs = dict( doViz=self.doViz, splitpropname=self.splitpropname, nPropIter=self.nPropIter, splitTHR=self.splitTHR, doDeleteExtra=self.doDeleteExtra, acceptFactor=self.acceptFactor)

    # Birth step
    if 'b' in self.moves and iterid > self.splitWait and iterid % self.splitEvery == 0:
      hmodel,SS,LP,evBound,MInfo = run_birth_move( hmodel, Data, SS, LP, evBound, \
                                        MoveInfo=MInfo, iterid=iterid, **splitArgs)
      self.nBirth += MInfo['didAccept']
      self.nBirthTry +=1
      if 'msg' in MInfo: print MInfo['msg']

    # Split step
    if 's' in self.moves and iterid > self.splitWait and iterid % self.splitEvery == 0:
      hmodel,SS,LP,evBound,MInfo = run_split_move( hmodel, Data, SS, LP, evBound, \
                                        MoveInfo=MInfo, iterid=iterid, **splitArgs)
      self.nSplit += MInfo['didAccept']
      self.nSplitTry +=1
      if 'msg' in MInfo: print MInfo['msg']

    # Merge step
    if 'm' in self.moves and (iterid+1) > self.mergeWait and (iterid+1) % self.mergeEvery == 0:
      hmodel,SS,LP,evBound,MInfo = run_merge_move( hmodel, Data, SS, LP, evBound, \
                                        MoveInfo=MInfo)
      self.nMerge += MInfo['didAccept']
      self.nMergeTry +=1
      if 'msg' in MInfo: print MInfo['msg']
        
    # Delete step      
    if 'd' in self.moves and iterid % self.pruneEvery == 0:
      hmodel, SS, LP, evBound, MInfo = run_delete_move( hmodel, Data, SS, LP, evBound, \
                                          MoveInfo=MInfo)
      self.nDeath += MInfo['didAccept']
      self.nDeathTry += 1
      if 'msg' in MInfo: print MInfo['msg']

    self.MInfo = MInfo
    return hmodel, SS, LP, evBound
