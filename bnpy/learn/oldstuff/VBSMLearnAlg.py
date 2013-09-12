'''
 Variational Bayes learning algorithm

Author: Mike Hughes (mike@michaelchughes.com)
'''
import numpy as np
import time
from IPython import embed

from .LearnAlg import LearnAlg
from .SplitMove import run_split_move
from .MergeMove import run_merge_move
from .DeleteMove import run_delete_move
from .BirthMove import run_birth_move

class VBSMLearnAlg( LearnAlg ):

  def __init__( self, splitEvery=10, mergeEvery=5, pruneEvery=5, sortEvery=5, splitWait=20, mergeWait=20, \
                      doViz=False, splitTHR=0.333, splitpropname='rs+batchVB', nPropIter=50, moves='bsmd', doDeleteExtra=False, **kwargs ):
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
    self.nPropIter = nPropIter
    
  def fit( self, hmodel,Data ):
    self.start_time = time.time()
    status = "max iters reached."
    prevBound = -np.inf
    evBound = -1
    LP = None
    self.nMerge = 0; self.nMergeTotal=0; self.nMergeTry=0;
    self.nSplit = 0; self.nSplitTotal=0; self.nSplitTry=0;
    self.nBirth = 0; self.nBirthTotal=0; self.nBirthTry=0;
    self.nDeath = 0; self.nDeathTotal=0; self.nDeathTry=0;

    splitArgs = dict( doViz=self.doViz, splitpropname=self.splitpropname, splitTHR=self.splitTHR, doDeleteExtra=self.doDeleteExtra)
    MInfo = dict()
    for iterid in xrange(self.Niter):

      ###################################################################### Usual VB steps
      ######################################################################
      if iterid > 0:
        # M-step
        hmodel.update_global_params( SS ) 
      
      # E-step
      LP = hmodel.calc_local_params( Data, LP )
      SS = hmodel.get_global_suff_stats( Data, LP )

      evBound = hmodel.calc_evidence( Data, SS, LP )
      
      # Moves to add/remove comps
      hmodel, SS, LP, evBound = self.run_moves( iterid, Data, hmodel, SS, LP, evBound)
          
      # Save and display progress
      self.save_state(hmodel, iterid+1, evBound, nObs=Data['nObs'])
      self.print_state(hmodel, iterid+1, evBound)

    #Finally, save, print and exit 
    self.save_state(hmodel,iterid+1, evBound, nObs=Data['nObs'], doFinal=True) 
    self.print_state(hmodel,iterid+1, evBound, doFinal=True, status=status)
    return LP

  ###################################################################### Moves to add/remove comps
  ######################################################################
  def run_moves( self, iterid, Data, hmodel, SS, LP, evBound ):    
    try:
      MInfo = self.MInfo
    except AttributeError:
      MInfo = dict()

    splitArgs = dict( doViz=self.doViz, splitpropname=self.splitpropname, \
                      splitTHR=self.splitTHR, doDeleteExtra=self.doDeleteExtra, nPropIter=self.nPropIter)

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
