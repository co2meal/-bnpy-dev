'''
LearnAlg.py
Abstract base class for learning algorithms for HModel models

Defines some generic routines for
  * saving global parameters
  * assessing convergence
  * printing progress updates to stdout
  * recording run-time
'''
from bnpy.ioutil import ModelWriter
from bnpy.util import closeAtMSigFigs, isEvenlyDivisibleFloat
import numpy as np
import time
import os
import logging
import scipy.io

Log = logging.getLogger('bnpy')
Log.setLevel(logging.DEBUG)

class LearnAlg(object):

  def __init__(self, savedir=None, seed=0, 
                     algParams=dict(), outputParams=dict(),
                     onLapCompleteFunc=lambda:None, onFinishFunc=lambda:None,
               ): 
    if type(savedir) == str:
      self.savedir = os.path.splitext(savedir)[0]
    else:
      self.savedir = None
    self.PRNG = np.random.RandomState(seed)
    self.algParams = algParams
    self.outputParams = outputParams
    self.TraceLaps = set()
    self.evTrace = list()
    self.SavedIters = set()
    self.PrintIters = set()
    self.nObsProcessed = 0
    
  def fit(self, hmodel, Data):
    ''' Execute learning algorithm for hmodel on Data
        This method is extended by any subclass of LearnAlg

        Returns
        -------
        LP : local params dictionary of resulting model
    '''
    pass
    
  def set_start_time_now(self):
    ''' Record start time (in seconds since 1970)
    '''
    self.start_time = time.time()    

  def add_nObs(self, nObs):
    self.nObsProcessed += nObs

  def get_elapsed_time(self):
    return time.time() - self.start_time

  def buildRunInfo(self, evBound, status):
    ''' Create dictionary of information about the current run
    '''
    return dict(evBound=evBound, status=status,
                evTrace=self.evTrace, lapTrace=self.TraceLaps)

  ##################################################### Fcns for birth/merges
  ##################################################### 
  def hasMove(self, moveName):
    if moveName in self.algParams:
      return True
    return False

  ##################################################### Verify evidence
  #####################################################  grows monotonically
  def verify_evidence(self, evBound=0.00001, prevBound=0):
    ''' Compare current and previous evidence (ELBO) values,
        verify that (within numerical tolerance) increases monotonically
    '''
    if np.isnan(evBound):
      raise ValueError("Evidence should never be NaN")
    if np.isinf(prevBound):
      return False
    isIncreasing = prevBound <= evBound
    #absDiff = np.abs(prevBound - evBound)
    #percDiff = absDiff / np.abs(prevBound)
    #convergeTHR = self.algParams['convergeTHR']
    #isWithinTHR = absDiff <= convergeTHR or percDiff <= convergeTHR
    M = self.algParams['convergeSigFig']
    isWithinTHR = closeAtMSigFigs(prevBound, evBound, M=M)
    if not isIncreasing:
      if not isWithinTHR:
        if self.hasMove('birth'):
          warnMsg = 'WARNING: ev decreased (during a birth)\n'
        else:
          warnMsg = 'WARNING: evidence decreased!\n'
        warnMsg += '    prev = % .15e\n' % (prevBound) \
                 + '     cur = % .15e\n' % (evBound)
        Log.error(warnMsg)
    return isWithinTHR 


  #########################################################  Save to file
  #########################################################      
  def save_state( self, hmodel, iterid, lap, evBound, doFinal=False):
    ''' Save state of the hmodel's global parameters and evBound
    '''
    traceEvery = self.outputParams['traceEvery']
    if traceEvery <= 0:
      traceEvery = -1
    doTrace = np.allclose(lap % traceEvery, 0) or iterid < 3

    if traceEvery > 0 and (doFinal or doTrace) and lap not in self.TraceLaps:
      # Record current evidence
      self.evTrace.append(evBound)
      self.TraceLaps.add(lap)

      # Exit here if we're not saving to disk
      if self.savedir is None:
        return

      # Define temporary function that creates files in this alg's output dir
      def mkfile(fname):
        return os.path.join(self.savedir, fname)

      # Record current state to plain-text files
      if iterid == 0:
        mode = 'w'
      else:
        mode = 'a'
      with open( mkfile('iters.txt'), mode) as f:        
        f.write('%d\n' % (iterid))
      with open( mkfile('laps.txt'), mode) as f:        
        f.write('%.4f\n' % (lap))
      with open( mkfile('evidence.txt'), mode) as f:        
        f.write('%.9e\n' % (evBound))
      with open( mkfile('nObs.txt'), mode) as f:
        f.write('%d\n' % (self.nObsProcessed))
      with open( mkfile('times.txt'), mode) as f:
        f.write('%.3f\n' % (self.get_elapsed_time()))
      if self.hasMove('birth') or self.hasMove('merge'):
        with open( mkfile('K.txt'), mode) as f:
          f.write('%d\n' % (hmodel.obsModel.K))

    saveEvery = self.outputParams['saveEvery']
    if saveEvery <= 0:
      return

    doSave = np.allclose(lap % saveEvery, 0) or iterid < 3
    if (doFinal or doSave) and iterid not in self.SavedIters:
      self.SavedIters.add(iterid)
      prefix = 'Iter%05d'%(iterid)
      ModelWriter.save_model(hmodel, self.savedir, prefix,
                              doSavePriorInfo=(iterid<1), doLinkBest=True)


  ######################################################### Plot Results
  ######################################################### 
  def plot_results(self, hmodel, Data, LP):
    ''' Plot learned model parameters
    '''
    pass

  #########################################################  Print State
  #########################################################  
  def print_state(self, hmodel, iterid, lap, evBound, doFinal=False, status='', rho=None):
    printEvery = self.outputParams['printEvery']
    if printEvery <= 0:
      return None
    doPrint = iterid < 3 or isEvenlyDivisibleFloat(lap, printEvery)
  
    if rho is None:
      rhoStr = ''
    else:
      rhoStr = '%.4f |' % (rho)

    if iterid == lap:
      lapStr = '%7d' % (lap)
    else:
      lapStr = '%7.2f' % (lap)

    maxLapStr = '%d' % (self.algParams['nLap'])
    
    logmsg = '  %s/%s after %6.0f sec. | K %4d | ev % .9e %s'
    logmsg = logmsg % (lapStr, 
                        maxLapStr,
                        self.get_elapsed_time(),
                        hmodel.allocModel.K,
                        evBound, 
                        rhoStr)
    if self.hasMove('birth') and hasattr(self, 'BirthCompIDs'):
      logmsg += "| Kbirth %3d " % (len(self.BirthCompIDs))
    if self.hasMove('merge') and hasattr(self, 'MergeLog'):
      logmsg += "| Kmerge %3d " % (len(self.MergeLog))



    if (doFinal or doPrint) and iterid not in self.PrintIters:
      self.PrintIters.add(iterid)
      Log.info(logmsg)
    if doFinal:
      Log.info('... done. %s' % (status))
      
  def print_msg(self, msg):
      ''' Prints a string msg to stdout,
            without needing to import logging method into subclass. 
      '''
      Log.info(msg)

  #########################################################
  def isFirstBatch(self, lapFrac):
    ''' Returns True/False for whether given batch is last (for current lap)
    '''
    if self.lapFracInc == 1.0: # Special case, nBatch == 1
      isFirstBatch = True
    else:
      isFirstBatch = np.allclose(lapFrac - np.floor(lapFrac), self.lapFracInc)
    return isFirstBatch

  def isLastBatch(self, lapFrac):
    ''' Returns True/False for whether given batch is last (for current lap)
    '''
    return lapFrac % 1 == 0

  def do_birth_at_lap(self, lapFrac):
    ''' Returns True/False for whether birth happens at given lap
    '''
    if 'birth' not in self.algParams:
      return False
    nLapTotal = self.algParams['nLap']
    frac = self.algParams['birth']['fracLapsBirth']
    if lapFrac > nLapTotal:
      return False
    return (nLapTotal <= 5) or (lapFrac <= np.ceil(frac * nLapTotal)) 
