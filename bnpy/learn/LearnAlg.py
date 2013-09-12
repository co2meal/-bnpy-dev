'''
Abstract base class for learning algorithms for HModel models

Defines some generic routines for
  ** saving global parameters
  ** assessing convergence
  ** printing progress updates to stdout
  ** recording run-time
'''
import numpy as np
import time
import os
import logging

from bnpy.ioutil import ModelWriter

Log = logging.getLogger('bnpy')
Log.setLevel(logging.DEBUG)

class LearnAlg(object):

  def __init__(self, savedir=None, seed=0, argDict=dict()):
    self.args = argDict
    self.savedir = os.path.splitext(savedir)[0]
    self.TraceIters = dict()
    self.SavedIters = dict()
    self.nObsProcessed = 0
    if 'printEvery' in argDict:
      self.printEvery = argDict['printEvery']
    else:
      self.printEvery = 0
    if 'saveEvery' in argDict:
      self.saveEvery = argDict['saveEvery']
    else:
      self.saveEvery = 0
    if 'convergeTHR' in argDict:
      self.convergeTHR = argDict['convergeTHR']
    
  def fit(self, hmodel, Data, seed):
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

  #########################################################  
  #########################################################  Verify evidence monotonic
  #########################################################  
  def verify_evidence(self, evBound=0.00001, prevBound=0, EPS=1e-9):
    ''' Compare current and previous evidence (ELBO) values,
        and verify that (within numerical tolerance) evidence increases monotonically
    '''
    isIncreasing = prevBound <= evBound
    absDiff = np.abs(prevBound - evBound)
    percDiff = absDiff / np.abs(prevBound)
    isWithinTHR = absDiff <= self.convergeTHR or percDiff <= self.convergeTHR
    if not isIncreasing:
      if not isWithinEPS:
        warnMsg = 'WARNING: evidence decreased!\n' \
          + '    prev = % .15e\n' % (prevBound) \
          + '     cur = % .15e\n' % (evBound)
        Log.severe(warnMsg)
    return isWithinTHR 


  #########################################################  
  #########################################################  Save to file
  #########################################################  
  def save_state( self, hmodel, iterid, lap, evBound, doFinal=False):
    if self.saveEvery <= 0:
      return    

    if iterid not in self.TraceIters:
      if iterid == 0:
        mode = 'w'
      else:
        mode = 'a'
      if doFinal or (iterid % (self.traceEvery)==0):
        self.TraceIters[iterid] = True
        with open( self.savedir+'iters.txt', mode) as f:        
          f.write('%.d\n' % (iterid))
        with open( self.savedir+'laps.txt', mode) as f:        
          f.write('%.4f\n' % (lap))
        with open( self.savedir+'evidence.txt', mode) as f:        
          f.write('%.9e\n' % (evBound))
        with open( self.savedir+'nObs.txt', mode) as f:
          f.write('%d\n' % (self.nObsProcessed))
        with open( self.savedir+'times.txt', mode) as f:
          f.write('%.3f\n' % (self.get_elapsed_time()))

    if iterid not in self.SavedIters:
      if doFinal or iterid < 3 or ( iterid % (self.saveEvery)==0 ):
        self.SavedIters[iterid] = True
        prefix = 'Iter%05d'%(iterid)
        ModelWriter.save_model( hmodel, self.savedir, prefix, doSavePriorInfo=(iterid<1) )

  #########################################################  
  #########################################################  Print State
  #########################################################  
  def print_state( self, hmodel, iterid, lap, evBound, doFinal=False, status='', rho=None):
    if self.printEvery <= 0:
      return None
    doPrint = iterid < 3 or iterid % self.printEvery==0
      
    if rho is None:
      rhoStr = ''
    else:
      rhoStr = '%.4f |' % (rho)

    logmsg = '  %5d/%d after %6.0f sec. | K %4d | ev % .9e %s'
    logmsg = logmsg % (iterid, 
                        self.Niter, 
                        self.get_elapsed_time(),
                        hmodel.allocModel.K,
                        evBound, 
                        rhoStr)
    
    Log.info(logmsg)
    if doFinal:
      Log.info('... done. %s' % (status))
