'''
 Abstract class for learning algorithms for HModel models

  Simply defines some generic routines for
    ** saving global parameters
    ** assessing convergence
    ** printing progress updates to stdout

Author: Mike Hughes (mike@michaelchughes.com)
'''
from IPython import embed
import numpy as np
import time
import os
import logging

from bnpy.ioutil import ModelWriter

Log = logging.getLogger('bnpy')
Log.setLevel(logging.DEBUG)

class LearnAlg(object):

  def __init__( self, savedir, argDict ):
    self.args = argDict
    self.savedir = os.path.splitext(savedir)[0]
    self.TraceIters = dict()
    self.SavedIters = dict()
    self.printEvery = argDict['printEvery']
    self.saveEvery = argDict['saveEvery']
    
  def fit(self, hmodel, Data, seed):
    pass
    
  #########################################################  
  #########################################################  Save to file
  #########################################################  
  def save_state( self, hmodel, iterid, evBound, nObs, doFinal=False):
    if self.saveEvery == -1:
      return
    if hasattr(self, 'nObsSoFar'):
      self.nObsSoFar += nObs
    else:
      self.nObsSoFar = nObs

    if iterid not in self.TraceIters:
      if iterid == 0:
        mode = 'w'
      else:
        mode = 'a'
      if doFinal or (iterid % (self.traceEvery)==0):
        self.TraceIters[iterid] = True
        with open( self.savedir+'iters.txt', mode) as f:        
          f.write( '%.3f\n' % (iterid) )
        with open( self.savedir+'evidence.txt', mode) as f:        
          f.write( '%.8e\n' % (evBound) )
        with open( self.savedir+'nObs.txt', mode) as f:
          f.write( '%d\n' % (self.nObs) )
        with open( self.savedir+'times.txt', mode) as f:
          f.write( '%.3f\n' % (time.time()-self.start_time) )

    if iterid in self.SavedIters:
      return
    if doFinal or iterid < 3 or ( iterid % (self.saveEvery)==0 ):
      self.SavedIters[iterid] = True
      prefix = 'Iter%05d'%(iterid)
      ModelWriter.save_model( hmodel, self.savedir, prefix, doSavePriorInfo=(iterid<1) )


  #########################################################  
  #########################################################  Verify evidence
  #########################################################  
  def verify_evidence(self, evBound, prevBound):
    isValid = prevBound < evBound or np.allclose( prevBound, evBound, rtol=self.convTHR )
    if not isValid:
      warnMsg = 'WARNING: evidence decreased!\n' \
        + '    prev = % .15e\n' % (prevBound) \
        + '     cur = % .15e\n' % (evBound)
      Log.severe(warnMsg)
    isConverged = np.abs(evBound-prevBound)/np.abs(evBound) <= self.convTHR
    return isConverged 

  #########################################################  
  #########################################################  Print State
  #########################################################  
  def print_state( self, hmodel, iterid, evBound, doFinal=False, status='', rho=None):
    if self.printEvery <= 0:
      return None
    doPrint = iterid < 3 or iterid % self.printEvery==0
    if not doPrint and doFinal:
      Log.info('... done. %s' % (status))
      return None
      
    if rho is None:
      rhoStr = ''
    else:
      rhoStr = '%.4f |' % (rho)
    logmsg = '  %5d/%d after %6.0f sec. | K %4d | ev % .9e %s'
    logmsg = logmsg % (iterid, 
                        self.Niter, 
                        time.time()-self.start_time,
                        hmodel.K,
                        evBound, 
                        rhoStr)
    
    Log.info(logmsg)
    if doFinal:
      Log.info('... done. %s' % (status))
