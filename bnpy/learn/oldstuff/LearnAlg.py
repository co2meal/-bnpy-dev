'''
 Abstract class for learning algorithms for HModel models

  Simply defines some generic routines for
    ** initialization (see the init module)
    ** saving global parameters
    ** assessing convergence
    ** printing progress updates to stdout

Author: Mike Hughes (mike@michaelchughes.com)
'''
from IPython import embed
import numpy as np
import time
import os

from bnpy.ioutil import ModelWriter

class LearnAlg(object):

  def __init__( self, savefilename=os.environ['BNPYOUTDIR'], nIter=100, \
                    initname='randsample',  convTHR=1e-10, \
                    printEvery=5, saveEvery=5, traceEvery=1, \
                    doVerify=False, \
                    rhodelay=1, rhoexp=0.6, \
                    **kwargs ):
    self.savefilename = savefilename
    self.initname = initname
    self.convTHR = convTHR
    self.Niter = nIter
    self.printEvery = printEvery
    self.saveEvery = saveEvery
    self.traceEvery = traceEvery
    self.TraceIters = dict()
    self.SavedIters = dict()
    self.doVerify = doVerify
    self.rhodelay =rhodelay
    self.rhoexp   = rhoexp
    self.saveext = 'mat'

  def fit( self, hmodel, Data, seed):
    pass
    
  def init_global_params( self, hmodel, Data, seed, **kwargs ):
    initmsg = '  Initialized params via %s. Seed %d.' % (self.initname, seed)
    obsType = hmodel.obsModel.get_info_string()
    self.seed = seed
    if obsType.count('Gauss') > 0:
      datamsg = 'Init Data: %d examples' % (Data['nObs'])
      InitEngine = GaussObsSetInitializer( initname=self.initname, seed=seed)
      InitEngine.init_global_params( hmodel, Data, **kwargs )      
    else:
      pass
    if self.printEvery > 0:
      print initmsg, datamsg
      

  #########################################################  
  #########################################################  Save to file
  #########################################################  
  def save_state( self, hmodel, iterid, evBound, nObs, doFinal=False):
    if self.saveEvery == -1:
      return
    if hasattr(self, 'nObs'):
      self.nObs += nObs
    else:
      self.nObs = nObs

    fname, ext = os.path.splitext( self.savefilename )
    if iterid not in self.TraceIters:
      if iterid == 0:
        mode = 'w'
      else:
        mode = 'a'
      if doFinal or ( iterid % (self.traceEvery)==0 ):
        self.TraceIters[iterid] = True
        with open( fname+'iters.txt', mode) as f:        
          f.write( '%d\n' % (iterid) )
        with open( fname+'evidence.txt', mode) as f:        
          f.write( '%.8e\n' % (evBound) )
        with open( fname+'nObs.txt', mode) as f:
          f.write( '%d\n' % (self.nObs) )
        with open( fname+'times.txt', mode) as f:
          f.write( '%.3f\n' % (time.time()-self.start_time) )

      if hasattr(self, 'moves') and (doFinal or iterid % (self.nBatch)==0 ):
        if 'b' in self.moves:
          with open( fname+'births.txt', mode) as f:
            f.write( '%d %d\n' % (self.nBirthTotal, self.nBirthTotalTry) )
        if 'm' in self.moves or 'M' in self.moves:
          with open( fname+'merges.txt', mode) as f:
            f.write( '%d %d\n' % (self.nMergeTotal, self.nMergeTotalTry) )
        if hasattr(self, 'MoveLog'):
          import shelve
          SDict = shelve.open( fname+'MoveLog.shelve' )
          PickleLog = dict( [k for k in self.MoveLog.items() ] )
          if 'lasttryiter' in PickleLog:
            PickleLog['lasttryiter'] = dict([ k for k in self.MoveLog['lasttryiter'].items()])
          SDict['MoveLog'] = PickleLog
          SDict.close()

    if iterid in self.SavedIters:
      return
    if doFinal or iterid < 2 or ( iterid % (self.saveEvery)==0 ):
      self.SavedIters[iterid] = True
      prefix = 'Iter%05d'%(iterid)
      ModelWriter.save_model( hmodel, fname, prefix, doSavePriorInfo=(iterid==1) )


  #########################################################  
  #########################################################  Verify evidence
  #########################################################  
  def verify_evidence(self, evBound, prevBound):
    isValid = prevBound < evBound or np.allclose( prevBound, evBound, rtol=self.convTHR )
    if not isValid:
      print 'WARNING: evidence decreased!'
      print '    prev = % .15e' % (prevBound)
      print '     cur = % .15e' % (evBound)
    isConverged = np.abs(evBound-prevBound)/np.abs(evBound) <= self.convTHR
    return isConverged 

  #########################################################  
  #########################################################  Print State
  #########################################################  
  def print_state( self, hmodel, iterid, evBound, doFinal=False, status='', rho=None):
    if self.printEvery <= 0:
      return None
    doPrint = iterid < 3 or iterid % self.printEvery==0
    doPrint = (doPrint and not doFinal) or ( not doPrint and doFinal)
    if not doPrint:
      if doFinal:
        print '... done. %s' % (status)
      return None
      
    if rho is None:
      rhoStr = ''
    else:
      rhoStr = '%.4f |' % (rho)
    logmsg = '  %5d/%s after %6.0f sec. | %s evidence % .9e'
    logmsg = logmsg % (iterid, str(self.Niter), time.time()-self.start_time,rhoStr, evBound)
    
    if hasattr(self, 'moves'):
      logmsg += ' | K = %4d ' % (hmodel.K)
      if 'm' in self.moves or 'M' in self.moves:
        logmsg += ' | %2d/%2d merges ' % (self.nMerge, self.nMergeTry)
        self.nMergeTotal += self.nMerge
        self.nMergeTotalTry += self.nMergeTry
        self.nMerge = 0; self.nMergeTry=0;
      if 's' in self.moves:
        logmsg += ' | %2d/%2d splits ' % (self.nSplit, self.nSplitTry)
        self.nSplitTotal += self.nSplit
        self.nSplitTotalTry += self.nSplitTry
        self.nSplit = 0; self.nSplitTry=0;
      if 'b' in self.moves:
        logmsg += ' | %2d/%2d birth ' % (self.nBirth, self.nBirthTry)
        self.nBirthTotal += self.nBirth
        self.nBirthTotalTry += self.nBirthTry
        self.nBirth = 0; self.nBirthTry=0;
      if 'd' in self.moves:
        logmsg += ' | %2d/%2d death' % (self.nDeath, self.nDeathTry)
        self.nDeathTotal += self.nDeath
        self.nDeathTotalTry += self.nDeathTry
        self.nDeath = 0; self.nDeathTry=0;
    print logmsg
    if doFinal:
      print '... done. %s' % (status)
