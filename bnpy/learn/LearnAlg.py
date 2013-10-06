'''
LearnAlg.py
Abstract base class for learning algorithms for HModel models

Defines some generic routines for
  * saving global parameters
  * assessing convergence
  * printing progress updates to stdout
  * recording run-time
'''
import numpy as np
import time
import os
import logging
import scipy.io
from distutils.dir_util import mkpath

from bnpy.ioutil import ModelWriter

Log = logging.getLogger('bnpy')
Log.setLevel(logging.DEBUG)

class LearnAlg(object):

  def __init__(self, savedir=None, seed=0, algParams=dict(), outputParams=dict()):
    self.savedir = os.path.splitext(savedir)[0]
    self.PRNG = np.random.RandomState(seed)
    self.algParams = algParams
    self.outputParams = outputParams
    self.TraceIters = set()
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

  #########################################################  
  #########################################################  Verify evidence monotonic
  #########################################################  
  def verify_evidence(self, evBound=0.00001, prevBound=0, EPS=1e-9):
    ''' Compare current and previous evidence (ELBO) values,
        and verify that (within numerical tolerance) evidence increases monotonically
    '''
    if np.isnan(evBound):
      raise ValueError("Evidence should never be NaN")
    if np.isinf(prevBound):
      return False
    isIncreasing = prevBound <= evBound
    absDiff = np.abs(prevBound - evBound)
    percDiff = absDiff / np.abs(prevBound)

    convergeTHR = self.algParams['convergeTHR']
    isWithinTHR = absDiff <= convergeTHR or percDiff <= convergeTHR
    if not isIncreasing:
      if not isWithinTHR:
        warnMsg = 'WARNING: evidence decreased!\n' \
          + '    prev = % .15e\n' % (prevBound) \
          + '     cur = % .15e\n' % (evBound)
        Log.error(warnMsg)
    return isWithinTHR 


  #########################################################  
  #########################################################  Save to file
  #########################################################      
  def save_state( self, hmodel, iterid, lap, evBound, doFinal=False):
    ''' Save state of the hmodel's global parameters and evBound'''
    saveEvery = self.outputParams['saveEvery']
    traceEvery = self.outputParams['traceEvery']
    
    if saveEvery <= 0:
      return    

    def mkfile(fname):
      ''' Create valid path to file in this alg's output directory 
      '''
      return os.path.join(self.savedir, fname)

    doTrace = np.allclose(lap % traceEvery, 0) or iterid < 3
    if iterid not in self.TraceIters:
      if iterid == 0:
        mode = 'w'
      else:
        mode = 'a'
      if doFinal or doTrace:
        self.TraceIters.add(iterid)
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

    if iterid not in self.SavedIters:
      if doFinal or iterid < 3 or np.allclose(lap % saveEvery, 0):
        self.SavedIters.add(iterid)
        prefix = 'Iter%05d'%(iterid)
        ModelWriter.save_model(hmodel, self.savedir, prefix,
                                doSavePriorInfo=(iterid<1), doLinkBest=True)

  #########################################################  
  #########################################################  Print State
  #########################################################  
  def print_state( self, hmodel, iterid, lap, evBound, doFinal=False, status='', rho=None):
    printEvery = self.outputParams['printEvery']
    if printEvery <= 0:
      return None
    doPrint = iterid < 3 or np.allclose(lap % printEvery, 0, atol=1e-8)
      
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
    if (doFinal or doPrint) and iterid not in self.PrintIters:
      self.PrintIters.add(iterid)
      Log.info(logmsg)
    if doFinal:
      Log.info('... done. %s' % (status))
      
  def calc_posterior_exp(self, amodel, Data, LP):
    N = Data.N
    EY_Beta = np.zeros((N,N))
    EY_SR = np.zeros((N,N))
    sigmas = LP["sigmas"]
    #rhos = LP["rhos"]
    beta = amodel.allocModel.PostBeta
    EB = beta / beta.sum()
    EO = amodel.obsModel.lambda_a / (amodel.obsModel.lambda_a + amodel.obsModel.lambda_b)
    for i in xrange( N ):
        for j in xrange( N ):
            EY_Beta[i,j] = np.dot(np.dot(EB.T, EO), EB )
            #EY_SR[i,j] = np.dot(np.dot(sigmas[:,i],EO), rhos[:,j].T )
            EY_SR[i,j] = np.dot(np.dot(sigmas[:,i],EO), sigmas[:,j].T )
    return dict(EY_Beta = EY_Beta, EY_SR = EY_SR, sigmas=sigmas, beta=beta, EB=EB, EO=EO, X=Data.X)

  def save_expectations(self, amodel, iterid, Data, LP):
    fname = self.savedir
    if not os.path.exists( fname):
        mkpath( fname )
    amatname = 'gen_eY.mat'
    outmatfile = os.path.join( fname, amatname )
    myDict = self.calc_posterior_exp(amodel, Data, LP)
    scipy.io.savemat( outmatfile, myDict, oned_as='row')
