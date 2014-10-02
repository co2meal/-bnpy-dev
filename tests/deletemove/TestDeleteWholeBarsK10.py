'''
'''
import unittest
import joblib
import numpy as np
from matplotlib import pylab

from bnpy.deletemove import runDeleteMove_Whole, DeleteLogger
DeleteLogger.configure(doWriteStdOut=1)

LPkwargs = dict(nCoordAscentItersLP=50, convThrLP=0.001, methodLP='scratch',
              routineLP='simple')

NITER = 5

class TestBarsK10(unittest.TestCase):

  def setUp(self):
    SaveVars = joblib.load('delete.dump')
    self.model = SaveVars['model']
    self.Data = SaveVars['Data']

    self.LP = self.model.calc_local_params(self.Data, **LPkwargs)
    self.SS = self.model.get_global_suff_stats(self.Data, self.LP,
                          doPrecompEntropy=1)
    self.model.update_global_params(self.SS)
    self.curELBO = self.model.calc_evidence(SS=self.SS)

  def testRunForward(self):
    model = self.model.copy()
    traceffwdELBO = list()
    traceffwdELBO.append(self.curELBO)
    for riter in xrange(NITER):
      LP = model.calc_local_params(self.Data, **LPkwargs)
      SS = self.model.get_global_suff_stats(self.Data, LP, doPrecompEntropy=1)
      model.update_global_params(SS)
      ELBO = model.calc_evidence(SS=SS)
      traceffwdELBO.append(ELBO)
    return traceffwdELBO

  def testRunDeleteMoveWhole(self, deleteCompID=13, strategy='reform'):
    newModel, newSS, newLP, newInfo = runDeleteMove_Whole(self.Data, 
                       self.model, self.SS, self.LP, self.curELBO,
                       deleteCompID=deleteCompID,
                       deleteRespStrategy=strategy,
                       nRefineIters=NITER, doQuitEarly=0, LPkwargs=LPkwargs)
    return newInfo['tracepropELBO']


  def testManyComparisons(self, compList=[13], loc='lower right',
                                H=3, W=5.5):
    global LPkwargs
    nrows = len(compList)
    ncols = 2
    ax = None
    ay = None
    pylab.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*W, nrows*H))
    for rowID, deleteCompID in enumerate(compList):
      for colID, LPinit in enumerate(['prior', 'scratch']):
        LPkwargs['methodLP'] = LPinit
        plotID = colID + rowID * ncols
        ax = pylab.subplot(nrows, ncols, plotID+1, sharey=ay, sharex=ax)
        ay = ax
        self.testCompareAndPlot(deleteCompID, block=False, loc=loc)
        if rowID == 0:
          pylab.title(LPinit)
        if colID == 0:
          pylab.ylabel('topicID %d' % (deleteCompID))
        loc=None
      ay = None

    pylab.draw()
    raw_input('>>')

  def testCompareAndPlot(self, deleteCompID=13, block=1, loc='lower right'):
    t_ffwd = self.testRunForward()
    pylab.plot(t_ffwd, 'ko-', label='baseline')
    
    t_renorm = self.testRunDeleteMoveWhole(deleteCompID, 'renorm')
    t_softev = self.testRunDeleteMoveWhole(deleteCompID, 'softev')
    t_softevoverlap = self.testRunDeleteMoveWhole(deleteCompID, 'softevoverlap')
    pylab.plot(t_renorm, 'r^--', label='delete+renorm')
    pylab.plot(t_softev, 'gs:', label='delete+softev')
    pylab.plot(t_softevoverlap, 'bv-', label='delete+softevoverlap')
    if loc is not None:
      pylab.legend(loc='lower right')
    pylab.show(block=block)
