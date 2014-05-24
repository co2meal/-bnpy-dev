import argparse
from matplotlib import pylab
import numpy as np
import os
import sys
import joblib
from distutils.dir_util import mkpath

import bnpy
from bnpy.birthmove import BirthMove, TargetPlanner, TargetDataSampler
from bnpy.allocmodel.admix import OptimizerForHDPStickBreak as OptimHDPSB

def MakePlotsForSavedBirthMove(birthmovefile, savepath=None):
  if not os.path.exists(birthmovefile):
    birthmovefile = os.path.join(CACHEDIR, birthmovefile)

  assert os.path.exists(birthmovefile)

  FFWD = joblib.load(os.path.join(birthmovefile, 'FastForwardResults.dump'))
  fwdInfo = FFWD['fwdInfo']

  Q = joblib.load(os.path.join(birthmovefile, 'BirthResults.dump'))
  xmodel = Q['xmodel']
  targetData = Q['targetData']
  Info = Q['Info']

  QQ = joblib.load(Q['cachefile'])
  bigData = QQ['Data']
  model = QQ['model']
  bigSS = QQ['SS']
  
  if hasattr(bigData, 'TrueParams'):
    plotBarsDataAndSaveToDisk(bigData, 'ExampleDocs-BigData.png', figID=1)
    plotBarsDataAndSaveToDisk(targetData, 'ExampleDocs-TargetData.png', figID=2)
    plotBarsTopicsBeforeAndAfter(model, xmodel, Info, Kmax=15)
  
  plotTraceStats(Info, Korig=bigSS.K)
 
  figH = pylab.figure(num=123)
  plotELBOTrace(Info, style='ro-', label='expanded')
  plotELBOTrace(fwdInfo, style='k.-', label='original')
  pylab.legend(loc='lower right')

  pylab.draw()
  if savepath is not None:
    savefile = os.path.join(savepath, 'traceELBO.png')
    pylab.savefig(savefile, bbox='tight')

########################################################### User-facing funcs ###########################################################  for diagnostics
def plotELBOTrace(Info, style='.-', label=''):
  pylab.plot(Info['traceELBO'], style, label=label)

def plotTraceStats(Info, Korig=0):
  assert 'traceN' in Info
  assert 'traceBeta' in Info

  figH = pylab.subplots(nrows=1, ncols=2)
  pylab.subplot(1,2,1)
  _plotOrigStuff(Info['traceN'], Korig)
  _plotNewStuff(Info['traceN'], Korig)
  pylab.xlim([-.5, Info['traceN'].shape[0]+.5])
  pylab.ylim([-1, 1.05*Info['traceN'][:, Korig:].max()])
  pylab.title('E[Ntokens] assigned to new topics')

  pylab.subplot(1,2,2)
  #_plotOrigStuff(Info['traceBeta'], Korig)
  _plotNewStuff(Info['traceBeta'], Korig)
  pylab.xlim([-.5, Info['traceBeta'].shape[0]+.5])
  pylab.ylim([-.0001, 1.05*Info['traceBeta'][:, Korig:].max()])
  pylab.title('E[beta] new topics')

  pylab.show(block=False)


def _plotOrigStuff( Trace, Korig=0):
  for k in xrange(Korig):
    pylab.plot( Trace[:, k], 'k.--')

def _plotNewStuff(Trace, Korig=0):
  for k in xrange(Korig, Trace.shape[1]):
    pylab.plot( Trace[:,k], 'o-') 
 


def pprint_accept_decision(Data, dataName, model, xmodel, Info):
  curELBO = model.calc_evidence(Data)
  propELBO = xmodel.calc_evidence(Data)
  fresh0ELBO = Info['freshInfo']['freshModelInit'].calc_evidence(Data)
  freshELBO = Info['freshInfo']['freshModelPostDelete'].calc_evidence(Data)

  print 'Using %s, should we accept? ' % (dataName)
  print '  fresh init? %d' % (fresh0ELBO >= curELBO)
  print '  fresh post? %d' % (freshELBO >= curELBO)
  print '      expand? %d' % (propELBO >= curELBO)

def plotELBOTraces(Data):
  pass
  #figID, ax = pylab.subplots(3, 1, sharex=1, figsize=(7.0, 10.0))
  #pylab.subplot(3,1,1)
  #plotELBOTraces(targetData, 'target D=%d' % (targetData.nDoc), model, Info)
  #pylab.subplot(3,1,2)
  #plotELBOTraces(holdData, 'heldout D=%d' % (holdData.nDoc), model, Info)
  #pylab.subplot(3,1,3)
  #plotELBOTraces(Data, 'all D=%d' % (Data.nDoc), model, Info)
  #pylab.legend(loc='lower right')

def plotBarsDataAndSaveToDisk(Data, titleStr='', figID=100, savepath=None):
  figID = pylab.figure(figID)
  bnpy.viz.BarsViz.plotExampleBarsDocs(Data, nDocToPlot=25, figID=figID, doShowNow=False)
  if savepath is not None and len(titleStr) > 0:
    savefile = os.path.join(savepath, titleStr)
    pylab.savefig(savefile, bbox='tight')
  pylab.show(block=False)

def plotBarsTopicsAndSaveToDisk(model, Kmax=None, savepath=None):
  bnpy.viz.BarsViz.plotBarsFromHModel(model, Kmax=Kmax)

def plotBarsTopicsBeforeAndAfter(model, xmodel, Info, Kmax=None, savepath=None):
  nRows = 2
  nCols = 2
  figID, ax = pylab.subplots(nrows=nRows, ncols=nCols, figsize=(8, 8))
  bnpy.viz.BarsViz.plotBarsFromHModel(model, Kmax=Kmax,
                                      figH=pylab.subplot(nRows,nCols,1))
  pylab.title('BEFORE')
  bnpy.viz.BarsViz.plotBarsFromHModel(xmodel, Kmax=Kmax,
                                     figH=pylab.subplot(nRows,nCols,2))
  pylab.title('AFTER')

  bnpy.viz.BarsViz.plotBarsFromHModel(Info['xbigModelInit'],
                                      Kmax=Kmax,
                                      figH=pylab.subplot(nRows,nCols,3))
  pylab.title('Expanded INIT')
  
  bnpy.viz.BarsViz.plotBarsFromHModel(Info['xbigModelRefined'],
                                      Kmax=Kmax,
                                      figH=pylab.subplot(nRows,nCols,4))
  pylab.title('Expanded CLEAN')

  figID.tight_layout()
  if savepath is not None:
    savefile = os.path.join(savepath, 'topics.png')
    pylab.savefig(savefile, bbox='tight')


########################################################### main
###########################################################
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('path', default=None)
  parser.add_argument('--data', default='BarsK10V900')
  parser.add_argument('--initName', default='K1')
  parser.add_argument('--jobName', type=str, default='')
  parser.add_argument('--task', type=int, default=1)
  args, unkList = parser.parse_known_args()
  kwargs = bnpy.ioutil.BNPYArgParser.arglist_to_kwargs(unkList)

  path = args.path
  if path is None or not os.path.exists(path):
    import RunSingleBirthExperiment as RBE
    outPath2 = RBE.createOutPath(args)
    if os.path.exists(outPath2):
      path = outPath2

  print path
  assert os.path.exists(path)

  MakePlotsForSavedBirthMove(path)
  
  pylab.show(block=False)
  keypress = raw_input('Press any key>>')
