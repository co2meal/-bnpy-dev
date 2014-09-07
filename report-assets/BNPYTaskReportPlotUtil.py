import os
import numpy as np
import glob
import scipy.io
import joblib
import sys

import bnpy

pylab = None
plotly = None
order = None

def ConfigAndSignIntoPlotly(pylabIN, plotlyIN):
  global pylab
  global plotly
  pylab = pylabIN
  plotly = plotlyIN
  plotly.sign_in("mike918", "fr8nzbudjm")


def LoadSingleRunActiveIDsForLap(taskpath, queryLap='final'):
  ''' Load vector of active comp ids for specific single lap

      Essentially reads a single line of the ActiveIDs.txt file from taskpath
  '''
  lappath = os.path.join(taskpath, 'laps.txt')
  laps = np.loadtxt(lappath)

  if queryLap != 'final':
    if queryLap not in laps:
      raise ValueError('Target lap not found.')

  idpath = os.path.join(taskpath, 'ActiveIDs.txt')
  with open(idpath, 'r') as f:
    for ii, curLap in enumerate(laps):
      idstr = f.readline().strip()
      if curLap == queryLap or (curLap == laps[-1] and queryLap == 'final'):
        idvec = np.asarray(idstr.split(' '), dtype=np.int32)
        return idvec
  

def LoadSingleRunCounts(taskpath, doSort=True):
  idpath = os.path.join(taskpath, 'ActiveIDs.txt')
  ctpath = os.path.join(taskpath, 'ActiveCounts.txt')
  fid = open(idpath, 'r')
  fct = open(ctpath, 'r')
  data = list()
  colids = list()
  rowids = list()
  for ii, idline in enumerate(fid.readlines()):
    idstr = str(idline.strip())
    ctstr = str(fct.readline().strip())
    idvec = np.asarray(idstr.split(' '), dtype=np.int32)
    ctvec = np.asarray(ctstr.split(' '), dtype=np.float)
    data.extend(ctvec)
    colids.extend(idvec)
    rowids.extend( ii * np.ones(idvec.size))

  ## Make SparseMatrix of Counts over time
  ## Each row of Counts gives count (or zero if dead/inactive) 
  ## at single lap
  data = np.asarray(data)
  data += 1e-9
  ij = np.vstack([rowids, colids])
  Counts = scipy.sparse.csr_matrix((data, ij))
  Counts = Counts.toarray()

  if not doSort:
    return Counts

  ## Sort columns from biggest to smallest (at last chkpt)
  sortIDs = np.argsort(-1*Counts[-1,:])
  badIDs = np.flatnonzero(Counts[-1, :] < 1e-9)
  nGood = Counts.shape[1] - len(badIDs)

  SCounts = np.zeros_like(Counts)
  SCounts[:, :nGood] = Counts[:, sortIDs[:nGood]]
  
  rankIDs = np.argsort(-1*np.sum(Counts[:, badIDs], axis=0))
  SCounts[:, nGood:] = Counts[:, badIDs[rankIDs]]

  global order
  order = np.hstack([sortIDs[:nGood], badIDs[rankIDs]])
  assert len(np.unique(order)) == SCounts.shape[1]
  return SCounts

Colors = ['b','r','g','m','c','k','y']
Styles = ['.-', 'o--', 'x:']

def PlotSingleRunELBOAndK(taskpath):
  pylab.subplots(nrows=1, ncols=2) #, figsize=(6, 2))
  pylab.subplot(1, 2, 1)
  PlotSingleRunELBO(taskpath)
  pylab.subplot(1, 2, 2)
  PlotSingleRunTruncationLevel(taskpath)

  pylab.tight_layout()

def PlotSingleRunELBO(taskpath):
  ''' Plot ELBO for single run
  '''
  lappath = os.path.join(taskpath, 'laps.txt')
  laps = np.loadtxt(lappath)

  elbopath = os.path.join(taskpath, 'evidence.txt')
  objFunc = np.loadtxt(elbopath)

  pylab.plot(laps, objFunc, '.-')
  pylab.ylabel('log evidence', fontsize=14)
  pylab.xlabel('laps thru training data', fontsize=14)


def PlotSingleRunCounts(taskpath, compsToShow=None):
  ''' Plot active component counts for single run
  '''
  lappath = os.path.join(taskpath, 'laps.txt')
  laps = np.loadtxt(lappath)

  Counts = LoadSingleRunCounts(taskpath)

  if compsToShow is not None:
    Counts = Counts[:, compsToShow]
    pylab.plot(laps, Counts, '.-')

  else:
    global order
    if order is None:
      order = np.arange(Counts.shape[1])

    import bnpy.viz.GaussViz
    Colors = bnpy.viz.GaussViz.Colors

    for ii, _ in enumerate(order):
      color = Colors[ii % len(Colors)]
      pylab.plot(laps, Counts[:, ii], '.-', color=color)

  pylab.ylabel('usage count', fontsize=14)
  pylab.xlabel('laps thru training data', fontsize=14)

def PlotSingleRunTruncationLevel(taskpath, activeThr=0.00001):
  lappath = os.path.join(taskpath, 'laps.txt')
  laps = np.loadtxt(lappath)
  Counts = LoadSingleRunCounts(taskpath)
  
  Kactive = np.sum(Counts > activeThr, axis=1)

  pylab.plot(laps, Kactive, '.-')
  pylab.ylabel('num active components', fontsize=14)
  pylab.xlabel('laps thru training data', fontsize=14)


def PlotSingleRunComps(taskpath, lap=None, MaxKToDisplay=50, **kwargs):
  ''' Show the learned components for a single algorithm run
  '''
  global order

  if lap is None:
    model = bnpy.load_model(taskpath)    
    activeIDs = LoadSingleRunActiveIDsForLap(taskpath, queryLap='final')
  else:
    model, lap = bnpy.ioutil.ModelReader.loadModelForLap(taskpath, lap)
    activeIDs = LoadSingleRunActiveIDsForLap(taskpath, queryLap=lap)

  if str(type(model.obsModel)).count('Gauss'):
    from bnpy.viz import GaussViz
    GaussViz.pylab = pylab
    GaussViz.plotGauss2DFromHModel(model, 
                                   activeCompIDs=activeIDs,
                                   compListToPlot=order,
                                   MaxKToDisplay=MaxKToDisplay, 
                                   **kwargs)
  else:
    from bnpy.viz import BarsViz
    BarsViz.pylab = pylab
    BarsViz.plotBarsFromHModel(model, 
                                activeCompIDs=activeIDs,
                                compListToPlot=order,
                                Kmax=MaxKToDisplay, 
                                **kwargs)

if __name__ == "__main__":
  from matplotlib import pylab
  pylab.ion()
  #taskpath = '/results/AdmixAsteriskK8/mytest-K80-random/1/'
  taskpath = "/results/MixBarsK10V900/mytest-K100-random/1"
  PlotSingleRunCounts(taskpath)

  #pylab.figure()
  PlotSingleRunComps(taskpath, lap=2)

  pylab.show()

