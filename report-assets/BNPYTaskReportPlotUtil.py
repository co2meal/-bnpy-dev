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
  pylab.subplots(nrows=1, ncols=2, figsize=(10, 2))
  pylab.subplot(1, 2, 1)
  PlotSingleRunELBO(taskpath)
  pylab.subplot(1, 2, 2)
  PlotSingleRunTruncationLevel(taskpath)

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


def PlotSingleRunComps(taskpath, lap=None, doSort=True, **kwargs):
  ''' Show the learned components for a single algorithm run
  '''
  global order

  from bnpy.viz import GaussViz
  GaussViz.pylab = pylab

  Counts = LoadSingleRunCounts(taskpath, doSort=False)

  if lap is None:
    model = bnpy.load_model(taskpath)    
    activeIDs = np.flatnonzero(Counts[-1, :])
  else:
    model, lap = bnpy.ioutil.ModelReader.loadModelForLap(taskpath, lap)
    lappath = os.path.join(taskpath, 'laps.txt')
    laps = np.loadtxt(lappath)
    rowID = np.flatnonzero(laps == lap)
    activeIDs = np.flatnonzero(Counts[rowID, :])

  if order is not None:
    myorder = list()
    for compID in order:
      for ii, activeID in enumerate(activeIDs):
        if activeID == compID:
          myorder.append(ii)

  model.obsModel.Post.reorderComps(myorder)
  GaussViz.plotGauss2DFromHModel(model, **kwargs)

if __name__ == "__main__":
  utilpath = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-1])
  sys.path.append(utilpath)

  #from matplotlib import pylab
  #pylab.ion()
  #Counts = PlotSingleRunCounts("/data/liv/liv-x/patch-models/results/bnpy/AdmixAsteriskK8/HDPFast/Gauss/moVB/defaultjob/1/")

  #pylab.figure()
  #Counts = PlotSingleRunComps("/data/liv/liv-x/patch-models/results/bnpy/AdmixAsteriskK8/HDPFast/Gauss/moVB/defaultjob/1/", lap=10)
  #pylab.show()

