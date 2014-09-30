import argparse
import matplotlib
from matplotlib import pylab
import numpy as np
import os
import sys
import joblib
from distutils.dir_util import mkpath
import wordcloud
import skimage.io
import scipy.io

import bnpy
from bnpy.allocmodel.admix import OptimizerForHDPStickBreak as OptimHDPSB

import MakeTargetPlots as MTP

def MakePlots(BirthResults, CurResults, Data, outpath, doSave=1):

  Korig = BirthResults['Korig']
  plotTraceStats(BirthResults, Korig=Korig)
  saveCurrentFig(doSave, outpath, 'TraceStats.png')

  plotELBOTrace(BirthResults, CurResults)
  saveCurrentFig(doSave, outpath, 'ELBO.png')

  if hasattr(Data, 'vocab_dict'):
    Vocab = [str(x[0][0]) for x in Data.vocab_dict]
  else:
    Vocab = None

  V = Data.vocab_size
  sqrtV = np.sqrt(V)
  if sqrtV == int(np.sqrt(V)):
    MTP._plotBarsTopicsSquare(BirthResults['initTopics'][Korig:])
    saveCurrentFig(doSave, outpath, 'InitTopics.png')

    MTP._plotBarsTopicsSquare(BirthResults['finalTopics'][Korig:])
    saveCurrentFig(doSave, outpath, 'FinalTopics.png')

    if 'cleanupTopics' in BirthResults:
      MTP._plotBarsTopicsSquare(BirthResults['cleanupTopics'][Korig:])
      saveCurrentFig(doSave, outpath, 'MinimalFinalTopics.png')
  else:
    MTP._plotTopicWordClouds(Vocab, BirthResults['initTopics'][Korig:],
                              outpath,
                              basename='InitTopics.png')

    MTP._plotTopicWordClouds(Vocab, BirthResults['finalTopics'][Korig:],
                              outpath,
                              basename='FinalTopics.png')

    if 'cleanupTopics' in BirthResults:
      cleanupTopics =  BirthResults['cleanupTopics'][Korig:]
      if cleanupTopics.shape[0] > Korig:
        MTP._plotTopicWordClouds(Vocab, cleanupTopics,
                              outpath,
                              basename='MinimumFinalTopics.png')

def saveCurrentFig(doSave, outpath, basename):
  if doSave:
    savefile = os.path.join(outpath, basename)
    pylab.savefig(savefile, bbox_inches='tight',
                            pad_inches=0,
                            transparent=0)

def plotELBOTrace(BirthResults, CurResults):
  pylab.figure(figsize=(6,4))
  if 'ELBOPostDelete' in BirthResults:
    xs = np.arange(len(BirthResults['traceELBO']))
    ys = BirthResults['ELBOPostDelete'] * np.ones(xs.size)
    pylab.plot(xs, ys, 'r--', label='after deletes')
  pylab.plot(BirthResults['traceELBO'], 'ro-', label='expanded')
  pylab.plot(CurResults['traceELBO'], 'b+-', label='original')
  pylab.legend(loc='lower right')
  pylab.draw()
  pylab.show(block=0)


def plotTraceStats(Info, Korig=0):
  assert 'traceN' in Info
  assert 'traceBeta' in Info

  figH = pylab.subplots(nrows=1, ncols=2, figsize=(15,6))
  pylab.subplot(1,2,1)
  _plotOrigStuff(Info['traceN'], Korig)
  _plotNewStuff(Info['traceN'], Korig)
  pylab.xlim([-.5, Info['traceN'].shape[0]+.5])

  MaxN = Info['traceN'][:, Korig:].max()
  pylab.ylim([-0.05*MaxN, 1.05*MaxN])
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
 