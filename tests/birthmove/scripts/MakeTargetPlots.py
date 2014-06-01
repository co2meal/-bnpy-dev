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

import bnpy.viz.BarsViz
#from bnpy.birthmove import BirthMove, TargetPlanner, TargetDataSampler
#from bnpy.allocmodel.admix import OptimizerForHDPStickBreak as OptimHDPSB

Colors = matplotlib.rcParams['axes.color_cycle'][:-1] # exclude black
vmin=0
vmax=0.01

def MakePlotsForTargetDataPlan(Plan, savepath, doSave=1):
  if savepath is not None:
    mkpath(savepath)

  Data = Plan['BigData']
  targetData = Plan['Data']

  if hasattr(Data, 'vocab_dict'):
    Vocab = [str(x[0][0]) for x in Data.vocab_dict]
  else:
    Vocab = None

  V = targetData.vocab_size
  sqrtV = np.sqrt(V)
  if sqrtV == int(np.sqrt(V)):
    exampleDocs, docIDs = dataToWordFreqMatrix(targetData, nExamples=15)
    _plotBarsTopicsSquare(exampleDocs, nRows=3)
    saveCurrentFig(doSave, savepath, 'TargetData.png')

    if 'ScoreMat' in Plan:
      ScoreMat = Plan['ScoreMat'][Plan['candidates']][docIDs]
      _plotBarsTopicsSquare(ScoreMat, nRows=3, cmap='jet', vmin=-0.01, vmax=0.01)
      saveCurrentFig(doSave, savepath, 'TargetDataScore.png')

    if 'targetWordFreq' in Plan and Plan['targetWordFreq'] is not None:
      _plotBarsTopicsSquare(Plan['targetWordFreq'])
      saveCurrentFig(doSave, savepath, 'TargetIcon.png')
    elif Plan['targetWordIDs'] is not None:
      v = np.zeros(targetData.vocab_size)
      v[Plan['targetWordIDs']] = 1
      _plotBarsTopicsSquare(v)
      saveCurrentFig(doSave, savepath, 'TargetIcon.png')

    if 'BigModel' in Plan:
      model = Plan['BigModel']      
      _plotBarsTopicsSquare(modelToTopicWordMatrix(model), Kmax=20, nRows=2)
      saveCurrentFig(doSave, savepath, 'OriginalTopics.png')
  elif not hasattr(Data, 'vocab_dict'):
    if 'BigModel' in Plan:
      model = Plan['BigModel']      
      plotSynth(modelToTopicWordMatrix(model))
      saveCurrentFig(doSave, savepath, 'OriginalTopics.png')

    exampleDocs, docIDs = dataToWordFreqMatrix(targetData, nExamples=15)
    plotSynth(exampleDocs)
    saveCurrentFig(doSave, savepath, 'TargetData.png')

    if 'targetWordFreq' in Plan and Plan['targetWordFreq'] is not None:
      plotSynth(Plan['targetWordFreq'])
      saveCurrentFig(doSave, savepath, 'TargetIcon.png')


  else:
    plotWordCloudsForRandomDocs(targetData, Vocab, savepath, doSave)

    if 'targetWordFreq' in Plan and Plan['targetWordFreq'] is not None:
      _plotTopicWordClouds(Vocab,
                           Plan['targetWordFreq'], 
                           savepath, basename='TargetIcon.png')
    elif Plan['targetWordIDs'] is not None:
      v = np.zeros(targetData.vocab_size)
      v[Plan['targetWordIDs']] = 1
      _plotTopicWordClouds(Vocab, v,
                           savepath, basename='TargetIcon.png')
      saveCurrentFig(doSave, savepath, 'TargetIcon.png')
    

    if 'BigModel' in Plan:
      model = Plan['BigModel']      
      _plotTopicWordClouds(Vocab,
                           modelToTopicWordMatrix(model),
                           savepath, prefix='Original')
      saveCurrentFig(doSave, savepath, 'OriginalTopics.png')


  '''
  if 'targetWordIDs' in targetInfo and targetInfo['targetWordIDs'] is not None:
    _plotTargetWordsProbs(targetInfo['ps'], targetInfo['targetWordIDs'], Vocab)
    pylab.draw()

    saveCurrentFig(doSave, birthmovefile, 'TargetSelection.png')
  elif 'ps' in targetInfo and targetInfo['ps'] is not None:
    _plotTargetCompProbs(targetInfo['ps'])
    saveCurrentFig(doSave, birthmovefile, 'TargetSelection.png')


  matplotlib.rcParams.update({'font.size': 22})
  figH = pylab.figure(num=123)
  plotELBOTrace(Info, style='ro-', label='expanded')
  plotELBOTrace(fwdInfo, style='k.-', label='original')
  pylab.legend(loc='lower right')
  pylab.draw()
  saveCurrentFig(doSave, birthmovefile, 'TraceELBO.png')
  '''
  if doSave:
    pylab.close('all')

def modelToTopicWordMatrix(model):
  K = model.obsModel.K
  topics = np.zeros((K, model.obsModel.comp[0].lamvec.size))
  for k in xrange(K):
    topics[k,:] = model.obsModel.comp[k].lamvec
  topics /= topics.sum(axis=1)[:,np.newaxis]
  return topics

def dataToWordFreqMatrix(Data, nExamples=10, seed=0):
  PRNG = np.random.RandomState(seed)
  nExamples = np.minimum(nExamples, Data.nDoc)
  docIDs = PRNG.choice(Data.nDoc, nExamples, replace=0)
  Examples = Data.select_subset_by_mask(docIDs)
  EmpFreq =  Examples.to_sparse_docword_matrix().toarray()
  EmpFreq /= EmpFreq.sum(axis=1)[:, np.newaxis]
  return EmpFreq, docIDs

def saveCurrentFig(doSave, savepath, basename):
  if doSave:
    savefile = os.path.join(savepath, basename)
    pylab.savefig(savefile, bbox_inches='tight',
                            pad_inches=0,
                            transparent=0)

def LoadOriginalDataModelAndSuffStats(birthmovefile, cachepath):
  if birthmovefile.endswith(os.path.sep):
    birthmovefile= birthmovefile[:-1]
  fparts = birthmovefile.split(os.path.sep)[:-2]
  fparts.append( cachepath.split(os.path.sep)[-1])
  fpath = os.path.sep.join(fparts)
  DUMP = joblib.load(fpath)
  return DUMP['Data'], DUMP['model'], DUMP['SS']

########################################################### Synth Data Viz ###########################################################
def plotSynth(topicsIN, cmap='gray'):
  if topicsIN.ndim == 1:
    topicsIN = topicsIN[np.newaxis,:]
  topics = topicsIN[:, :300]
  K,V = topics.shape

  pylab.figure()
  pylab.imshow(topics, vmin=vmin, vmax=vmax, interpolation='nearest',
                       cmap=cmap, aspect=topics.shape[1]/float(K))
  pylab.xticks([])
  pylab.yticks([])
  pylab.show(block=False)
########################################################### Bars Data Viz ###########################################################

def plotBarsData(Data, savepath=None):
  figID = pylab.figure(figsize=(4,4))
  topics = dataToTopicWordMatrix(Data)
  _plotBarsTopicsSquare(topics)

def _plotBarsTopicsSquare(topicsIN, nRows=1, Kmax=15, 
                          xlabels=list(),
                          cmap='gray', vmax=vmax, vmin=vmin,
                          targetInfo=None):
  if topicsIN.ndim == 1:
    topicsIN = topicsIN[np.newaxis,:]
  K, V = topicsIN.shape
  J = int(np.sqrt(V))

  Kplot = np.minimum(K, Kmax)
  topics = topicsIN[:Kplot].copy()

  nCols = int(np.ceil(Kplot/float(nRows)))
  figsize = (nCols * nRows, 2 * nRows)

  figH, ax = pylab.subplots(nrows=nRows, ncols=nCols, figsize=figsize)
  for k in xrange(Kplot):
    pylab.subplot(nRows, nCols, k+1)
    imSq = np.reshape(topics[k,:], (J,J))
    pylab.imshow(imSq, vmin=vmin, vmax=vmax, interpolation='nearest',
                       cmap=cmap, aspect=1.0)
    pylab.xticks([])
    pylab.yticks([])
    if len(xlabels) > k:
      pylab.xlabel( xlabels[k])
    if targetInfo is not None and targetInfo['ktarget'] is not None:
      if k == targetInfo['ktarget']:
        pylab.title('TARGET', fontsize=10)
  pylab.tight_layout()
  pylab.draw()
  if not pylab.isinteractive():
    pylab.show(block=False)

########################################################### Target Data Viz ###########################################################
def plotWordCloudsForRandomDocs(Data, Vocab, birthmovefile, doSave):
  PRNG = np.random.RandomState(0)
  docIDs = PRNG.choice(Data.nDoc, np.minimum(Data.nDoc, 6), replace=False)
  Temp = Data.select_subset_by_mask(docIDs)
  DWMat = Temp.to_sparse_docword_matrix().toarray()
  DWMat /= DWMat.sum(axis=1)[:,np.newaxis]
  _plotTopicWordClouds(Vocab, DWMat, birthmovefile, 
                          prefix='TargetData')
  pylab.subplot(1,6,1)
  pylab.title('nDocTotal=%d' % (Data.nDoc))
  saveCurrentFig(doSave, birthmovefile, 'TargetData.png')

def _plotTopicWordClouds(VocabList, topics, savepath, ktarget=None,
                         basename=None, prefix='', N=None, W=200, H=300):
  if basename is None:
    basename = prefix + 'WordCloud_%02d.png'
  if topics.ndim == 1:
    topics = topics[np.newaxis,:]

  K, V = topics.shape
  if K <= 10:
    nRows = 1
    nCols = K
    figH = pylab.subplots(nrows=nRows, ncols=nCols, figsize=(K*3,3))
  elif K <= 20:
    nRows = 2
    nCols = K/2
    figH = pylab.subplots(nrows=nRows, ncols=nCols, figsize=(K/2*3,4.5))

  def random_color_func(word, font_size, position, orientation):
    PRNG = np.random.RandomState(hash(word) % 10000)
    return "hsl(%d" % PRNG.randint(0, 100) + ", 80%, 50%)"

  for k in range(K):
    sortedIDs = np.argsort(-1* topics[k,:] )
    sortedWordFreqPairs = [ (VocabList[s], topics[k,s])
                             for s in sortedIDs[:20]]
    elts = wordcloud.fit_words(sortedWordFreqPairs, 
                               prefer_horiz=1.0, width=W, height=H)
    if basename.count('%'):
      savefile = os.path.join(savepath, basename % (k))
    else:
      savefile = os.path.join(savepath, basename)
    wordcloud.draw(elts, savefile, width=W, height=H, scale=2,
                         color_func=random_color_func,
                  )

    pylab.subplot( nRows, nCols, k+1)
    Im = skimage.io.imread(savefile)
    pylab.imshow(Im)
    pylab.xticks([])
    pylab.yticks([])
    if N is not None:
      pylab.xlabel('%.0f' % (N[k]))
    if k == ktarget:
      pylab.title("TARGET")
  pylab.show(block=0)


########################################################### Target Planner Probs ###########################################################
def _plotTargetWordsProbs(ps, wordIDs, Vocab):
  V = ps.size
  sqrtV = np.sqrt(V)
  if sqrtV == int(np.sqrt(V)):
    figH = pylab.subplots(nrows=1, ncols=2, figsize=(8,8))
    shpSq = (sqrtV, sqrtV)
    psSq = np.reshape(ps, shpSq)

    choiceVec = np.zeros(V)
    choiceVec[wordIDs] = 1
    choiceSq = np.reshape(choiceVec, shpSq)
    pylab.subplot(1,2,1)
    pylab.imshow(psSq, interpolation='nearest', vmin=0, vmax=1.0/V)
    pylab.xticks([])
    pylab.yticks([])
    pylab.subplot(1,2,2)
    pylab.imshow(choiceSq, interpolation='nearest', vmin=0, vmax=1.0)
    pylab.xticks([])
    pylab.yticks([])    
  else:
    fH = _plotTargetCompProbs(ps)
    start = 0
    x = 0.15
    y = 0.8
    L = 5
    while start < len(wordIDs):
      wString = ' '.join([Vocab[w] for w in wordIDs[start:start+L]])
      pylab.figtext(x, y, wString)
      start += L
      y -= 0.05
    pylab.draw()

def _plotTargetCompProbs(ps):
    figH = pylab.figure()
    pylab.plot( ps, 'k.-')
    pylab.ylim([-.01, 1.01*ps.max()])
    return figH



def plotELBOTrace(Info, style='.-', label=''):
  pylab.plot(Info['traceELBO'], style, label=label)

def plotTraceStats(Info, Korig=0):
  assert 'traceN' in Info
  assert 'traceBeta' in Info

  figH = pylab.subplots(nrows=1, ncols=2, figsize=(15,6))
  pylab.subplot(1,2,1)
  _plotOrigStuff(Info['traceN'], Korig)
  _plotNewStuff(Info['traceN'], Korig)
  pylab.xlim([-.5, Info['traceN'].shape[0]+.5])
  pylab.ylim([-1, 1.05*Info['traceN'][:, Korig:].max()])
  pylab.title('E[Ntokens] assigned to new topics')

  pylab.subplot(1,2,2)

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


"""
########################################################### main
###########################################################
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('path', default=None)
  parser.add_argument('--doSave', type=int, default=1)
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

  assert os.path.exists(path)

  MakePlotsForSavedBirthMove(path, doSave=args.doSave)

  if not args.doSave:  
    pylab.show(block=False)
    keypress = raw_input('Press any key>>')
"""
