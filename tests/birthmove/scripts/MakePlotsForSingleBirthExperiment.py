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
from bnpy.birthmove import BirthMove, TargetPlanner, TargetDataSampler
from bnpy.allocmodel.admix import OptimizerForHDPStickBreak as OptimHDPSB

wordcloud.FONT_PATH = '/Library/Fonts/Microsoft/Times New Roman.ttf'
Colors = matplotlib.rcParams['axes.color_cycle'][:-1] # exclude black

vmin=0
vmax=0.01

def MakePlotsForSavedBirthMove(birthmovefile, doSave=1):
  if not os.path.exists(birthmovefile):
    birthmovefile = os.path.join(CACHEDIR, birthmovefile)

  assert os.path.exists(birthmovefile)

  matplotlib.rcParams.update({'font.size': 16})

  try:
    fwdInfo = joblib.load(os.path.join(birthmovefile, 'FastForwardResults.dump'))
    Info = joblib.load(os.path.join(birthmovefile, 'BirthResults.dump'))
    Data, model, SS = LoadOriginalDataModelAndSuffStats(birthmovefile,
                                                       Info['cachefile'])

  except IOError:
    print "NO RESULTS FOUND: ", birthmovefile
    return

  Info['SS'] = SS

  targetData = Info['targetData']
  if birthmovefile.count('huffpost') > 0:
    MAT = scipy.io.loadmat('/data/huffpost/huffpost_bnpy.mat')
    Vocab = [str(x[0][0]) for x in MAT['vocab_dict']]
  else:
    Vocab = None

  
  V = targetData.vocab_size
  sqrtV = np.sqrt(V)
  if sqrtV == int(np.sqrt(V)):
    plotBarsData(targetData)
    saveCurrentFig(doSave, birthmovefile, 'TargetData.png')

    plotBarsTopicsSquareBeforeAndAfter(Info, birthmovefile, doSave=doSave)
    #saveCurrentFig(doSave, birthmovefile, 'Topics.png')
  else:
    plotWordCloudsForTopics(Info, Vocab, birthmovefile, doSave)
    plotWordCloudsForRandomDocs(targetData, Vocab, birthmovefile, doSave)

  plotTraceStats(Info, Korig=Info['Korig'])
  saveCurrentFig(doSave, birthmovefile, 'TraceStats.png')
  
  targetInfo = Info['targetInfo']

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

  if doSave:
    pylab.close('all')

def saveCurrentFig(doSave, birthmovefile, basename):
  if doSave:
    savefile = os.path.join(birthmovefile, basename)
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

########################################################### User-facing funcs ###########################################################  for diagnostics
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
  saveCurrentFig(doSave, birthmovefile, 'TargetDataWordCloud.png')


def plotWordCloudsForTopics(Info, Vocab, birthmovefile, doSave):
  Korig = Info['Korig']
  _plotTopicWordClouds(Vocab, Info['initTopics'][:Korig], birthmovefile, 
                          prefix='Orig', ktarget=Info['targetInfo']['ktarget'],
                          N= Info['SS'].N[:Korig])
  saveCurrentFig(doSave, birthmovefile, 'OriginalWordCloud.png')
  _plotTopicWordClouds(Vocab, Info['initTopics'][Korig:], birthmovefile,
                          prefix='Init')
  saveCurrentFig(doSave, birthmovefile, 'InitWordCloud.png')
  _plotTopicWordClouds(Vocab, Info['finalTopics'][Korig:], birthmovefile,
                          prefix='Final')
  saveCurrentFig(doSave, birthmovefile, 'FinalWordCloud.png')

def _plotTopicWordClouds(VocabList, topics, savepath, ktarget=None,
                                    prefix='', N=None, W=200, H=300):
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
    savefile = os.path.join(savepath, '%sWordCloud_%02d.png' % (prefix, k))
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
 



def plotBarsData(Data, savepath=None):
  figID = pylab.figure(figsize=(4,4))
  bnpy.viz.BarsViz.plotExampleBarsDocs(Data, nDocToPlot=25, figID=figID, doShowNow=False)
  pylab.show(block=False)

def _plotBarsTopics(topicsIN, Kmax=15):
  K, V = topicsIN.shape
  topics = np.zeros((Kmax, V))
  if K < Kmax:
    topics[:K] = topicsIN
  else:
    topics[:Kmax] = topicsIN[:Kmax]
  pylab.imshow(topics, vmin=vmin, vmax=vmax, interpolation='nearest',
                       cmap='gray', aspect=float(V)/Kmax)
  pylab.xticks([])
  pylab.yticks([])

def plotBarsTopicsBeforeAndAfter(Info, savepath=None):
  nRows = 1
  nCols = 2
  figID, ax = pylab.subplots(nrows=nRows, ncols=nCols, figsize=(8, 8))
  pylab.subplot( nRows, nCols, 1)
  pylab.title('BEFORE')
  _plotBarsTopics(Info['initTopics'])

  pylab.subplot( nRows, nCols, 2)
  pylab.title('AFTER')
  _plotBarsTopics(Info['finalTopics'])
  pylab.tight_layout()

def plotBarsTopicsSquareBeforeAndAfter(Info, birthmovefile, doSave=0):
  Korig = Info['Korig']

  _plotBarsTopicsSquare(Info['initTopics'][:Korig],
                         targetInfo=Info['targetInfo'])
  saveCurrentFig(doSave, birthmovefile, 'OriginalTopics.png')

  _plotBarsTopicsSquare(Info['initTopics'][Korig:])
  saveCurrentFig(doSave, birthmovefile, 'InitTopics.png')

  _plotBarsTopicsSquare(Info['finalTopics'][Korig:])
  saveCurrentFig(doSave, birthmovefile, 'FinalTopics.png')


def _plotBarsTopicsSquare(topicsIN, Kmax=15, targetInfo=None):
  K, V = topicsIN.shape
  J = int(np.sqrt(V))

  Kplot = np.minimum(K, Kmax)
  topics = topicsIN[:Kplot].copy()

  B = 2
  figsize = (B/2 * Kplot, B)
  figH, ax = pylab.subplots(nrows=1, ncols=Kplot, figsize=figsize)
  for k in xrange(Kplot):
    pylab.subplot(1, Kplot, k+1)
    imSq = np.reshape(topics[k,:], (J,J))
    pylab.imshow(imSq, vmin=vmin, vmax=vmax, interpolation='nearest',
                       cmap='gray', aspect=1.0)
    pylab.xticks([])
    pylab.yticks([])
    if targetInfo is not None and targetInfo['ktarget'] is not None:
      if k == targetInfo['ktarget']:
        pylab.title('TARGET', fontsize=10)
  pylab.tight_layout()
  pylab.draw()
  pylab.show(block=False)

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
