'''
BarsViz.py

Visualization tools for toy bars data for topic models.
'''
from matplotlib import pylab
import numpy as np

imshowArgs = dict(interpolation='nearest', 
                  cmap='bone', 
                  vmin=0.0, 
                  vmax=0.1)

def plotExampleBarsDocs(Data, docIDsToPlot=None, figID=None,
                              vmax=None, nDocToPlot=16, doShowNow=False,
                              seed=0, randstate=np.random.RandomState(0)):
    if seed is not None:
      randstate = np.random.RandomState(seed)
    if figID is None:
      pylab.figure()
    V = Data.vocab_size
    sqrtV = int(np.sqrt(V))
    assert np.allclose(sqrtV * sqrtV, V)
    if docIDsToPlot is not None:
      nDocToPlot = len(docIDsToPlot)
    else:
      size = np.minimum(Data.nDoc, nDocToPlot)
      docIDsToPlot = randstate.choice(Data.nDoc, size=size, replace=False)
    nRows = np.floor(np.sqrt(nDocToPlot))
    nCols = np.ceil(nDocToPlot / nRows)
    if vmax is None:
      DocWordArr = Data.getDocTypeCountMatrix()
      vmax = int(np.max(np.percentile(DocWordArr, 98, axis=0)))

    for plotPos, docID in enumerate(docIDsToPlot):
        start = Data.doc_range[docID]
        stop = Data.doc_range[docID+1]
        wIDs = Data.word_id[start:stop]
        wCts = Data.word_count[start:stop]
        docWordHist = np.zeros(V)
        docWordHist[wIDs] = wCts
        squareIm = np.reshape(docWordHist, (np.sqrt(V), np.sqrt(V)))

        pylab.subplot(nRows, nCols, plotPos+1)
        pylab.imshow(squareIm, interpolation='nearest', vmin=0, vmax=vmax)
        pylab.axis('image')
        pylab.xticks([])
        pylab.yticks([])
    pylab.tight_layout()
    if doShowNow:
      pylab.show()

def plotBarsFromHModel(hmodel, Data=None, doShowNow=False, figH=None,
                       doSquare=1,
                       compsToHighlight=None, compListToPlot=None,
                       activeCompIDs=None,  Kmax=None,
                       width=6, height=3, vmax=None):    
  if hasattr(hmodel.obsModel, 'Post'):
    lam = hmodel.obsModel.Post.lam
    topics = lam / lam.sum(axis=1)[:,np.newaxis]
  else:
    topics = hmodel.obsModel.EstParams.phi.copy()

  ## Determine intensity scale for topic-word image
  global imshowArgs
  if vmax is not None:
    imshowArgs['vmax'] = vmax
  else:
    imshowArgs['vmax'] = 1.5 * np.percentile(topics, 95)

  if doSquare:
    figH = showTopicsAsSquareImages(topics, 
                                    activeCompIDs=activeCompIDs,
                                    compsToHighlight=compsToHighlight,
                                    compListToPlot=compListToPlot,
                                    Kmax=Kmax,
                                    **imshowArgs)
  else:
    if figH is None:
      figH = pylab.figure(figsize=(width,height))
    else:
      pylab.axes(figH)
    showAllTopicsInSingleImage(topics, compsToHighlight, **imshowArgs)
  if doShowNow:
    pylab.show()
  return figH

def showTopicsAsSquareImages(topics, 
                             activeCompIDs=None,
                             compsToHighlight=None,
                             compListToPlot=None,
                             Kmax=50,
                             W=1, H=1, **imshowArgs):
  K, V = topics.shape
  sqrtV = int(np.sqrt(V))
  assert np.allclose(sqrtV, np.sqrt(V))

  if compListToPlot is None:
    compListToPlot = np.arange(0, K)
  if activeCompIDs is None:
    activeCompIDs = np.arange(0, K)
  compsToHighlight = np.asarray(compsToHighlight)
  if compsToHighlight.ndim == 0:
    compsToHighlight = np.asarray([compsToHighlight])

  ## Create Figure
  Kplot = np.minimum(len(compListToPlot), Kmax)
  ncols = 5 #int(np.ceil(Kplot / float(nrows)))
  nrows = int(np.ceil(Kplot / float(ncols)))
  figH, ha = pylab.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*W,nrows*H))
  
  for plotID, compID in enumerate(compListToPlot):
    if plotID >= Kmax:
      print 'DISPLAY LIMIT EXCEEDED. Showing %d/%d components' \
             % (plotID, len(activeCompIDs))      
      break

    if compID not in activeCompIDs:
      aH = pylab.subplot(nrows, ncols, plotID+1)
      aH.axis('off')
      continue

    kk = np.flatnonzero(compID == activeCompIDs)[0]
    topicIm = np.reshape(topics[kk,:], (sqrtV, sqrtV))

    ax = pylab.subplot(nrows, ncols, plotID+1)
    pylab.imshow(topicIm, aspect=1.0, **imshowArgs)
    pylab.xticks([])
    pylab.yticks([])

    ## Draw colored border around highlighted topics
    if compID in compsToHighlight:
      [i.set_color('green') for i in ax.spines.itervalues()]
      [i.set_linewidth(3) for i in ax.spines.itervalues()]

  ## Disable empty plots!
  for kdel in xrange(plotID+2, nrows*ncols+1):
    aH = pylab.subplot(nrows, ncols, kdel)
    aH.axis('off')

  ## Fix margins between subplots
  pylab.subplots_adjust(wspace=0.04, hspace=0.04, left=0.01, right=0.99,
                        top=0.99, bottom=0.01)
  return figH



def showAllTopicsInSingleImage(topics, compsToHighlight, **imshowArgs):
    K, V = topics.shape
    aspectR = V / float(K)
    pylab.imshow(topics, aspect=aspectR, **imshowArgs)
    if compsToHighlight is not None:
      ks = np.asarray(compsToHighlight)
      if ks.ndim == 0:
        ks = np.asarray([ks])
      pylab.yticks(ks, ['**** %d' % (k) for k in ks])
