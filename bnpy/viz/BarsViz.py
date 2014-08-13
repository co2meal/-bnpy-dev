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
                       compsToHighlight=None, sortBySize=False,
                       width=6, height=3, vmax=None, Ktop=None, Kmax=None):    
    if hasattr(hmodel.obsModel, 'Post'):
      lam = hmodel.obsModel.Post.lam
      topics = lam / lam.sum(axis=1)[:,np.newaxis]
    else:
      topics = hmodel.obsModel.EstParams.phi.copy()

    K, V = topics.shape
    if Kmax is not None:
      K = np.minimum(Kmax, K)
      topics = topics[:K]

    ## Determine intensity scale for topic-word image
    global imshowArgs
    if vmax is not None:
      imshowArgs['vmax'] = vmax
    else:
      imshowArgs['vmax'] = 1.5 * np.percentile(topics, 95)

    if doSquare:
      showTopicsAsSquareImages(topics, compsToHighlight, **imshowArgs)
    else:
      if figH is None:
        figH = pylab.figure(figsize=(width,height))
      else:
        pylab.axes(figH)
      showAllTopicsInSingleImage(topics, compsToHighlight, **imshowArgs)
    if doShowNow:
      pylab.show()
    return figH

def showTopicsAsSquareImages(topics, compsToHighlight, **imshowArgs):
  K, V = topics.shape
  sqrtV = int(np.sqrt(V))
  assert np.allclose(sqrtV, np.sqrt(V))
  nrows = int(np.maximum(int(np.sqrt(K)), 1))
  ncols = int(np.ceil(K / float(nrows)))
  compsToHighlight = np.asarray(compsToHighlight)
  if compsToHighlight.ndim == 0:
    compsToHighlight = np.asarray([compsToHighlight])

  W = 1
  H = 1
  hf, ha = pylab.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*W,nrows*H))
  for k in xrange(K):
    ax = pylab.subplot(nrows, ncols, k+1)
    topicIm = np.reshape(topics[k,:], (sqrtV, sqrtV))
    if k in compsToHighlight:
      #topicIm[0, :] = imshowArgs['vmax']
      #topicIm[-1, :] = imshowArgs['vmax']
      #topicIm[:, 0] = imshowArgs['vmax']
      #topicIm[:, -1] = imshowArgs['vmax']
      ax.spines['bottom'].set_color('green')
      ax.spines['top'].set_color('green')
      ax.spines['left'].set_color('green')
      ax.spines['right'].set_color('green')
      [i.set_linewidth(3) for i in ax.spines.itervalues()]

    pylab.imshow(topicIm, aspect=1.0, **imshowArgs)
    pylab.xticks([])
    pylab.yticks([])
  for kdel in xrange(K+1, nrows*ncols+1):
    aH = pylab.subplot(nrows, ncols, kdel)
    aH.axis('off')
  #pylab.tight_layout()
  pylab.subplots_adjust(wspace=0.04, hspace=0.04, left=0.01, right=0.99,
                        top=0.99, bottom=0.01)
  
def showAllTopicsInSingleImage(topics, compsToHighlight, **imshowArgs):
    K, V = topics.shape
    aspectR = V / float(K)
    pylab.imshow(topics, aspect=aspectR, **imshowArgs)
    if compsToHighlight is not None:
      ks = np.asarray(compsToHighlight)
      if ks.ndim == 0:
        ks = np.asarray([ks])
      pylab.yticks(ks, ['**** %d' % (k) for k in ks])
