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
                              vmax=None, nDocToPlot=16, doShowNow=True,
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

def plotBarsFromHModel(hmodel, Data=None, doShowNow=True, figH=None,
                       compsToHighlight=None, sortBySize=False,
                       width=6, height=3, vmax=None, Ktop=None, Kmax=None):
    if figH is None:
      figH = pylab.figure(figsize=(width,height))
    else:
      pylab.axes(figH)
    
    if hasattr(hmodel.obsModel, 'Post'):
      lam = hmodel.obsModel.Post.lam
      topics = lam / lam.sum(axis=1)[:,np.newaxis]
    else:
      topics = hmodel.obsModel.EstParams.phi.copy()

    K, V = topics.shape
    if Kmax is not None:
      K = np.minimum(Kmax, K)
      topics = topics[:K]
    aspectR = V / float(K)

    ## Determine intensity scale for topic-word image
    global imshowArgs
    if vmax is not None:
      imshowArgs['vmax'] = vmax
    else:
      imshowArgs['vmax'] = 1.5 * np.percentile(topics, 95)

    pylab.imshow(topics, aspect=aspectR, **imshowArgs)
    if compsToHighlight is not None:
      ks = np.asarray(compsToHighlight)
      if ks.ndim == 0:
        ks = np.asarray([ks])
      pylab.yticks(ks, ['**** %d' % (k) for k in ks])
    if doShowNow and figH is None:
      pylab.show()