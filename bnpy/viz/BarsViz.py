'''
BarsViz.py

Visualization tools for toy bars data for topic models.
'''
from matplotlib import pylab
import numpy as np

def plotExampleBarsDocs(Data, docIDsToPlot=None, nDocToPlot=9, doShowNow=True):
    pylab.figure()
    V = Data.vocab_size
    sqrtV = int(np.sqrt(V))
    assert np.allclose(sqrtV * sqrtV, V)
    if docIDsToPlot is not None:
      nDocToPlot = len(docIDsToPlot)
    else:
      docIDsToPlot = np.random.choice(Data.nDoc, size=nDocToPlot, replace=False)
    nRows = np.floor(np.sqrt(nDocToPlot))
    nCols = np.ceil(nDocToPlot / nRows)

    for plotPos, docID in enumerate(docIDsToPlot):
        # Parse current document
        start,stop = Data.doc_range[docID,:]
        wIDs = Data.word_id[start:stop]
        wCts = Data.word_count[start:stop]
        docWordHist = np.zeros(V)
        docWordHist[wIDs] = wCts
        squareIm = np.reshape(docWordHist, (np.sqrt(V), np.sqrt(V)))

        pylab.subplot(nRows, nCols, plotPos)
        pylab.imshow(squareIm, interpolation='nearest')
    if doShowNow:
      pylab.show()

def plotBarsFromHModel(hmodel, Data=None, doShowNow=True, width=12, height=3):
    pylab.figure(figsize=(width,height))
    K = hmodel.allocModel.K
    VocabSize = hmodel.obsModel.comp[0].lamvec.size
    learned_tw = np.zeros( (K, VocabSize) )
    for k in xrange(K):
        lamvec = hmodel.obsModel.comp[k].lamvec 
        learned_tw[k,:] = lamvec / lamvec.sum()
    if Data is not None and hasattr(Data, "true_tw"):
        # Plot the true parameters and learned parameters
        pylab.subplot(121)
        pylab.imshow(Data.true_tw, interpolation="nearest", cmap="bone")
        pylab.colorbar()
        pylab.title('True Topic x Word')
        pylab.subplot(122)
        pylab.imshow(learned_tw, interpolation="nearest", cmap="bone")
        pylab.colorbar()
        pylab.title('Learned Topic x Word')
    else:
        # Plot just the learned parameters
        pylab.imshow(learned_tw, interpolation="nearest", cmap="bone")
        pylab.colorbar
        pylab.title('Learned Topic x Word')
    if doShowNow:
      pylab.show()
