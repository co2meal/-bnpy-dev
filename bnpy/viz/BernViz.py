'''
BernViz.py

Visualization tools for beta-bernoulli observation models.
'''

from matplotlib import pylab
import numpy as np

imshowArgs = dict(interpolation='nearest',
                  cmap='bone',
                  vmin=0.0,
                  vmax=1.0)

def plotCompsFromHModel(hmodel, doShowNow=False, figH=None,
                        doSquare=0,
                        xlabels=[],
                        compsToHighlight=None, compListToPlot=None,
                        activeCompIDs=None, Kmax=50,
                        width=6, height=3, vmax=None,
                        block=0,  # unused
                        jobname='',  # unused
                        **kwargs):
    if vmax is not None:
        kwargs['vmax'] = vmax
    if hasattr(hmodel.obsModel, 'Post'):
        hmodel.obsModel.setEstParamsFromPost()
    phi = hmodel.obsModel.EstParams.phi.copy()

    # Determine intensity scale for image pixels
    global imshowArgs
    if vmax is not None:
        imshowArgs['vmax'] = vmax

    if figH is None:
        figH = pylab.figure(figsize=(width, height))
    else:
        pylab.axes(figH)
    dim = phi.shape[1]
    if dim > 9 and isPerfectSquare(dim):
        plotCompsAsSquareImages(phi, compsToHighlight, **kwargs)
    else:
        plotCompsAsRowsInSingleImage(phi, compsToHighlight, **kwargs)
  if doShowNow:
    pylab.show()
  return figH

def plotCompsAsRowsInSingleImage(phi, compsToHighlight, **kwargs):
  K, D = phi.shape
  aspectR = D / float(K)
  pylab.imshow(phi, aspect=aspectR, **imshowArgs)
  if compsToHighlight is not None:
    ks = np.asarray(compsToHighlight)
    if ks.ndim == 0:
      ks = np.asarray([ks])
    pylab.yticks(ks, ['**** %d' % (k) for k in ks])



def plotCompsAsSquareImages(phi, 
                            compsToHighlight=None,
                            compListToPlot=None,
                            activeCompIDs=None,
                            xlabels=[],
                            Kmax=50,
                            W=1, H=1, figH=None, 
                            **imshowArgs):
  if len(xlabels) > 0:
    H = 1.5 * H
  K, V = phi.shape
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
  if figH is None:
    ## Make a new figure
    figH, ha = pylab.subplots(nrows=nrows, ncols=ncols,
                              figsize=(ncols*W,nrows*H))
  else:
    ## Use existing figure
    ## TODO: Find a way to make this call actually change the figsize
    figH, ha = pylab.subplots(nrows=nrows, ncols=ncols,
                              figsize=(ncols*W,nrows*H), num=figH.number)

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
    phiIm = np.reshape(phi[kk,:], (sqrtV, sqrtV))

    ax = pylab.subplot(nrows, ncols, plotID+1)
    pylab.imshow(phiIm, aspect=1.0, **imshowArgs)
    pylab.xticks([])
    pylab.yticks([])

    ## Draw colored border around highlighted topics
    if compID in compsToHighlight:
      [i.set_color('green') for i in ax.spines.itervalues()]
      [i.set_linewidth(3) for i in ax.spines.itervalues()]

    if xlabels is not None:
      if len(xlabels) > 0:
        pylab.xlabel(xlabels[plotID], fontsize=15)

  ## Disable empty plots!
  for kdel in xrange(plotID+2, nrows*ncols+1):
    aH = pylab.subplot(nrows, ncols, kdel)
    aH.axis('off')

  ## Fix margins between subplots
  pylab.subplots_adjust(wspace=0.04, hspace=0.04, left=0.01, right=0.99,
                        top=0.99, bottom=0.01)
  return figH



def plotDataAsSquareImages(Data, unitIDsToPlot=None,
                           figID=None,
                           nPlots=16, doShowNow=False,
                           seed=0, randstate=np.random.RandomState(0),
                           **kwargs):
    if seed is not None:
      randstate = np.random.RandomState(seed)
    if figID is None:
      pylab.figure()

    V = Data.dim
    assert isPerfectSquare(V)
    sqrtV = int(np.sqrt(V))
    if unitIDsToPlot is not None:
      nPlots = len(unitIDsToPlot)
    else:
      size = np.minimum(Data.nObs, nPlots)
      unitIDsToPlot = randstate.choice(Data.nObs, size=size, replace=False)
    nRows = np.floor(np.sqrt(nPlots))
    nCols = np.ceil(nPlots / nRows)

    for plotPos, unitID in enumerate(unitIDsToPlot):
        squareIm = np.reshape(Data.X[unitID], (sqrtV, sqrtV))
        pylab.subplot(nRows, nCols, plotPos+1)
        pylab.imshow(squareIm, **imshowArgs)
        pylab.axis('image')
        pylab.xticks([])
        pylab.yticks([])
    pylab.tight_layout()
    if doShowNow:
      pylab.show()

def isPerfectSquare(n):
    return np.allclose(n, int(np.sqrt(n))**2)
