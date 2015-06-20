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
                       activeCompIDs=None,  Kmax=50,
                       width=6, height=3, vmax=None, 
                       block=0, # unused
                       jobname='', # unused
                       **kwargs): 
  if vmax is not None:
    kwargs['vmax'] = vmax   
  if hasattr(hmodel.obsModel, 'Post'):
    hmodel.obsModel.setEstParamsFromPost()
  phi = hmodel.obsModel.EstParams.phi.copy()

  ## Determine intensity scale for image pixels
  global imshowArgs
  if vmax is not None:
    imshowArgs['vmax'] = vmax

  if doSquare:
    raise NotImplementedError('TO DO')
  else:
    if figH is None:
      figH = pylab.figure(figsize=(width,height))
    else:
      pylab.axes(figH)
    plotCompsInSingleImage(phi, compsToHighlight, **kwargs)
  if doShowNow:
    pylab.show()
  return figH

def plotCompsInSingleImage(phi, compsToHighlight, **kwargs):
  K, D = phi.shape
  aspectR = D / float(K)
  pylab.imshow(phi, aspect=aspectR, **imshowArgs)
  if compsToHighlight is not None:
    ks = np.asarray(compsToHighlight)
    if ks.ndim == 0:
      ks = np.asarray([ks])
    pylab.yticks(ks, ['**** %d' % (k) for k in ks])