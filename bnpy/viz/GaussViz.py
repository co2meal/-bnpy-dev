'''
GaussViz.py

Visualizing 2D covariance matrix of learned Gaussian mixture models
'''
import numpy as np
from matplotlib import pylab

Colors = [ (1,0,0), (1,0,1), (0,1,0), (0,1,1), (0,0,1), (1,0.6,0)]

def plotGauss2DFromHModel(hmodel, compListToPlot=None, compsToHighlight=None, wTHR=0.01):
  ''' Plot 2D contours for components in hmodel in current pylab figure
      Args
      -------
      hmodel : bnpy HModel object
      compListToPlot : array-like of integer IDs of components within hmodel
      compsToHighlight : int or array-like of integer IDs to highlight
                          if None, all components get unique colors
                          if not None, only highlighted components get colors.
      wTHR : float threshold on minimum weight assigned to component before it is "plottable"      
  '''
  if compsToHighlight is not None:
    compsToHighlight = np.asarray(compsToHighlight)
    if compsToHighlight.ndim == 0:
      compsToHighlight = np.asarray([compsToHighlight])
  else:
    compsToHighlight = list()  
  if compListToPlot is None:
    compListToPlot = np.arange(0, hmodel.allocModel.K)
  try:
    w = np.exp(hmodel.allocModel.Elogw)
  except Exception:
    w = hmodel.allocModel.w

  colorID = 0
  for kk in compListToPlot:
    if w[kk] < wTHR and kk not in compsToHighlight:
      continue
    mu = hmodel.obsModel.get_mean_for_comp(kk)
    Sigma = hmodel.obsModel.get_covar_mat_for_comp(kk)
    if kk in compsToHighlight or len(compsToHighlight) == 0:
      plotGauss2DContour(mu, Sigma, color=Colors[colorID])
      colorID = (colorID + 1) % len(Colors)
    elif kk not in compsToHighlight:
      plotGauss2DContour(mu, Sigma, color='k')
      
  pylab.axis('image')   
  
def plotGauss2DContour(mu, Sigma, color='b', radiusLengths=[0.5, 1.25, 2]):
  ''' Plot elliptical contours for the first 2 dimensions of covariance matrix Sigma,
      location specified by corresponding dims from vector mu
  '''
  mu = np.asarray(mu)
  Sigma = np.asarray(Sigma)

  mu = mu[ :2]
  Sigma = Sigma[ :2, :2]
  D,V = np.linalg.eig( Sigma )
  sqrtSigma = np.dot( V, np.sqrt(np.diag(D)) )

  # Prep for plotting elliptical contours 
  # by creating grid of (x,y) points along perfect circle
  ts = np.arange( -np.pi, np.pi, 0.01 )
  x  = np.sin(ts)
  y  = np.cos(ts)
  Zcirc = np.vstack([x, y])
  
  # Warp circle into ellipse defined by Sigma's eigenvectors
  Zellipse = np.dot( sqrtSigma, Zcirc )

  # plot contour lines across several radius lengths
  # TODO: instead, choose radius by percentage of prob mass contained within
  for r in radiusLengths:
    Z = r * Zellipse + mu[:,np.newaxis]
    pylab.plot(Z[0], Z[1], '.', markerfacecolor=color, markeredgecolor=color )
