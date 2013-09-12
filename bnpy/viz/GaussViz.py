'''
  Visualizing learned Gaussian models
'''
import numpy as np
from matplotlib import pylab

def plotGauss2DFromModel( model, Hrange=None, Krange=None, kID=None, coffset=0, wTHR=0.01 ):
  if kID is not None:
    Krange = list( kID)
  elif Krange is None:
    Krange = np.arange( 0, model.K )
    
  Colors = [ (1,0,0), (1,1,0), (0,1,0), (0,1,1), (0,0,1)]
  try:
    w = np.exp( model.allocModel.Elogw )
  except Exception:
    w = model.allocModel.w

  if Hrange is not None:
    Krange = set(Krange).difference( Hrange)
    for kk in Krange:
      if w[kk] < wTHR:
        continue
      mu = model.obsModel.get_mean(kk)
      Sigma = model.obsModel.get_covar_mat(kk)
      color = 'c'
      plotGauss2D( mu, Sigma, color=color )

    for kk in Hrange:
      mu = model.obsModel.get_mean(kk)
      Sigma = model.obsModel.get_covar_mat(kk)
      color = Colors[kk % len(Colors)]
      plotGauss2D( mu, Sigma, color=color )

  else:
    for kk in Krange:
      if w[kk] < wTHR:
        continue
      mu = model.obsModel.get_mean(kk)
      Sigma = model.obsModel.get_covar_mat(kk)
      color = Colors[ (kk+coffset) % len(Colors)]
      plotGauss2D( mu, Sigma, color=color )

  pylab.axis('image')

ts = np.arange( -np.pi, np.pi, 0.01 )
x  = np.sin( ts)
y  = np.cos( ts)
def plotGauss2D( mu, Sigma, color='b' ):
    mu = np.asarray(mu)
    Sigma = np.asarray(Sigma)

    mu = mu[ :2]
    Sigma = Sigma[ :2, :2]
    D,V = np.linalg.eig( Sigma )
    sqrtSigma = np.dot( V, np.sqrt(np.diag(D)) )

    Zraw = np.vstack( [x, y] )
    Zraw = np.dot( sqrtSigma, Zraw )

    for Rad in np.linspace( 0.5, 2, 3):
      Z = Rad*Zraw + mu[:,np.newaxis]
      pylab.plot( Z[0], Z[1], '.', markerfacecolor=color, markeredgecolor=color )
