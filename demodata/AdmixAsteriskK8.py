'''
AdmixAsteriskK8.py

Toy dataset of 8 Gaussian components with full covariance.
  
Generated data form an well-separated blobs arranged in "asterisk" shape when plotted in 2D.
'''
import scipy.linalg
import numpy as np
from bnpy.util.RandUtil import rotateCovMat
from bnpy.data import XData, MinibatchIterator

########################################################### User-facing 
def get_data(seed=8675309, nObsTotal=25000, **kwargs):
  '''
    Args
    -------
    seed : integer seed for random number generator,
            used for actually *generating* the data
    nObsTotal : total number of observations for the dataset.

    Returns
    -------
      Data : bnpy XData object, with nObsTotal observations
  '''
  X, TrueZ = get_X(seed, nObsTotal)
  Data = XData(X=X, TrueZ=TrueZ)
  Data.summary = get_data_info()
  return Data
  
def get_minibatch_iterator(seed=8675309, nObsTotal=25000, **kwargs):
  '''
    Args
    --------
    seed : integer seed for random number generator,
            used for actually *generating* the data
    dataorderseed : integer seed that determines
                     (a) how data is divided into minibatches
                     (b) order these minibatches are traversed

   Returns
    -------
      bnpy MinibatchIterator object, with nObsTotal observations
        divided into nBatch batches
  '''
  X, TrueZ = get_X(seed, nObsTotal)
  Data = XData(X=X)
  Data.summary = get_data_info()
  DataIterator = MinibatchIterator(Data, **kwargs)
  return DataIterator

def get_short_name( ):
  ''' Return short string used in filepaths to store solutions
  '''
  return 'AdmixAsteriskK8'

def get_data_info():
  return 'Admixture Asterisk Toy Data. %d true clusters.' % (K)

###########################################################  Set Toy Parameters
###########################################################

K = 8
D = 2

gamma = 1.0

## Create "true" mean parameters
## Placed evenly spaced around a circle
Rad = 1.0
ts = np.linspace(0, 2*np.pi, K+1)
ts = ts[:-1]
Mu = np.zeros( (K,D))
Mu[:,0] = np.cos(ts)
Mu[:,1] = np.sin(ts)

## Create "true" covariance parameters
## Each is a rotation of a template with major axis much larger than minor one
V = 1.0/16.0
SigmaBase = np.asarray([[ V, 0], [0, V/100.0]])
Sigma = np.zeros( (K,D,D) )
for k in xrange(K):
  Sigma[k] = rotateCovMat(SigmaBase, k*np.pi/4.0)
# Precompute cholesky decompositions
cholSigma = np.zeros(Sigma.shape)
for k in xrange( K ):
  cholSigma[k] = scipy.linalg.cholesky( Sigma[k] )

def sample_data_from_comp(k, Nk, PRNG):
  return Mu[k,:] + np.dot(cholSigma[k].T, PRNG.randn(D, Nk) ).T

def get_X(seed, nDocTotal, nObsPerDoc):
  PRNG = np.random.RandomState(seed)
  Npercomp = PRNG.multinomial(nObsTotal, w)
  X = list()
  for k in range(K):
    X.append( sample_data_from_comp( k, Npercomp[k], PRNG) )
    trueList.append( k*np.ones( Npercomp[k] ) )
  X = np.vstack( X )
  TrueZ = np.hstack( trueList )
  permIDs = PRNG.permutation( X.shape[0] )
  X = X[permIDs]
  TrueZ = TrueZ[permIDs]
  return X, TrueZ


########################################################### Main
###########################################################

def plot_true_clusters():
  from bnpy.viz import GaussViz
  for k in range(K):
    c = k % len(GaussViz.Colors)
    GaussViz.plotGauss2DContour(Mu[k], Sigma[k], color=GaussViz.Colors[c])

if __name__ == "__main__":
  from matplotlib import pylab
  pylab.figure()
  Data = get_data(nObsTotal=5000)
  plot_true_clusters()
  pylab.show(block=True)