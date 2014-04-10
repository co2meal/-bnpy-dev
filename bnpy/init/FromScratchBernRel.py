'''
FromScratchBernRelational.py

Initialize params of HModel with multinomial observations from scratch.
'''
import numpy as np
from scipy.special import digamma
from scipy.cluster import vq

hasRexAvailable = True
try:
  import KMeansRex
except ImportError:
  hasRexAvailable = False

def init_global_params(hmodel, Data, initname='randexamples',
                               seed=0, K=0, initarg=None, **kwargs):
  ''' Initialize hmodel's global parameters in-place.

      Returns
      -------
      Nothing. hmodel is updated in place.
      Global Paramters are:
        lamA, lamB = K x K stochastic block matrix
        theta = N x K+1 matrix of community membership probabilities
  '''
  PRNG = np.random.RandomState(seed)
  N = Data.nNodeTotal
  if initname == 'randexamples':
    # Generate a sparse matrix given observed positive edges
    #Data.to_sparse_matrix()
    # Create assortative stochastic block matrix
    lamA = np.zeros( K ) + (Data.nPosEdges / K) # assortative ( K x 1 ) vs. (K x K)
    lamB = np.zeros( K ) + (Data.nAbsEdges / (K*K)) # assortative
    # Create theta used for
    theta = np.zeros( (N,K+1) )
    alpha = np.ones(K)
    beta = np.ones(K) / K

    for ii in xrange(N):
        theta[ii,:K] = PRNG.dirichlet(alpha)
        theta[ii,:] += 1

    #theta = Data.true_pi
    # Initialize global stick-breaking weights beta to be 1/K (uniform)

    # Set the global parameters for the hmodel
    hmodel.set_global_params(K=K, beta=beta, lamA=lamA, lamB=lamB, theta=theta)
    return
  else:
    raise NotImplementedError('Unrecognized initname ' + initname)
