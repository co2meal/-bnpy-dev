import numpy as np
from bnpy.data import SeqXData, MinibatchIterator
import scipy.io

'''
AutoRegK4.py

A simple toy dataset that uses an autoregressive gaussian likelihood and
K = 4 state HMM allocation model.

The dataset can be vizualized by running python AutoRegK4.py from the command
line.
'''

#Transition matrix
transPi = np.asarray([[0.97, 0.01, 0.01, 0.01], \
                      [0.01, 0.97, 0.01, 0.01], \
                      [0.01, 0.01, 0.97, 0.01], \
                      [0.01, 0.01, 0.01, 0.97]])

initState = 1

K = 4
D = 2

#Using the variables below, the autoregressive likelihood says:
#  x[n] = A*Xprev[n] + Normal(0, Sigma)
#Define linear scale parameters, A
a1 = 0.9995
A = np.zeros((K, D, D))
A[0] = np.asarray([[1, 0], [0, 0]]) #red
A[1] = np.asarray([[0, 0], [0, -1]]) #blue
A[2] = np.asarray([[0, 0], [0, 0]]) #green
A[3] = np.asarray([[1, 0], [0, 1]]) #yellow

#Define noise parameters, Sigma
s1 = 0.001
s2 = 0.003
Sigma = np.zeros((K, D, D))
Sigma[0] = np.diag([s1, s2])
Sigma[1] = np.diag([s2, s1])
Sigma[2] = np.diag([s2, s1])
Sigma[3] = np.diag([s2, s1])
cholSigma = np.zeros_like(Sigma)
for k in xrange( K ):
  cholSigma[k] = scipy.linalg.cholesky( Sigma[k] )


def get_data(seed=8675309, seqLens=((6000,)), **kwargs):
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
  X, Xprev, TrueZ, seqInds = genToyData(seed, seqLens)
  Data = SeqXData(X = X, TrueZ = TrueZ, Xprev = Xprev, seqInds = seqInds)
  Data.name = get_short_name()
  Data.summary = get_data_info()
  return Data



def genToyData(seed=0, seqLens=((6000,))):
  #Setup the seqIndicies
  seqInds = list([0])
  for ind, sl in enumerate(seqLens):
    seqInds.append(seqInds[ind] + sl)
  nObsTotal = np.sum(seqLens)

  Xprev = np.zeros((nObsTotal+1, D))
  X = np.zeros((nObsTotal, D))
  
  
  Xprev[0,:] = np.zeros(D)

  #Pre-generate the noise that will be added at each step
  PRNG = np.random.RandomState(seed)
  XX = np.zeros((K, nObsTotal, D))
  for k in xrange(K):
    PRNG = np.random.RandomState(seed+k)
    XX[k,:,:] = np.dot(cholSigma[k].T, PRNG.randn(D, nObsTotal) ).T

  PRNG = np.random.RandomState(seed+K)
  rs = PRNG.rand(nObsTotal)
  Z = np.zeros(nObsTotal)
  for n in xrange(nObsTotal):
    if n == 0:
      Z[n] = 0
    else:
      trans = PRNG.multinomial(1, transPi[Z[n-1]])
      Z[n] = np.nonzero(trans)[0][0]
    X[n] = np.dot(A[Z[n]], Xprev[n]) + XX[Z[n], n]
    Xprev[n+1,:] = X[n,:]

  return X, Xprev[:-1,:], Z, seqInds


def get_short_name():
  return 'AutoRegK4'

def get_data_info():
  return 'Toy Autoregressive gaussian data with K = 4 clusters.'
    

if __name__ == '__main__':
  X, Xprev, Z, _ = \
              genToyData(seed=0, seqLens = ((6000,)));
  from matplotlib import pylab

  IDs0 = np.flatnonzero(Z == 0)
  IDs1 = np.flatnonzero(Z == 1)
  IDs2 = np.flatnonzero(Z == 2)
  IDs3 = np.flatnonzero(Z == 3)
  B = np.max(np.abs(X))

  pylab.subplot(3, 1, 1)
  pylab.plot(IDs0, X[IDs0, 0], 'r.')
  pylab.plot(IDs1, X[IDs1, 0], 'b.')
  pylab.plot(IDs2, X[IDs2, 0], 'g.')
  pylab.plot(IDs3, X[IDs3, 0], 'y.')
  pylab.ylim([-B, B])

  pylab.subplot(3, 1, 2)
  pylab.plot(IDs0, X[IDs0, 1], 'r.')
  pylab.plot(IDs1, X[IDs1, 1], 'b.')
  pylab.plot(IDs2, X[IDs2, 1], 'g.')
  pylab.plot(IDs3, X[IDs3, 1], 'y.')
  pylab.ylim([-B, B])

  pylab.subplot(3, 1, 3)
  pylab.scatter(X[IDs0, 0], X[IDs0, 1], c = 'r')
  pylab.scatter(X[IDs1, 0], X[IDs1, 1], c = 'b')
  pylab.scatter(X[IDs3, 0], X[IDs3, 1], c = 'y')
  pylab.scatter(X[IDs2, 0], X[IDs2, 1], c = 'g')

  pylab.tight_layout()
  pylab.show()
