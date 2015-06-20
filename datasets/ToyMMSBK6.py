'''
ToyMMSBK6.py
'''
import numpy as np
from bnpy.data import GraphXData


########################################################### User-facing 
###########################################################
def get_data(seed=123, nNodes=100, K=6, alpha=0.05,
             tau1=1.0, tau0=10.0, genSparse=True, isDirected=True, **kwargs):
  ''' 
    Args
    -------
    seed : integer seed for random number generator,
            used for actually *generating* the data
    seqLens : total number of observations in each sequence

    Returns
    -------
    Data : bnpy GraphXData object, with nObsTotal observations
  '''

  sourceID, destID, Z, w, pi = \
                               gen_data(seed, nNodes, K, alpha, tau1, tau0)
  TrueParams = dict(Z=Z, w=w, pi=pi)
  adjList = np.tile(np.arange(nNodes), (nNodes, 1))
  data = GraphXData(X=None, sourceID=sourceID,
                    destID=destID, nNodesTotal=nNodes, nNodes=nNodes,
                    TrueParams=TrueParams, isSparse=True, isDirected=False)
  data.name = get_short_name()
  return data


def get_short_name():
  return 'ToyMMSBK6'

########################################################### Data generation
###########################################################
w = np.asarray([
  [    .75,    .05,    .05,    .05,    .05,   .05],
  [    .05,    .75,    .05,    .05,    .05,   .05],
  [    .05,    .05,    .75,    .05,    .05,   .05],
  [    .05,    .05,    .05,    .75,    .05,   .05],
  [    .05,    .05,    .05,    .05,    .75,   .05],
  [    .05,    .05,    .05,    .05,    .05,   .75]
  ])
diag = .8
eps = .0001
w[:,:] = eps
w[np.diag_indices(6)] = diag


def gen_data(seed, nNodes, K, alpha,
             tau1=3.0, tau0=1.0, genSparse=False, isDirected=True, **kwargs):
  N = nNodes
  if not hasattr(alpha, '__len__'):
    alpha = alpha * np.ones(K)
  prng = np.random.RandomState(seed)
  pi = prng.dirichlet(alpha, size=nNodes)

  # Generate community assignments
  #z = np.zeros((N,N), dtype=int)
  #for i in xrange(N):
  #  z[i,:] = prng.choice(xrange(K), p=pi[i,:], size=nNodes)
  s = np.zeros((N,N), dtype=int)
  r = np.zeros((N,N), dtype=int)
  for i in xrange(N):
    s[i,:] = prng.choice(xrange(K), p=pi[i,:], size=nNodes)
    r[:,i] = prng.choice(xrange(K), p=pi[i,:], size=nNodes)

  # Make source / receiver assignments
  TrueZ = np.zeros((N,N,2), dtype=int)
  TrueZ[:,:,0] = s
  TrueZ[:,:,1] = r

  # Generate graph
  sourceID = list()
  destID = list()
  for i in xrange(N):
    for j in xrange(N):
      if i == j:
        continue
      #if j > i:
      #  break
      #ind1 = np.min([z[i,j], z[j,i]])
      #ind2 = np.max([z[i,j], z[j,i]])
      #y_ij = prng.binomial(n=1, p=w[ind1,ind2])
      y_ij = prng.binomial(n=1, p=w[s[i,j],r[i,j]])
      if y_ij == 1:
        sourceID.append(i)
        destID.append(j)
  return sourceID, destID, TrueZ, w, pi


if __name__ == '__main__':
  import matplotlib.pyplot as plt
  from bnpy.viz import RelationalViz as relviz
  Data = get_data(nNodes=100)
  colors = np.argmax(Data.TrueParams['pi'], axis=1)

  relviz.plotTrueLabels('ToyMMSBK6', Data, gtypes=['Actual','VarDist', 'EdgePr'],
                        mixColors=True, thresh=.73, colorEdges=False, title='')


  #relviz.plotTransMtx(Data, gtypes=['Actual', 'EdgePr'])
  # Plot subset of pi
  Epi = Data.TrueParams['pi']
  fix, ax = plt.subplots(1)
  ax.imshow(Epi[0:30,:], cmap='Greys', interpolation='nearest')
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  #relviz.plotEpi(None, Data)

  # Plot w
  fig, ax = plt.subplots(1)
  im = ax.imshow(1-w, cmap='gray', interpolation='nearest')
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  

  plt.show()



