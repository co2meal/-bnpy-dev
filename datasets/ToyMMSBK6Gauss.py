'''
ToyMMSBK6.py
'''
import numpy as np
import scipy.io
from bnpy.data import GraphXData



########################################################### User-facing 
###########################################################
def get_data(seed=123, nNodes=500, K=6, alpha=0.05,
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

  X, Z, pi, mu, sigma = \
             gen_data(seed, nNodes, alpha)
  TrueParams = dict(Z=Z, pi=pi, mu=mu, sigma=sigma)
  X = np.reshape(X, X.shape+(1,))
  data = GraphXData(X=X, sourceID=None,
                    destID=None, nNodesTotal=nNodes, nNodes=nNodes,
                    TrueParams=TrueParams, isSparse=False)
  data.name = get_short_name()
  return data


def get_short_name():
  return 'ToyMMSBK6Gauss'



########################################################### Data generation
###########################################################
K = 6
diagMus = 2.0
delta = .1
epsilon = 1e-4
mus = np.ones(K) * diagMus
sigmas = np.ones(K) * 1.0
mus = [.25, .35, .45, .55, .65, .75]
mus = [m*100 for m in mus]




def gen_data(seed, nNodes, alpha, **kwargs):

  prng = np.random.RandomState(seed)
  np.random.seed(seed)

  N = nNodes
  if not hasattr(alpha, '__len__'):
    alpha = alpha * np.ones(K)
  pi = prng.dirichlet(alpha, size=nNodes)
  
  # Make source / receiver assignments
  s = np.zeros((N,N), dtype=int)
  r = np.zeros((N,N), dtype=int)
  for i in xrange(N):
    s[i,:] = prng.choice(xrange(K), p=pi[i,:], size=nNodes)
    r[:,i] = prng.choice(xrange(K), p=pi[i,:], size=nNodes)
  TrueZ = np.zeros((N,N,2), dtype=int)
  TrueZ[:,:,0] = s
  TrueZ[:,:,1] = r

  
  # Generate graph
  X = np.zeros((N,N))
  cnt = 0
  for i in xrange(N):
    for j in xrange(N):
      if i == j:
        continue
      if s[i,j] == r[i,j]:
        X[i,j] = np.random.normal(mus[s[i,j]], sigmas[s[i,j]])
        cnt += 1
  print 'THE IN COM COUNT IS ', cnt
  M = np.max(np.abs(X))
  for i in xrange(N):
    for j in xrange(N):
      if i == j:
        continue
      if s[i,j] != r[i,j]:
        inInterval = prng.binomial(n=1, p=1-epsilon)
        if inInterval:
          X[i,j] = np.random.uniform(low=-delta, high=delta)
        else:
          negativeHalf = prng.binomial(n=1, p=.5)
          if negativeHalf:
            X[i,j] = np.random.uniform(low=-M, high=-delta)
          else:
            X[i,j] = np.random.uniform(low=delta, high=M)

  print 'GENERATED WITH M = ', M
  return X, TrueZ, pi, mus, sigmas


if __name__ == '__main__':
  import networkx as nx
  import matplotlib.pyplot as plt

  Data = get_data()
  N = Data.nNodesTotal

  Epi = Data.TrueParams['pi']
  K = np.shape(Epi)[1]
  colors = np.sum(Epi*np.arange(K)[np.newaxis,:], axis=1)

  labels = np.arange(N)
  
  G = nx.DiGraph()
  thresh = Data.Xmatrix > 0
  edgeColors = list()
  for i in xrange(Data.nNodesTotal):
    for j in xrange(Data.nNodesTotal):
      if thresh[i,j] == 1:
        G.add_edge(i,j)
        edgeColors.append(Data.Xmatrix[i,j,0])

        
  fig, ax = plt.subplots(1)
  pos = nx.spring_layout(G)
  nx.draw_networkx_nodes(G, pos, node_color=colors, cmap='gist_rainbow')
  nx.draw_networkx_labels(G, pos, labels=dict(zip(np.arange(N),labels)))
  nx.draw_networkx_edges(G, pos, edge_color=edgeColors, edge_cmap=plt.cm.Greys,
                         arrows=False)
  scipy.io.savemat('x.mat', {'X':Data.Xmatrix})
  from IPython import embed; embed()
  plt.show()

