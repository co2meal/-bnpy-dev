'''
BinaryGraphK5.py

Binary Toy Graph with K=5 communities.
'''
import numpy as np
import random
from bnpy.data import GraphData

SEED = 8675309
PRNG = np.random.RandomState(SEED)

# FIXED DATA GENERATION PARAMS
K = 5 # Number of communities
N = 50 # Number of nodes
beta_a = 0.1 # hyperparameter over block matrix entries
beta_b = 0.1 # hyperparameter over block matrix entries

Defaults = dict()
Defaults['nNodeTotal'] = 50

# Initialize adjacency matrix and stochastic block matrix
sb = np.zeros( (K,K) ) + 0.01
sb[0,0] = .9
sb[1,1] = .9
sb[2,2] = .9
sb[3,3] = .9
sb[4,4] = .9

# function to generate adjacency matrix
def gen_graph(K, N, sb):

    # define the edge indices and edge values
    edge_id = list()

    # generate community memberships
    pi = np.zeros( (N,K) )
    alpha = np.zeros(K) + .1
    for ii in xrange(N):
        pi[ii,:] = PRNG.dirichlet(alpha)

    for ii in xrange(N):
        for jj in xrange(N):
            if ii != jj and ii < jj:
                s = PRNG.choice(5, 1, p=pi[ii,:])
                r = PRNG.choice(5, 1, p=pi[jj,:])
                if PRNG.rand() < sb[s,r]:
                    edge_id.append([ii,jj])

    E = len(edge_id)
    edge_weight = np.ones(E)
    return (edge_id, edge_weight)

# template function to wrap data in bnpy format
def get_data(**kwargs):
    ''' Grab data from matfile specified by matfilepath
    '''
    edge_id, edge_weight = gen_graph(K,N,sb)
    Data = GraphData(edge_id=edge_id, edge_value=edge_weight, nNodeTotal=N)
    Data.summary = get_data_info(K, Data.nNodeTotal, Data.nEdgeTotal)
    return Data

def get_minibatch_iterator(nBatch=10, nLap=1, dataorderseed=0, **kwargs):
    pass

def get_data_info(K,N,E):
    return 'Toy Binary Graph Dataset where K=%d . N=%d. E=%d' % (K,N,E)