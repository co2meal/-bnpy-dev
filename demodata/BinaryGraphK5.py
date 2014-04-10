'''
BinaryGraphK5.py

Binary Toy Graph with K=5 communities.
'''
import numpy as np
import random
from bnpy.data import GraphData

SEED = 5141982
PRNG = np.random.RandomState(SEED)

# FIXED DATA GENERATION PARAMS
K = 5 # Number of communities
N = 75 # Number of nodes
beta_a = 0.1 # hyperparameter over block matrix entries
beta_b = 0.1 # hyperparameter over block matrix entries

Defaults = dict()
Defaults['nNodeTotal'] = N

# Initialize adjacency matrix and stochastic block matrix
sb = np.zeros( (K,K) ) + 0.0001
sb[0,0] = .95
sb[1,1] = .95
sb[2,2] = .95
sb[3,3] = .95
sb[4,4] = .95

# function to generate adjacency matrix
def gen_graph(K, N, sb):

    # define the edge indices and edge values
    edge_val = list()
    edge_exclude = list() # edges to exclude (10%)
    exclusion_thresh = 0.95 # 1 = no excluded edges

    # generate community memberships
    pi = np.zeros( (N,K) ) + 1
    # alpha = np.zeros(K) + .1
    A = np.zeros( (N,N) )
    for ii in xrange(N):
        k = np.floor(ii/15)
        #k2 = PRNG.choice(K, 1, p=(np.ones(K)/K))
        pi[ii,k] += 20
        #pi[ii,k2] += 10
        pi[ii,:] = pi[ii,:] / np.sum(pi[ii,:])

    for ii in xrange(N):
        for jj in xrange(ii+1,N):
            if ii != jj and ii < jj:
                s = PRNG.choice(5, 1, p=pi[ii,:])
                r = PRNG.choice(5, 1, p=pi[jj,:])
                # If this edge is not being exlcuded, just add to edge_id
                if PRNG.rand() <= exclusion_thresh:
                    if PRNG.rand() < sb[s,r]:
                        edge_val.append([ii,jj,1])
                        A[ii,jj] = 1
                else: # include this as an edge that needs to be excluded
                    if PRNG.rand() < sb[s,r]:
                        edge_exclude.append([ii,jj,1])
                    else:
                        edge_exclude.append([ii,jj,0])
                    A[ii,jj] = -1


    edge_val = np.asarray(np.squeeze(edge_val), dtype=np.int32)
    edge_exclude = np.asarray(np.squeeze(edge_exclude), dtype=np.int32)

    return (edge_val, edge_exclude, A, pi)

# template function to wrap data in bnpy format
def get_data(**kwargs):
    ''' Grab data from matfile specified by matfilepath
    '''
    edge_val, edge_exclude, A, pi = gen_graph(K,N,sb)
    Data = GraphData(edge_val = edge_val, nNodeTotal=N, edge_exclude=edge_exclude)
    Data.summary = get_data_info(K, Data.nNodeTotal, Data.nEdgeTotal)
    Data.get_edges_all() # Grab the full set of edges for inference

    # For debugging only
    import scipy.io
    savedir = '/Users/daeil/Dropbox/research/bnpy/results/BinaryGraphK5/HDPRelAssortModel/BernRel/VB/defaultjob/1/data.mat'
    scipy.io.savemat( savedir, dict(A=A, pi=pi))
    Data.true_pi = pi

    return Data

def get_minibatch_iterator(nBatch=10, nLap=1, dataorderseed=0, **kwargs):
    pass

def get_data_info(K,N,E):
    return 'Toy Binary Graph Dataset where K=%d . N=%d. E=%d' % (K,N,E)