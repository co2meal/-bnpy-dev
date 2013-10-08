'''
DirRelBinarySingleK.py

Simple toy dataset of 5 underlying clusters that forms a directed binary graph.  
'''
import numpy as np
from bnpy.data import GraphData, MinibatchIterator
######################################################################  Generate Toy Params

def get_data(seed=8675309, K=10, N=100):
    X, TrueZ = get_X(seed, K, N)
    Data = GraphData(X=X)
    Data.summary = get_data_info()
    return Data

def get_data_info():
    return 'Single Cluster Toy Graph. Ktrue=%d. N=%d.' % (5,100)
            
def get_X(seed, K, N):
    E = N * N
    # data array to hold edge information
    # col1 = node_i, col2 = node_j, col3 = y_ij
    Y = np.zeros((E, 3)) 
    A = np.zeros((N, N))
    Z = np.zeros((N, 1))
    # Generate a very sparse Stochastic Block Matrix
    omega = np.random.beta(.01, .01, [K, K])

    offset = np.round(N / K)
    for k in xrange(K):
        end = (k + 1) * offset
        start = (k + 1) * offset - offset
        Z[start:end] = k

    e = 0
    for i in xrange(N):
        for j in xrange(N):
            if i != j:
                zi = int(Z[i])
                zj = int(Z[j])
                Y[e, 0] = i
                Y[e, 1] = j

                if np.random.rand() < omega[zi, zj]:
                    Y[e, 2] = 1
                    A[i, j] = 1
                else:
                    Y[e, 2] = 0
                e += 1
    
    #print A.sum()        
    return Y, Z      
