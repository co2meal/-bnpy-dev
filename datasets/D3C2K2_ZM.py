'''
D3C2K6.py
Simple 3D zero-mean toy dataset of 2 Gaussian components
'''
import numpy as np
from numpy.linalg import inv
from bnpy.data import XData

import sys
sys.path.insert(0, '/Library/Python/2.7/site-packages/mpl_toolkits/')
from mplot3d import Axes3D
import matplotlib.pyplot as plt


# ########################################################## User-facing
def get_data(seed=8675309, nObsTotal=2500, **kwargs):
    X, TrueZ, a, W = get_X(seed, nObsTotal)
    TrueParams = dict()
    TrueParams['W'] = W
    TrueParams['a'] = a
    TrueParams['K'] = K
    TrueParams['Mu'] = Mu
    TrueParams['h'] = h
    TrueParams['Pi'] = Pi
    TrueParams['Psi'] = Psi
    Data = XData(X=X, TrueZ=TrueZ, TrueParams=TrueParams)
    Data.name = 'D3C2K2'
    Data.summary = get_data_info()
    return Data

def get_short_name():
    return 'D3C2K2'

def get_data_info():
    return 'FA Toy Data. %d true clusters.' % (K)


###########################################################  Set Toy Parameters
###########################################################

K = 2
D = 3
C = 2

Pi = np.asarray([1., 2.])
Pi = Pi / Pi.sum()

Mu = np.zeros((K, D))

h = np.zeros((K, C))
h[0] = np.asarray([1.0, 2.0])
h[1] = np.asarray([3.0, 1.0])
#h[1] = np.asarray([np.inf, 1.0])
h *= 1e-3

Psi = np.diag([1, 2, 1])

def sample_data_from_comp(k, Nk, Lamk, PRNG):
    a_k = PRNG.randn(C, Nk)
    X_k = Mu[k] + np.dot(Lamk, a_k).T
    return X_k, a_k.T

def get_X(seed, nObsTotal):
    PRNG = np.random.RandomState(seed)
    trueList = list()
    Npercomp = PRNG.multinomial(nObsTotal, Pi)
    X = list()
    a = list()
    W = np.zeros((K, D, C))
    for k in range(K):
        for d in range(D):
            W[k][d] = PRNG.multivariate_normal(np.zeros(C),inv(np.diag(h[k])))
        X_k, a_k = sample_data_from_comp(k, Npercomp[k], W[k], PRNG)
        X.append(X_k)
        a.append(a_k)
        trueList.append(k*np.ones(Npercomp[k]))
    X = np.vstack(X)
    a = np.vstack(a)
    TrueZ = np.hstack(trueList)
    permIDs = PRNG.permutation(X.shape[0])
    X = X[permIDs]
    a = a[permIDs]
    TrueZ = TrueZ[permIDs]
    return X, TrueZ, a, W

def plot_data_by_k(Data):
    from bnpy.viz import GaussViz
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for k in range(K):
        c = k % len(GaussViz.Colors)
        idx = np.flatnonzero(Data.TrueParams['Z'] == k)
        x = Data.X[idx, 0]
        y = Data.X[idx,1]
        z = Data.X[idx,2]
        ax.scatter(x, y, z, c=GaussViz.Colors[c], marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

if __name__ == '__main__':
    Data = get_data(nObsTotal=5000)
    plot_data_by_k(Data)
