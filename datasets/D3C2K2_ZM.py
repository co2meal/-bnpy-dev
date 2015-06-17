'''
D3C2K6.py
Simple 3D zero-mean toy dataset of 2 Gaussian components
'''
import numpy as np
from numpy.linalg import inv
from scipy.linalg import cholesky
from bnpy.data import XData
from bnpy.util import dotATA
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ########################################################## User-facing
def get_data(seed=8675309, nObsTotal=250000, **kwargs):
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

Psi = np.zeros((K,D,D))
Psi[0] = np.diag([1, 5, 1]) * 100
Psi[1] = np.diag([1, 1, 5]) * 1000

def sample_data_from_comp(k, Nk, Lamk, PRNG):
    a_k = PRNG.randn(C, Nk)
    X_k_mean = Mu[k] + np.dot(Lamk, a_k).T
    X_k = np.zeros((Nk, D))
    for n in xrange(Nk):
        X_k[n] = PRNG.multivariate_normal(X_k_mean[n], Psi[k])
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
