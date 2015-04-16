'''
D3E2K6.py
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
    X, TrueZ, Y, Lam = get_X(seed, nObsTotal)
    TrueParams = dict()
    TrueParams['Lam'] = Lam
    TrueParams['Y'] = Y
    TrueParams['K'] = K
    TrueParams['Mu'] = Mu
    TrueParams['Nu'] = Nu
    TrueParams['Pi'] = Pi
    TrueParams['Psi'] = Psi
    Data = XData(X=X, TrueZ=TrueZ, TrueParams=TrueParams)
    Data.name = 'D3E2K6'
    Data.summary = get_data_info()
    return Data

def get_short_name():
    return 'D3E2K2'

def get_data_info():
    return 'FA Toy Data. %d true clusters.' % (K)


###########################################################  Set Toy Parameters
###########################################################

K = 2
D = 3
E = 2

Pi = np.asarray([1., 2.])
Pi = Pi / Pi.sum()

Mu = np.zeros((K, D))

Nu = np.zeros((K, E))
Nu[0] = np.asarray([1.0, 2.0])
Nu[1] = np.asarray([3.0, 1.0])
#Nu[1] = np.asarray([np.inf, 1.0])
Nu *= 1e-3

Psi = np.diag([1, 2, 1])

def sample_data_from_comp(k, Nk, Lamk, PRNG):
    Y_k = PRNG.randn(E, Nk)
    X_k = Mu[k] + np.dot(Lamk, Y_k).T
    return X_k, Y_k.T

def get_X(seed, nObsTotal):
    PRNG = np.random.RandomState(seed)
    trueList = list()
    Npercomp = PRNG.multinomial(nObsTotal, Pi)
    X = list()
    Y = list()
    Lam = np.zeros((K, D, E))
    for k in range(K):
        for d in range(D):
            Lam[k][d] = PRNG.multivariate_normal(np.zeros(E),inv(np.diag(Nu[k])))
        X_k, Y_k = sample_data_from_comp(k, Npercomp[k], Lam[k], PRNG)
        X.append(X_k)
        Y.append(Y_k)
        trueList.append(k*np.ones(Npercomp[k]))
    X = np.vstack(X)
    Y = np.vstack(Y)
    TrueZ = np.hstack(trueList)
    permIDs = PRNG.permutation(X.shape[0])
    X = X[permIDs]
    Y = Y[permIDs]
    TrueZ = TrueZ[permIDs]
    return X, TrueZ, Y, Lam

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
