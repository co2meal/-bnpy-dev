import numpy as np
from bnpy.obsmodel import ZeroMeanFactorAnalyzerObsModel

# f = 1e5
# g = 1e0
#
# s = 1e0
# t = 1e-5

f = 100
g = 1

s = 10
t = 1

C = 3
D = 4
N = 1e4

def sampleH(N=None):
    return np.random.gamma(f, 1./g, (N,C))

def sampleW(h, N=None):
    assert not np.any(np.isclose(h,0))
    sigma = 1./np.sqrt(h)
    WWT = np.zeros((D, D))
    for n in xrange(int(N)):
        W = np.zeros((D,C))
        for c in xrange(C):
            W[:,c] = np.random.normal(0., sigma[n,c], D)
        WWT += np.dot(W, W.T)
    WWT /= N
    return WWT

def calcCov(WWT, s, t):
    Cov = WWT + np.diag(t/(s-1) * np.ones(D))
    return Cov

if __name__ == '__main__':

    h = sampleH(N=N)

    WWT = sampleW(h, N=N)

    Cov = calcCov(WWT, s, t)

    print Cov