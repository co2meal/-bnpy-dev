import numpy as np
from bnpy.obsmodel import ZeroMeanFactorAnalyzerObsModel

f = 1e5
g = 1e0

s = 1e0
t = 1e-5

C = 3
D = 4
N = 1e4

def sampleH():
    return np.random.gamma(f, 1./g, (N,C))

def sampleW(h):
    assert not np.any(np.isclose(h,0))
    sigma = 1./np.sqrt(h)
    W = np.zeros((N,D,C))
    for n in xrange(int(N)):
        for c in xrange(C):
            W[n,:,c] = np.random.normal(0., sigma[n,c], D)
    WMean = np.mean(W, axis=0)
    WCov = np.zeros((D,C))
    for d in xrange(D):
        for c in xrange(C):
            WCov[d,c] = np.dot(W[:,d,c]-WMean[d,c], W[:,d,c]-WMean[d,c]) / N
    return WMean, WCov

def samplePhi():
    Phi = np.random.gamma(s, 1./t, (N,D))
    Phi = np.mean(Phi, axis=0)
    return Phi

def calcCov(WMean, WCov, Phi):
    Cov = np.inner(WMean,WMean) \
          + np.diag(np.sum(WCov,axis=1)) \
          + np.diag(1./Phi)
    return Cov

if __name__ == '__main__':

    h = sampleH()

    WMean, WCov = sampleW(h)

    Phi = samplePhi()

    Cov = calcCov(WMean, WCov, Phi)

    print Cov