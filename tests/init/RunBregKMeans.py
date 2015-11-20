from __future__ import print_function

import numpy as np
import bnpy
runBregKMeans = bnpy.init.FromScratchBregman.runKMeans_BregmanDiv

from bnpy.viz.PlotUtil import pylab

def test_DiagGauss(K=50, N=1000, D=1, W=None, eps=1e-10, nu=0.001, kappa=0.001):
    import StarCovarK5
    Data = StarCovarK5.get_data(nObsTotal=N)
    if D < Data.X.shape[1]:
        Data = bnpy.data.XData(X=Data.X[:,:D])
    hmodel = bnpy.HModel.CreateEntireModel(
        'VB', 'DPMixtureModel', 'DiagGauss',
        dict(gamma0=10),
        dict(ECovMat='eye', sF=0.5, nu=nu, kappa=kappa),
        Data)
    if W:
        W = np.asarray(W)
        if W.size != N: 
            PRNG = np.random.RandomState(0)
            W = PRNG.rand(N)
    Z, Mu, Lscores = runBregKMeans(
        Data.X, K, hmodel.obsModel,
        W=W, smoothFrac=0, smoothFracInit=1.0,
        logFunc=print, eps=eps)
    try:
        assert np.all(np.diff(Lscores) <= 0)
    except AssertionError:
        from IPython import embed; embed()
    return Z, Mu, Lscores


def test_Gauss(K=50, N=1000, D=1, W=None, eps=1e-10, nu=0.001, kappa=0.001):
    import StarCovarK5
    Data = StarCovarK5.get_data(nObsTotal=N)
    if D < Data.X.shape[1]:
        Data = bnpy.data.XData(X=Data.X[:,:D])
    hmodel = bnpy.HModel.CreateEntireModel(
        'VB', 'DPMixtureModel', 'Gauss',
        dict(gamma0=10),
        dict(ECovMat='eye', sF=0.5, nu=nu, kappa=kappa),
        Data)
    if W:
        W = np.asarray(W)
        if W.size != N: 
            PRNG = np.random.RandomState(0)
            W = PRNG.rand(N)
    Z, Mu, Lscores = runBregKMeans(
        Data.X, K, hmodel.obsModel,
        W=W, smoothFrac=0, smoothFracInit=1.0,
        logFunc=print, eps=eps)
    try:
        assert np.all(np.diff(Lscores) <= 0)
    except AssertionError:
        from IPython import embed; embed()
    return Z, Mu, Lscores

def test_ZeroMeanGauss(K=50, N=1000, D=1, W=None, eps=1e-10):
    import StarCovarK5
    Data = StarCovarK5.get_data(nObsTotal=N)
    if D < Data.X.shape[1]:
        Data = bnpy.data.XData(X=Data.X[:,:D])
    hmodel = bnpy.HModel.CreateEntireModel(
        'VB', 'DPMixtureModel', 'ZeroMeanGauss',
        dict(gamma0=10),
        dict(ECovMat='eye', sF=0.5, nu=0.001),
        Data)
    if W:
        W = np.asarray(W)
        if W.size != N: 
            PRNG = np.random.RandomState(0)
            W = PRNG.rand(N)
    Z, Mu, Lscores = runBregKMeans(
        Data.X, K, hmodel.obsModel, 
        W=W, smoothFrac=0, smoothFracInit=1.0,
        logFunc=print, eps=eps)
    assert np.all(np.diff(Lscores) <= 0)
    return Z, Mu, Lscores

def test_Bern(K=50, N=1000, W=None):
    import SeqOfBinBars9x9
    Data = SeqOfBinBars9x9.get_data(nDocTotal=N, T=1)
    hmodel = bnpy.HModel.CreateEntireModel(
        'VB', 'DPMixtureModel', 'Bern',
        dict(gamma0=10),
        dict(lam1=0.1, lam0=0.1),
        Data)
    if W:
        W = np.asarray(W)
        if W.size != N:   
            PRNG = np.random.RandomState(0)
            W = PRNG.rand(N)
    Z, Mu, Lscores = runBregKMeans(
        Data.X, K, hmodel.obsModel,
        W=W, smoothFrac=0.0, smoothFracInit=1.0,
        logFunc=print)
    assert np.all(np.diff(Lscores) <= 0)

def test_Mult(K=50, N=1000, W=None):
    import BarsK10V900
    Data = BarsK10V900.get_data(nWordsPerDoc=33, nDocTotal=N)
    X = Data.getDocTypeCountMatrix()
    hmodel = bnpy.HModel.CreateEntireModel(
        'VB', 'DPMixtureModel', 'Mult',
        dict(gamma0=10),
        dict(lam=0.01),
        Data)
    if W:
        W = np.asarray(W)
        if W.size != N:   
            PRNG = np.random.RandomState(0)
            W = PRNG.rand(N)
    Z, Mu, Lscores = runBregKMeans(
        X, K, hmodel.obsModel,
        W=W, smoothFrac=0.0, smoothFracInit=1.0,
        logFunc=print)
    assert np.all(np.diff(Lscores) <= 0)

if __name__ == '__main__':
    for N in [5, 10, 33, 211, 345, 500, 1000]:
        print('')
        for K in [1, 3, 5, 7, 10, 20, 50]:
            if K > N:
                continue
            test_Bern(K, N, W=1)
            #test_Mult(K, N, W=1)
            #test_DiagGauss(K, N, D=2, W=1, eps=1e-10)
            #test_ZeroMeanGauss(K, N, D=2, W=1, eps=1e-10)
            
            #test_Gauss(K, N, D=2, W=1, eps=1e-10)
            #test_Gauss(K, N, D=2, W=1, eps=1e-10, 
            #    nu=3, kappa=2)
