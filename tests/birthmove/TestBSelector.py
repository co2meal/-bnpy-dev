import numpy as np
import unittest
import bnpy

from bnpy.util.RandUtil import rotateCovMat
from bnpy.viz.PlotUtil import pylab
from bnpy.ioutil.BNPYArgParser import arglist_to_kwargs
import sys

def makeDataset(K=5, Nk=100, Nvec=None, 
                VarMajor=1.0, VarMinorRatio=1.0/25, **kwargs):
    if Nvec is None:
        Nvec = Nk * np.ones(K)
    Nvec = np.asarray(Nvec, dtype=np.int32)

    PRNG = np.random.RandomState(int(20*VarMajor))
    # Create basic 2D cov matrix with major axis much longer than minor one
    SigmaBase = np.asarray([[VarMajor, 0], [0, VarMinorRatio*VarMajor]])

    # Create several Sigmas by rotating this basic covariance matrix
    Z = [None for k in range(K)]
    X = [None for k in range(K)]
    for k in xrange(K):
        Sigma_k = rotateCovMat(SigmaBase, k * np.pi / K)
        Z[k] = k * np.ones(Nvec[k])
        X[k] = PRNG.multivariate_normal([0, 0], Sigma_k, size=Nvec[k])
        
    Z = np.hstack(Z)
    X = np.vstack(X)
    return bnpy.data.XData(X=X, TrueZ=Z)

def makeInitModelWithMergedComps(model, Data, compsToMerge=[(0,1)], **kwargs):
    initZ = Data.TrueParams['Z'].copy()
    for knew, ktuple in enumerate(compsToMerge):
        for k in ktuple:
            initZ[initZ == k] = 1000 + knew
    # relabel initZ by unique entries
    uZ = np.unique(initZ)
    initresp = np.zeros((Data.nObs, uZ.size))
    for uid, k in enumerate(uZ):
        initresp[initZ == k, uid] = 1.0
    
    initLP = dict(resp=initresp)
    initSS = model.get_global_suff_stats(Data, initLP)

    tmpmodel = model.copy()
    tmpmodel.update_global_params(initSS)
    for aiter in range(10):
        LP = tmpmodel.calc_local_params(Data)
        SS = tmpmodel.get_global_suff_stats(Data, LP)
        tmpmodel.update_global_params(SS)
    return tmpmodel, SS


def testBSelectMethod_Ldata(**kwargs):
    Data = makeDataset(**kwargs)
    PriorArgs = dict(ECovMat='diagcovdata', sF=1.0)
    PriorArgs.update(kwargs)
    allocModel = bnpy.allocmodel.DPMixtureModel(
        'VB', gamma0=1.0)
    obsModel = bnpy.obsmodel.ZeroMeanGaussObsModel(
        'VB', Data=Data, **PriorArgs)
    model = bnpy.HModel(allocModel, obsModel)

    tmpmodel, SS = makeInitModelWithMergedComps(model, Data, **kwargs)
    Lvec = tmpmodel.obsModel.calcELBO_Memoized(SS=SS, returnVec=1)

    print Lvec
    bnpy.viz.PlotComps.plotCompsFromHModel(tmpmodel)
    pylab.show()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ECovMat', default='diagcovdata')
    args, unkList = parser.parse_known_args()
    kwargs = arglist_to_kwargs(unkList, doConvertFromStr=False)
    kwargs.update(args.__dict__)
    testBSelectMethod_Ldata(**kwargs)
    
