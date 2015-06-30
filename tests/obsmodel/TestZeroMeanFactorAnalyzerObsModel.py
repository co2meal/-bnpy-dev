import numpy as np
import unittest
import bnpy

if __name__ == '__main__':
    import D3C2K2_ZM
    Data = D3C2K2_ZM.get_data(nObsTotal=1e3)
    hModel, RInfo = bnpy.run(Data, 'FiniteMixtureModel','ZeroMeanFactorAnalyzer', 'VB',
                             C=2, nLap=100, K=3, printEvery=1)
    import sys
    sys.path.append('datasets')
    kA = 0
    kB = 2
    SS = RInfo['SS']
    WMean, WCov, hShape, hInvScale, PhiShape, PhiInvScale, aCov = \
        hModel.obsModel.calcPostParamsForComp(SS, kA=kA, kB=kB)
    print SS.N
    SS.mergeComps(kA, kB)
    hModel.update_global_params(SS)
    LP = hModel.calc_local_params(Data, None)
    SS = hModel.get_global_suff_stats(Data, LP)
    evBound = hModel.calc_evidence(Data, SS, LP)
    print evBound

