import numpy as np
import unittest
import bnpy

if __name__ == '__main__':
    bnpy.run('AsteriskK8', 'DPMixtureModel', 'Gauss', 'moVB', K=10, nBatch=10, nLap=100,moves='birth,merge,delete')

    # import D3C2K2_ZM
    # Data = D3C2K2_ZM.get_data(nObsTotal=1e3)
    # hModel, RInfo = bnpy.run(Data, 'FiniteMixtureModel','ZeroMeanFactorAnalyzer', 'VB',
    #                          C=2, nLap=20, K=3, printEvery=1)
    #
    # kA = 1
    # kB = 2
    # SS = RInfo['SS']
    # # WMean, WCov, hShape, hInvScale, PhiShape, PhiInvScale, aCov = \
    # #     hModel.obsModel.calcPostParamsForComp(SS, kA=kA, kB=kB)
    # # print SS.N
    # SS.mergeComps(kA, kB)
    # hModel.update_global_params(SS)
    # LP = hModel.calc_local_params(Data, None)
    # SS = hModel.get_global_suff_stats(Data, LP)
    # evBound = hModel.calc_evidence(Data, SS, LP)
    # print "New ELBO after merge move: %f" % evBound


    import D2C1K2_ZM
    Data = D2C1K2_ZM.get_data(nObsTotal=1e3)
    hModel, RInfo = bnpy.run(Data, 'FiniteMixtureModel','ZeroMeanFactorAnalyzer', 'VB',
                             C=1, nLap=20, K=3, printEvery=1)

    kA = 0
    kB = 2
    SS = RInfo['SS']
    # WMean, WCov, hShape, hInvScale, PhiShape, PhiInvScale, aCov = \
    #     hModel.obsModel.calcPostParamsForComp(SS, kA=kA, kB=kB)
    print SS.N
    SS.mergeComps(kA, kB)
    hModel.update_global_params(SS)
    LP = hModel.calc_local_params(Data, None)
    SS = hModel.get_global_suff_stats(Data, LP)
    evBound = hModel.calc_evidence(Data, SS, LP)
    print "New ELBO after merge move: %f" % evBound