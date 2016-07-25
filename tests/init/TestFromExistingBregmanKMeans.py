from __future__ import print_function

import numpy as np
import Symbols as S
import bnpy

from bnpy.init.FromExistingBregman import runKMeans_BregmanDiv_existing

if __name__ == '__main__':
    # Create training set
    Xlist = list()
    for patch_name in ['A', 'B', 'C', 'D', 'E', 'F']:
        X_ND = S.generate_patches_for_symbol(patch_name, 200)
        Xlist.append(X_ND)
    X = np.vstack(Xlist)
    TrainData = bnpy.data.XData(X)
    TrainData.name = 'SimpleSymbols'

    # Train model on this set
    hmodel, RInfo = bnpy.run(
        TrainData, 'FiniteMixtureModel', 'ZeroMeanGauss', 'memoVB',
        initname='bregmankmeans', K=20, 
        nLap=50, moves='merge,shuffle', m_startLap=10,
        ECovMat='eye', sF=0.01)

    # Create test set, with some novel clusters and some old ones
    # Create training set
    Xlist = list()
    for patch_name in ['A', 'B', 'C', 'D',
                       'slash', 'horiz_half', 'vert_half', 'cross']:
        X_ND = S.generate_patches_for_symbol(patch_name, 200)
        Xlist.append(X_ND)
    X = np.vstack(Xlist)
    TestData = bnpy.data.XData(X)
    TestData.name = 'SimpleSymbols'

    # Run FromExistingBregman procedure on test set
    Z, Mu, Lscores = runKMeans_BregmanDiv_existing(
        TestData.X, 10, hmodel.obsModel,
        Niter=0, logFunc=print)
    # Verify recover of novel clusters
    from IPython import embed; embed()