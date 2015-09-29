import numpy as np
import bnpy
import BarsK10V900
import copy

from bnpy.util.StateSeqUtil import calcHammingDistance, \
    alignEstimatedStateSeqToTruth
from bnpy.birthmove import createSplitStats
from bnpy.birthmove import BLogger, BirthProposalError
from bnpy.viz.PlotUtil import pylab

BLogger.DEFAULTLEVEL = 'print'
b_kwargs = dict(
    b_startLap = 0,
    b_stopLap = -1,
    b_nStuckBeforeQuit = 10,
    b_creationProposalName = 'BregDiv',
    b_Kfresh = 10,
    b_nRefineSteps = 5,
    b_NiterForBregmanKMeansInit = 1,
    b_minRespForEachTargetAtom = 0.1,
    b_minNumAtomsInEachTargetDoc = 100,
    b_minNumAtomsForNewComp = 1,
    b_minNumAtomsForTargetComp = 2,
    b_minPercChangeInNumAtomsToReactivate = 0.01,
    b_debugOutputDir = None,
    )

def main():
    Data = BarsK10V900.get_data(nDocTotal=40, nWordsPerDoc=500)
    pathA = "/data/liv/xdump/BarsK10V900/btest-nDoc=40-b_minPerc=0.1/3/"
    pathB = "/data/liv/xdump/BarsK10V900/btest-nDoc=40-b_minPerc=0.1/2/"
    aM, aLP, aSS, aLscore = loadModelWithLPSSandLscore(Data, pathA, 'A')
    bM, bLP, bSS, bLscore = loadModelWithLPSSandLscore(Data, pathB, 'B')

    bZ = bLP['resp'].argmax(axis=1)
    ktarget = bZ[32]
    targetUID = bSS.uids[ktarget]

    docIDsInClusterWith32 = np.flatnonzero(bZ == ktarget)

    cLP = copy.deepcopy(bLP)
    cLP['resp'][10,:] = 1e-40
    cLP['resp'][10,10] = 1.0
    calcLscoreFromModel(Data, bM, cLP, 'B swap 10')

    cLP = copy.deepcopy(bLP)
    cLP['resp'][32,:] = 1e-40
    cLP['resp'][32,10] = 1.0
    calcLscoreFromModel(Data, bM, cLP, 'B swap 32')

    cLP = copy.deepcopy(bLP)
    cLP['resp'][32,:] = 1e-40
    cLP['resp'][32,10] = 1.0
    cLP['resp'][10,:] = 1e-40
    cLP['resp'][10,10] = 1.0
    calcLscoreFromModel(Data, bM, cLP, 'B swap 10&32')

    cLP = dict(
        resp=np.hstack([bLP['resp'], 1e-40 * np.ones((40,1))])
        )
    cLP['resp'][32,:] = 1e-40
    cLP['resp'][32,-1] = 1.0
    calcLscoreFromModel(Data, bM, cLP, 'B new 32')

    cLP = dict(
        resp=np.hstack([bLP['resp'], 1e-40 * np.ones((40,1))])
        )
    cLP['resp'][10,:] = 1e-40
    cLP['resp'][10,-1] = 1.0
    calcLscoreFromModel(Data, bM, cLP, 'B new 10')

    bnpy.viz.BarsViz.plotExampleBarsDocs(Data, docIDsToPlot=[32], vmax=10)
    pylab.title('Doc 32');

    bnpy.viz.BarsViz.plotExampleBarsDocs(Data, docIDsToPlot=[10], vmax=10)
    pylab.title('Doc 10');

    bnpy.viz.BarsViz.plotExampleBarsDocs(
        Data, docIDsToPlot=docIDsInClusterWith32, vmax=10)
    pylab.title('Cluster 0');
    pylab.show(block=0)

    raw_input("Press any key to continue >>")

    for doc in [1, 2, 3, 10, 32]:
        ktarget = bZ[doc]
        targetUID = bSS.uids[ktarget]
        try:
            propXSS, Info = createSplitStats(
                Data, bM, bLP, curSSwhole=bSS,
                targetUID=targetUID,
                newUIDs=np.arange(100, 110),
                lapFrac=doc,
                b_cleanupMaxNumMergeIters=10,
                b_cleanupMaxNumAcceptPerIter=2,
                **b_kwargs)
        except BirthProposalError as e:
            print e


def calcLscoreFromModel(Data, M, LP=None, label=''):
    M = M.copy()
    if LP is None:
        LP = M.calc_local_params(Data)
    SS = M.get_global_suff_stats(Data, LP, doPrecompEntropy=1)
    M.update_global_params(SS)
    Lscore = M.calc_evidence(SS=SS)
    print "%15s K=%d  L=%.3f Ntotal=%.2f" % (label, SS.K, Lscore, SS.N.sum())
    return M, LP, SS, Lscore        

def loadModelWithLPSSandLscore(Data, path, label=''):
    M = bnpy.load_model(path)
    LP = M.calc_local_params(Data)
    SS = M.get_global_suff_stats(Data, LP, doPrecompEntropy=1)
    M.update_global_params(SS)
    Lscore = M.calc_evidence(SS=SS)
    print "%15s K=%d  L=%.3f Ntotal=%.2f" % (label, SS.K, Lscore, SS.N.sum())
    return M, LP, SS, Lscore

if __name__ == "__main__":
    main()
