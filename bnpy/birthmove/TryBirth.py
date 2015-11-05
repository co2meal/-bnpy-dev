import argparse
import numpy as np
import os

from bnpy.ioutil.DataReader import loadDataFromSavedTask, loadLPKwargsFromDisk
from bnpy.ioutil.ModelReader import loadModelForLap
from bnpy.birthmove.BCreateOneProposal import \
    makeSummaryForBirthProposal_HTMLWrapper
import bnpy.birthmove.BLogger as BLogger

DefaultBirthArgs = dict(
    b_nStuckBeforeQuit=10,
    b_creationProposalName='bregmankmeans',
    b_Kfresh=10,
    b_nRefineSteps=10,
    b_NiterForBregmanKMeans=1,
    b_minRespForEachTargetAtom=0.1,
    b_minNumAtomsInEachTargetDoc=50,
    b_minNumAtomsForNewComp=1,
    b_minNumAtomsForTargetComp=2,
    b_minPercChangeInNumAtomsToReactivate=0.01,
    b_cleanupMaxNumMergeIters=10,
    b_cleanupMaxNumAcceptPerIter=1,
    b_debugOutputDir='/tmp/',
    b_debugWriteHTML=1,
    Kmax=100,
    )

def tryBirthForTask(
        taskoutpath=None,
        lap=None, lapFrac=0,
        targetUID=0,
        **kwargs):
    '''

    Post Condition
    --------------
    * Logging messages are printed.
    * HTML report is saved.
    '''
    BirthArgs = dict(**DefaultBirthArgs)
    BirthArgs.update(kwargs)

    if lap is not None:
        lapFrac = lap

    Data = loadDataFromSavedTask(taskoutpath)
    curModel, lapFrac = loadModelForLap(taskoutpath, lapFrac)
    LPkwargs = loadLPKwargsFromDisk(taskoutpath)

    curLP = curModel.calc_local_params(Data, **LPkwargs)
    curSS = curModel.get_global_suff_stats(
        Data, curLP,
        trackDocUsage=1, doPrecompEntropy=1, trackTruncationGrowth=1)
    curLscore = curModel.calc_evidence(SS=curSS)
    
    xSS = makeSummaryForBirthProposal_HTMLWrapper(
        Data, curModel, curLP,
        curSSwhole=curSS,
        targetUID=int(targetUID),
        newUIDs=range(curSS.K, curSS.K + int(BirthArgs['b_Kfresh'])),
        LPkwargs=LPkwargs,
        **BirthArgs)

    '''
    propModel, propSS = createBirthProposal(curModel, SS, xSS)
    didAccept, AcceptInfo = evaluateBirthProposal(
        curModel=curModel, curSS=curSS, propModel=propModel, propSS=propSS)
    '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('taskoutpath', type=str)
    parser.add_argument('--lap', type=float, default=None)
    parser.add_argument('--lapFrac', type=float, default=None)
    parser.add_argument('--outputdir', type=str, default='/tmp/')
    args = parser.parse_args()

    BLogger.configure(args.outputdir,
        doSaveToDisk=0,
        doWriteStdOut=1,
        stdoutLevel=0) 
    tryBirthForTask(**args.__dict__)
