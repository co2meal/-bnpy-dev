import argparse
import numpy as np
import os

from bnpy.ioutil.DataReader import loadDataFromSavedTask, loadLPKwargsFromDisk
from bnpy.ioutil.ModelReader import loadModelForLap

def tryBirthForTask(taskoutpath=None, lap=None, lapFrac=0, **kwargs):
    '''

    Post Condition
    --------------
    * Logging messages are printed.
    * HTML report is saved.
    '''
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
    print curLscore
    '''
    xSS = makeSummaryForBirthProposal(
        Dslice=Data, curLPslice=LP, ...)

    propModel, propSS = createBirthProposal(curModel, SS, xSS)
    didAccept, AcceptInfo = evaluateBirthProposal(
        curModel=curModel, curSS=curSS, propModel=propModel, propSS=propSS)
    '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('taskoutpath', type=str)
    parser.add_argument('--lap', type=float, default=None)
    parser.add_argument('--lapFrac', type=float, default=None)
    args = parser.parse_args()
    tryBirthForTask(**args.__dict__)
