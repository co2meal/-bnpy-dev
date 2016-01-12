'''
PlotComps.py

Executable for plotting learned parameters for each component

Usage (command-line)
-------
python -m bnpy.viz.PlotComps dataName aModelName obsModelName algName [kwargs]

'''
import numpy as np
import argparse
import os
import sys

from PlotUtil import pylab
import bnpy.ioutil.BNPYArgParser as BNPYArgParser
import bnpy.viz
from bnpy.ioutil import ModelReader
from bnpy.viz.TaskRanker import rankTasksForSingleJobOnDisk
from bnpy.viz.PlotTrace import taskidsHelpMsg
from bnpy.viz.PrintTopics import uidsAndCounts2strlist
from bnpy.ioutil.DataReader import loadDataFromSavedTask

def plotCompsFromHModel(hmodel, **kwargs):
    """ Show plot of learned clusters for provided model.
    """
    obsName = hmodel.getObsModelName()
    if obsName.count('Gauss'):
        if hmodel.obsModel.D > 2:
            bnpy.viz.GaussViz.plotCovMatFromHModel(hmodel, **kwargs)
        elif hmodel.obsModel.D == 2:
            bnpy.viz.GaussViz.plotGauss2DFromHModel(hmodel, **kwargs)
        elif hmodel.obsModel.D == 1:
            bnpy.viz.GaussViz.plotGauss1DFromHModel(hmodel, **kwargs)
    elif obsName.count('Bern'):
        if 'vocabList' in kwargs and kwargs['vocabList'] is not None:
            bnpy.viz.PrintTopics.plotCompsFromHModel(hmodel, **kwargs)
        else:
            bnpy.viz.BernViz.plotCompsFromHModel(hmodel, **kwargs)
    elif obsName.count('Mult'):
        if 'vocabList' in kwargs and kwargs['vocabList'] is not None:
            bnpy.viz.PrintTopics.plotCompsFromHModel(hmodel, **kwargs)
        else:
            bnpy.viz.BarsViz.plotBarsFromHModel(hmodel, **kwargs)



def plotCompsForTask(taskpath, lap=None,
                     dataName=None, **kwargs):
    ''' Show plot of learned clusters for single run of saved results on disk
    '''
    # Verify given absolute path is valid.
    taskpath_originalarg = taskpath
    if not os.path.isdir(taskpath) and not taskpath.startswith(os.path.sep):
        # Fallback: prepend BNPYOUTDIR to handle "shortcut" names
        taskpath = os.path.join(os.environ['BNPYOUTDIR'], taskpath)
    if not os.path.exists(taskpath):
        raise ValueError('Task path not found: \n' + taskpath_originalarg)

    # Read dataName from the taskpath
    if dataName is None:
        dataName = taskpath.replace(os.environ['BNPYOUTDIR'],
                                    '').split(os.path.sep)[0]

    # Hack to load vocabulary
    vocabList = None
    Data = loadDataFromSavedTask(taskpath)
    if hasattr(Data, 'vocabList'):
        vocabList = Data.vocabList

    # Load hmodel stored at specified lap
    queryLap = lap
    hmodel, lap = ModelReader.loadModelForLap(taskpath, queryLap)
    if queryLap is not None and not np.allclose(lap, queryLap):
        print 'Query lap %.2f unavailable. Using %.2f instead.' \
            % (queryLap, lap)

    plotCompsFromHModel(hmodel, vocabList=vocabList, **kwargs)


def plotCompsForJob(jobpath='', taskids=[1], lap=None,
                    **kwargs):
    ''' Show plot of learned clusters from run(s) saved results on disk
    '''

    # Verify given absolute path is valid.
    jobpath_originalarg = jobpath
    if not os.path.isdir(jobpath):
        # Fallback: try to prepend BNPYOUTDIR to handle "shortcut" names
        jobpath = os.path.join(os.environ['BNPYOUTDIR'], jobpath)
    if not os.path.isdir(jobpath):
        raise ValueError('Not valid path: ' + jobpath_originalarg)

    print taskids
    taskids = BNPYArgParser.parse_task_ids(jobpath, taskids)
    print taskids
    for tt, taskid in enumerate(taskids):
        if tt == 0 and isinstance(taskid, str):
            if taskid.startswith('.'):
                rankTasksForSingleJobOnDisk(jobpath)
        taskpath = os.path.join(jobpath, str(taskid))
        plotCompsForTask(taskpath, lap=lap, **kwargs)
    if 'block' in kwargs:
        pylab.show(block=kwargs['block'])


def plotCompsFromSS(hmodel, SS, outfilepath=None, **kwargs):
    ''' Plot components defined by provided summary statistics.

    Provided hmodel is just used for its hyperparameters, not current comps.

    Post condition
    --------------
    Create matplotlib figure showing (subset of) SS.K components.
    '''
    if 'xlabels' not in kwargs:
        xlabels = uidsAndCounts2strlist(SS)
        kwargs['xlabels'] = xlabels    
    tmpModel = hmodel.copy()
    tmpModel.obsModel.update_global_params(SS)
    plotCompsFromHModel(tmpModel, **kwargs)
    if outfilepath is not None:
        pylab.savefig(outfilepath)
        pylab.close('all')
        print 'Wrote: %s' % (outfilepath)

def parseArgs(**kwargs):
    ''' Read args from stdin into defined dict fields
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('dataName')
    parser.add_argument('jobname')
    parser.add_argument('--lap', default=None, type=float)
    parser.add_argument('--taskids', type=str, default=None,
                        help=taskidsHelpMsg)
    args = parser.parse_args()
    jobpath = os.path.join(os.environ['BNPYOUTDIR'],
                           args.dataName,
                           args.jobname)
    argDict = args.__dict__
    argDict['jobpath'] = jobpath
    return argDict

if __name__ == "__main__":
    argDict = parseArgs()
    plotCompsForJob(block=1, **argDict)
