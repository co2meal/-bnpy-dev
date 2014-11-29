'''
PlotComps.py

Executable for plotting learned parameters for each component

Usage (command-line)
-------
python -m bnpy.viz.PlotComps dataName aModelName obsModelName algName [kwargs]

'''
from matplotlib import pylab
import numpy as np
import argparse
import os
import sys

import bnpy.ioutil.BNPYArgParser as BNPYArgParser
import bnpy.viz
from bnpy.ioutil import ModelReader


def plotCompsForTask(taskpath, lap=None, figH=None,
                     dataName=None, **kwargs):
  ''' Show plot of learned clusters for single run of saved results on disk
  '''
  ## Verify given absolute path is valid.
  taskpath_originalarg = taskpath
  if not os.path.isdir(taskpath) and not taskpath.startswith(os.path.sep):
    # Fallback: prepend BNPYOUTDIR to handle "shortcut" names
    taskpath = os.path.join(os.environ['BNPYOUTDIR'], taskpath)
  if not os.path.exists(taskpath):
    raise ValueError('Task path not found: \n' + taskpath_originalarg)

  ## Read dataName from the taskpath
  if dataName is None:
    dataName = taskpath.replace(os.environ['BNPYOUTDIR'], 
                                '').split(os.path.sep)[0]

  ## Load hmodel stored at specified lap
  queryLap = lap
  hmodel, lap = ModelReader.loadModelForLap(taskpath, queryLap)
  if queryLap is not None and not np.allclose(lap, queryLap):
    print 'Query lap %.2f unavailable. Using %.2f instead.' \
           % (queryLap, lap)

  obsName = hmodel.getObsModelName()
  if obsName.count('Gauss'):
    if hmodel.obsModel.D > 2:
      bnpy.viz.GaussViz.plotCovMatFromHModel(hmodel, figH=figH)
    elif hmodel.obsModel.D == 2:
      bnpy.viz.GaussViz.plotGauss2DFromHModel(hmodel, figH=figH)
    elif hmodel.obsModel.D == 1:
      bnpy.viz.GaussViz.plotGauss1DFromHModel(hmodel, figH=figH)
  elif obsName.count('Mult'):
    if dataName.lower().count('bars') > 0:
      bnpy.viz.BarsViz.plotBarsFromHModel(hmodel, figH=figH) 

def plotCompsForJob(jobpath='', taskids=[1], lap=None, 
                    **kwargs):
  ''' Show plot of learned clusters from run(s) saved results on disk
  '''

  ## Verify given absolute path is valid.
  jobpath_originalarg = jobpath
  if not os.path.isdir(jobpath):
    # Fallback: try to prepend BNPYOUTDIR to handle "shortcut" names
    jobpath = os.path.join(os.environ['BNPYOUTDIR'], jobpath)
  if not os.path.isdir(jobpath):
    raise ValueError('Not valid path: ' + jobpath_originalarg)

  taskids = BNPYArgParser.parse_task_ids(jobpath, taskids)
  for taskid in taskids:
    taskpath = os.path.join(jobpath, str(taskid))
    plotCompsForTask(taskpath, lap=lap, **kwargs)
  if 'block' in kwargs:
    pylab.show(block=kwargs['block'])

def parseArgs(**kwargs):
  ''' Read args from stdin into defined dict fields
  '''
  parser = argparse.ArgumentParser()
  parser.add_argument('dataName')
  parser.add_argument('jobname')
  parser.add_argument('--lap', default=None, type=float)
  parser.add_argument('--taskids', type=str, default=None,
         help="int ids of trials/runs to plot from given job." \
              + " Example: '4' or '1,2,3' or '2-6'.")
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

