'''
PlotELBO.py

Executable for plotting the learning objective function (log evidence)
  vs. time/number of passes thru data (laps)

Usage (command-line)
-------
python -m bnpy.viz.PlotELBO dataName aModelName obsModelName algName [kwargs]
'''
from matplotlib import pylab
import numpy as np
import argparse
import glob
import os
import scipy.io

from bnpy.ioutil import BNPYArgParser
from JobFilter import filterJobs

import matplotlib
matplotlib.rcParams['text.usetex'] = False

Colors = [(0,0,0), # black
          (0,0,1), # blue
          (1,0,0), # red
          (0,1,0.25), # green (darker)
          (1,0,1), # magenta
          (0,1,1), # cyan
          (1,0.6,0), #orange
         ]

XLabelMap = dict(laps='num pass thru data',
                 iters='num steps in alg',
                 times='elapsed time (sec)'
                )  
YLabelMap = dict(evidence='log evidence',
                 K='num components',
                 )
     
def plotJobsThatMatchKeywords(jpathPattern='/tmp/', 
         xvar='laps', yvar='evidence', loc='upper right',
         taskids=None, savefilename=None, 
         **kwargs):
  ''' Create line plots for all jobs matching pattern and provided keyword args

      Example
      ---------
      plotJobsThatMatchKeywords('MyData', '
  '''
  if not jpathPattern.startswith(os.path.sep):
    jpathPattern = os.path.join(os.environ['BNPYOUTDIR'], jpathPattern)
  jpaths, legNames = filterJobs(jpathPattern, **kwargs)
  nLines = len(jpaths)
  for lineID in xrange(nLines):
    plot_all_tasks_for_job(jpaths[lineID], legNames[lineID], 
                           xvar=xvar, yvar=yvar,
                           taskids=taskids, colorID=lineID)
  if loc is not None:
    pylab.legend(loc=loc)  
  if savefilename is not None:
    pylab.show(block=False)
    pylab.savefig(args.savefilename)
  else:
    try:
      pylab.show(block=True)
    except TypeError:
      pass # when using IPython notebook
        

def plot_all_tasks_for_job(jobpath, label, taskids=None,
                                             colorID=0,
                                             yvar='evidence',
                                             xvar='laps'):
  ''' Create line plot in current figure for each task/run of jobpath
  '''
  if not os.path.exists(jobpath):
    raise ValueError("PATH NOT FOUND: %s" % (jobpath))
  
  color = Colors[ colorID % len(Colors)]
  taskids = BNPYArgParser.parse_task_ids(jobpath, taskids)

  for tt, taskid in enumerate(taskids):
    try:
      xs = np.loadtxt(os.path.join(jobpath, taskid, xvar+'.txt'))
      ys = np.loadtxt(os.path.join(jobpath, taskid, yvar+'.txt'))
    except IOError as e:
      try:
        xs, ys = loadXYFromTopicModelFiles(jobpath, taskid)
      except ValueError:
        raise e

    plotargs = dict(markersize=10, linewidth=2, label=None,
                    color=color, markeredgecolor=color)
    if tt == 0:
      plotargs['label'] = label
    pylab.plot(xs, ys, '.-', **plotargs)

  pylab.xlabel(XLabelMap[xvar])
  pylab.ylabel(YLabelMap[yvar])
   

def loadXYFromTopicModelFiles(jobpath, taskid, xvar='laps', yvar='K'):
  ''' Load x and y variables for line plots from TopicModel files
  '''
  tmpathList = glob.glob(os.path.join(jobpath, taskid, 'Lap*TopicModel.mat'))
  if len(tmpathList) < 1:
    raise ValueError('No TopicModel.mat files found')
  tmpathList.sort() # ascending, from lap 0 to lap 1 to lap 100 to ...
  basenames = [x.split(os.path.sep)[-1] for x in tmpathList];
  laps = np.asarray([float(x[3:11]) for x in basenames]);
  Ks = np.zeros_like(laps)

  for tt, tmpath in enumerate(tmpathList):
    if yvar == 'K':
      Q = scipy.io.loadmat(tmpath, variable_names=['K', 'probs'])
      try:
        Ks[tt] = Q['K']
      except KeyError:
        Ks[tt] = Q['probs'].size
    else:
      raise ValueError('Unknown yvar type for topic model: ' + yvar)
  return laps, Ks
  
def parse_args():
  ''' Returns Namespace of parsed arguments retrieved from command line
  '''
  parser = argparse.ArgumentParser()
  parser.add_argument('dataName', type=str, default='AsteriskK8')
  parser.add_argument('jpath', type=str, default='demo*')

  parser.add_argument('--xvar', type=str, default='laps',
        help="name of x axis variable to plot. one of {iters,laps,times}")
  parser.add_argument('--taskids', type=str, default=None,
        help="int ids of tasks (trials/runs) to plot from given job." \
              + " Example: '4' or '1,2,3' or '2-6'.")
  parser.add_argument('--savefilename', type=str, default=None,
        help="location where to save figure (absolute path directory)")

  args, unkList = parser.parse_known_args()

  argDict = BNPYArgParser.arglist_to_kwargs(unkList)
  argDict.update(args.__dict__)
  argDict['jpathPattern'] = os.path.join(os.environ['BNPYOUTDIR'],
                                   args.dataName,
                                   args.jpath)
  del argDict['dataName']
  del argDict['jpath']
  return argDict

if __name__ == "__main__":
  argDict = parse_args()
  plotJobsThatMatchKeywords(**argDict)

