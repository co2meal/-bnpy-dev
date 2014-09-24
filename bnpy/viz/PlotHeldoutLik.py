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
import os
import glob
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

XLabelMap = dict(laps='num pass thru train data',
                )  
YLabelMap = dict(evidence='per-token heldout log-lik',
                 )
     
def plotJobsThatMatchKeywords(jpathPattern='/tmp/', 
         xvar='laps', yvar='evidence', 
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
                           taskids=taskids, colorID=lineID)

  pylab.legend(loc='best')  
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
    taskoutpath = os.path.join(jobpath, taskid)
    hpaths = glob.glob(os.path.join(taskoutpath, '*HeldoutLik.mat'))
    hpaths.sort()
    basenames = [x.split(os.path.sep)[-1] for x in hpaths];
    laps = np.asarray([float(x[3:11]) for x in basenames]);

    ys = np.zeros_like(laps)
    for ii, hpath in enumerate(hpaths):
      MatVars = scipy.io.loadmat(hpath)
      ys[ii] = float(MatVars['avgPredLL'])

    plotargs = dict(markersize=10, linewidth=2, label=None,
                    color=color, markeredgecolor=color)
    if tt == 0:
      plotargs['label'] = label
    pylab.plot(laps, ys, '.-', **plotargs)

  pylab.xlabel(XLabelMap[xvar])
  pylab.ylabel(YLabelMap[yvar])
   
  
def parse_args():
  ''' Returns Namespace of parsed arguments retrieved from command line
  '''
  parser = argparse.ArgumentParser()
  parser.add_argument('dataName', type=str, default='AsteriskK8')
  parser.add_argument('jpath', type=str, default='demo*')

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

