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
                 iters='num alg steps',
                 times='elapsed time (sec)'
                )  
YLabelMap = dict(evidence='train objective',
                 K='num topics K',
                 )
     
def plotJobsThatMatchKeywords(jpathPattern='/tmp/', **kwargs):
  ''' Create line plots for all jobs matching pattern and provided keyword args

      Example
      ---------
      plotJobsThatMatchKeywords('MyData', '
  '''
  if not jpathPattern.startswith(os.path.sep):
    jpathPattern = os.path.join(os.environ['BNPYOUTDIR'], jpathPattern)
  jpaths, legNames = filterJobs(jpathPattern, **kwargs)
  plotJobs(jpaths, legNames, **kwargs)

def plotJobs(jpaths, legNames, styles=None, density=2,
             xvar='laps', yvar='evidence', loc='upper right',
             taskids=None, savefilename=None, tickfontsize=None,
             bbox_to_anchor=None, **kwargs):
  ''' Create line plots for provided jobs 
  '''
  nLines = len(jpaths)
  nLeg = len(legNames)
  assert nLines == nLeg
  for lineID in xrange(nLines):
    if styles is None:
      curStyle = dict(colorID=lineID)
    else:
      curStyle = styles[lineID]

    plot_all_tasks_for_job(jpaths[lineID], legNames[lineID], 
                           xvar=xvar, yvar=yvar,
                           taskids=taskids, density=density, **curStyle)
  if loc is not None:
    pylab.legend(loc=loc, bbox_to_anchor=bbox_to_anchor)  
  if tickfontsize is not None:
    pylab.tick_params(axis='both', which='major', labelsize=tickfontsize)

  if savefilename is not None:
    try:
      pylab.show(block=False)
    except TypeError:
      pass # when using IPython notebook
    pylab.savefig(savefilename, bbox_inches='tight', pad_inches=0)
  else:
    try:
      pylab.show(block=True)
    except TypeError:
      pass # when using IPython notebook
        

def plot_all_tasks_for_job(jobpath, label, taskids=None,
                                           lineType='.-',
                                           color=None,
                                           colorID=0,
                                           density=2,
                                           yvar='evidence',
                                           markersize=10,
                                           linewidth=2,
                                           xvar='laps', **kwargs):
  ''' Create line plot in current figure for each task/run of jobpath
  '''
  if not os.path.exists(jobpath):
    raise ValueError("PATH NOT FOUND: %s" % (jobpath))
  if color is None:
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
        try:
          xs, ys = loadXYFromTopicModelSummaryFiles(jobpath, taskid)
        except ValueError:
          raise e

    ## Make sure that xs are sorted
    if xvar == 'laps':
      diff = xs[1:] - xs[:-1]
      goodIDs = np.flatnonzero(diff > 0)
      if len(goodIDs) < xs.size-1:
        print 'WARNING: looks like we had multiple runs writing to this file!'
        xs = np.hstack([xs[goodIDs], xs[-1]])
        ys = np.hstack([ys[goodIDs], ys[-1]])

    if xvar == 'laps' and xs.size > 15:
      curDensity = xs.size / (xs[-1] - xs[0])
      while curDensity > density:
        xs = xs[::2]
        ys = ys[::2]
        curDensity = xs.size / (xs[-1] - xs[0])

    plotargs = dict(markersize=markersize, linewidth=linewidth, label=None,
                    color=color, markeredgecolor=color)
    plotargs.update(kwargs)
    if tt == 0:
      plotargs['label'] = label
    pylab.plot(xs, ys, lineType, **plotargs)


  if xvar == 'laps' and yvar == 'evidence':
    if np.sum(xs > 5.0) > 5:
      ymin = ys.max()
      ymax = ys.min()
      for line in pylab.gca().get_lines():
        xd = line.get_xdata()
        yd = line.get_ydata()
        loc = np.searchsorted(xd, 2)
        ymin = np.minimum(ymin, np.percentile(yd[loc:], 2.5))
        ymax = np.maximum(ymax, yd[loc:].max())
      pylab.ylim([ymin, ymax + 0.1*(ymax-ymin)])

  pylab.xlabel(XLabelMap[xvar])
  pylab.ylabel(YLabelMap[yvar])
   



def loadXYFromTopicModelSummaryFiles(jobpath, taskid, xvar='laps', yvar='K'):
  ''' Load x and y variables for line plots from TopicModel files
  '''
  ypath = os.path.join(jobpath, taskid, 'predlik-' + yvar + '.txt')
  if not os.path.exists(ypath):
    raise ValueError('No TopicModel summary text files found')
  lappath =  os.path.join(jobpath, taskid, 'predlik-lapTrain.txt')
  xs = np.loadtxt(lappath)
  ys = np.loadtxt(ypath)
  return xs, ys

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

