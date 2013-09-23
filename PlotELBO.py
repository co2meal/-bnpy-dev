'''
PlotELBO.py

Basic executable for plotting the ELBO vs. time/iterations/num. data items processed

Usage
-------
python PlotELBO.py /path/to/bnpy/saved/job/ [options]

Options
--------
--savefilename : absolute path to directory to save figure
                 Ex: ~/Desktop/myfigure.pdf or ~/Desktop/myfigure.png
                 
--xvar : name of x axis variable to plot. one of {iters,laps,times}
--taskids : ids of the tasks (individual runs) of the given job to plot.
             Ex: "1" or "3" or "1,2,3" or "1-6"
'''
from matplotlib import pylab
import numpy as np
import argparse
import glob
import os

Colors = [(1,0,0), (1,0,1), (0,1,0), (0,1,1), (0,0,1), (1,0.6,0)]

XLabelMap = dict(laps='num pass thru data',
                  iters='num steps in alg',
                  times='elapsed time (sec)'
                  )
                  
def parse_task_ids(jobpath, taskids):
  if taskids is None:
    fulltaskpaths = glob.glob(os.path.join(jobpath,'*'))
    taskids = [os.path.split(tpath)[-1] for tpath in fulltaskpaths]
  elif taskids.count(',') > 0:
    taskids = [t for t in taskids.split(',')]
  elif taskids.count('-') > 0:
    fields = taskids.split('-')
    if not len(fields)==2:
      raise ValueError("Bad taskids specification")
    fields = np.int32(np.asarray(fields))
    taskids = np.arange(fields[0],fields[1]+1)
    taskids = [str(t) for t in taskids]
  else:
    taskids = taskids
  return taskids
             
             
def plot_all_tasks_for_job(jobpath, args, jobname=None, color=None):
  ''' Create line plot in current matplotlib figure
      for each task/run of the designated jobpath
  '''
  if not os.path.exists(jobpath):
    raise ValueError("No such path: %s" % (jobpath))
  
  taskids = parse_task_ids(jobpath, args.taskids)
    
  xAll = list()
  yAll = list()
  xLocs = list()
  yLocs = list()
  for tt, taskid in enumerate(taskids):
    xs = np.loadtxt(os.path.join(jobpath, taskid, args.xvar+'.txt'))
    ys = np.loadtxt(os.path.join(jobpath, taskid, 'evidence.txt'))
    if color is None:
      pylab.plot(xs, ys, '.-', markersize=10, linewidth=2)
    else:
      if tt == 0:
        label = jobname
      else:
        label = None
      pylab.plot(xs, ys, '.-', markersize=10, linewidth=2, color=color, label=label)
    if len(ys) > 0:
      xLocs.append(xs[-1])
      yLocs.append(ys[-1])
      yAll.extend(ys[1:])
      xAll.extend(xs[1:])
      
  # Zoom in to the useful part of the ELBO trace
  if len(yAll) > 0:
    ymin = np.percentile(yAll, 1)
    ymax = np.max(yAll)
    blankmargin = 0.03*(ymax - ymin)
    pylab.ylim( [ymin, ymax + blankmargin])
    
  if args.doShowTaskNums and len(taskids) > 0:
    yNudge = 0.03 * (ymax - ymin)
    for tt in range(len(taskids)):
      xNudge = 0.05 * (np.max(xAll) - np.min(xAll))
      xNudge += 0.25*np.random.randn() * xNudge
      pylab.text( xLocs[tt] - xNudge, yLocs[tt] - yNudge, str(taskids[tt]))
  pylab.xlabel(XLabelMap[args.xvar])
  pylab.ylabel('log evidence')
  
                  
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('jobpath', type=str, default=None,
        help='absolute path to directory where bnpy model saved ' + \
              'Example: /home/myusername/bnpyresults/StarData/MixModel/ZMGauss/EM/abc/')
  parser.add_argument('--jobnames', type=str, default=None)
  parser.add_argument('--taskids', type=str, default=None,
        help="int ids of the tasks (individual runs) of the given job to plot." +\
              'Ex: "1" or "3" or "1,2,3" or "1-6"')
  parser.add_argument('--xvar', type=str, default='laps',
        help="name of x axis variable to plot. one of {iters,laps,times}")
  parser.add_argument('--savefilename', type=str, default=None,
        help="absolute path to directory to save figure")
  parser.add_argument('--doShowTaskNums', action='store_true', default=False,
        help="if present, do show task numbers next to corresponding line plot")
  args = parser.parse_args()

  if args.jobnames is None:
    plot_all_tasks_for_job(args.jobpath, args)
  else:
    jobnames = args.jobnames.split(',')
    for jj,jobname in enumerate(jobnames):
      cID = jj % len(Colors)
      curjobpath = os.path.join(args.jobpath, jobname)
      plot_all_tasks_for_job(curjobpath, args, jobname=jobname, color=Colors[cID])    
    pylab.legend()
  
  if args.savefilename is not None:
    pylab.show(block=False)
    pylab.savefig(args.savefilename)
  else:
    pylab.show(block=True)
  
if __name__ == "__main__":
  main()

