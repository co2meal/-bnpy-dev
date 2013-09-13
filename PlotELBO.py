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
import os

XLabelMap = dict(laps='num pass thru data',
                  iters='num steps in alg',
                  times='elapsed time (sec)'
                  )
                  
def parse_task_ids(args):
  if args.taskids is None:
    fulltaskpaths = glob.glob(args.jobpath)
    print fulltaskpaths
    taskids = [os.path.split(tpath)[-1] for tpath in fulltaskpaths]
  elif args.taskids.count(',') > 0:
    taskids = [t for t in args.taskids.split(',')]
  elif args.taskids.count('-') > 0:
    fields = args.taskids.split('-')
    if not len(fields)==2:
      raise ValueError("Bad taskids specification")
    fields = np.int32(np.asarray(fields))
    taskids = np.arange(fields[0],fields[1]+1)
    taskids = [str(t) for t in taskids]
  else:
    taskids = args.taskids
  return taskids
                  
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('jobpath', type=str, default=None,
        help='absolute path to directory where bnpy model saved ' + \
              'Example: /home/myusername/bnpyresults/StarData/MixModel/ZMGauss/EM/abc/')
  parser.add_argument('--taskids', type=str, default='None',
        help="int ids of the tasks (individual runs) of the given job to plot." +\
              'Ex: "1" or "3" or "1,2,3" or "1-6"')
  parser.add_argument('--xvar', type=str, default='laps',
        help="name of x axis variable to plot. one of {iters,laps,times}")
  parser.add_argument('--savefilename', type=str, default=None,
        help="absolute path to directory to save figure")
  args = parser.parse_args()

  if not os.path.exists(args.jobpath):
    raise ValueError("No such path: %s" % (args.jobpath))
  taskids = parse_task_ids(args)
  
  for taskid in taskids:
    xs = np.loadtxt(os.path.join(args.jobpath, taskid, args.xvar+'.txt'))
    ys = np.loadtxt(os.path.join(args.jobpath, taskid, 'evidence.txt'))
    pylab.plot(xs, ys, '.-', markersize=10, linewidth=2)
    
  pylab.xlabel(XLabelMap[args.xvar])
  pylab.ylabel('log evidence')
  
  if args.savefilename is not None:
    pylab.show(block=False)
    pylab.savefig(args.savefilename)
  else:
    pylab.show(block=True)
  
if __name__ == "__main__":
  main()

