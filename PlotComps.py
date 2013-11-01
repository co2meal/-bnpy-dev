'''
PlotELBO.py

Basic executable for plotting learned parameters for each component

Usage
-------
python PlotComps.py dataName allocModelName obsModelName algName [options]

Options
--------
--savefilename : absolute path to directory to save figures
                 Ex: ~/Desktop/myfigure.pdf or ~/Desktop/myfigure.png
                 
--taskids : ids of the tasks (individual runs) of the given job to plot.
             Ex: "1" or "3" or "1,2,3" or "1-6"
'''
from matplotlib import pylab
import numpy as np
import argparse
import os
import sys
import bnpy
import PlotELBO

def loadData(jobpath):
  ''' Load in bnpy Data obj associated with given learning task.
      TODO: make dataseed work
  '''
  bnpyoutdir = os.environ['BNPYOUTDIR']
  subdirpath = jobpath[len(bnpyoutdir):]
  fields = subdirpath.split(os.path.sep)
  dataname = fields[0]
  sys.path.append(os.environ['BNPYDATADIR'])
  datamodulepath = os.path.join(os.environ['BNPYDATADIR'], dataname+".py")
  if not os.path.exists(datamodulepath):
    raise ValueError("Could not find data %s" % (dataname))
  datamod = __import__(dataname, fromlist=[])
  return datamod.get_data()
  
def plotData(Data, nObsPlot=5000):
  ''' Plot data items, at most nObsPlot distinct points (for quick rendering)
  '''
  if type(Data) == bnpy.data.XData:
    PRNG = np.random.RandomState(nObsPlot)
    pIDs = PRNG.permutation(Data.nObs)[:nObsPlot]
    pylab.plot(Data.X[pIDs,0], Data.X[pIDs,1], 'k.')  
  
def main():
  parser = argparse.ArgumentParser()  
  parser.add_argument('dataName', type=str,
        help='name of python script that produces data to analyze.')
  parser.add_argument('allocModelName', type=str,
        help='name of allocation model. {MixModel, DPMixModel}')
  parser.add_argument('obsModelName', type=str,
        help='name of observation model. {Gauss, ZMGauss}')
  parser.add_argument('algName', type=str,
        help='name of learning algorithms to consider, {EM, VB, moVB, soVB}.')
  parser.add_argument('--jobname', type=str, default='defaultjob',
        help='name of experiment whose results should be plotted')
        
  parser.add_argument('--taskids', type=str, default=None,
        help="int ids for tasks (individual runs) of the given job to plot." + \
              'Ex: "1" or "3" or "1,2,3" or "1-6"')
  parser.add_argument('--savefilename', type=str, default=None,
        help="absolute path to directory to save figure")
  parser.add_argument('--doPlotData', action='store_true', default=False,
        help="if present, also plot training data")
  parser.add_argument('--iterid', type=int, default=None)
  args = parser.parse_args()


  rootpath = os.path.join(os.environ['BNPYOUTDIR'], args.dataName, \
                              args.allocModelName, args.obsModelName)
  jobpath = os.path.join(rootpath, args.algName, args.jobname)

  if not os.path.exists(jobpath):
    raise ValueError("No such path: %s" % (jobpath))
  taskids = PlotELBO.parse_task_ids(jobpath, args.taskids)

  if args.doPlotData:
    Data = loadData(jobpath)

  if args.savefilename is not None and len(taskids) > 0:
    try:
      args.savefilename % ('1')
    except TypeError:
      raise ValueError("Missing or bad format string in savefilename %s" %  
                        (args.savefilename)
                      )  
  for taskid in taskids:
    taskpath = os.path.join(jobpath, taskid)
    if args.iterid is None:
      prefix = "Best"
    else:
      prefix = "Iter%05d" % (args.iterid)
    hmodel = bnpy.ioutil.ModelReader.load_model(taskpath, prefix)

    figHandle = pylab.figure()
    if args.doPlotData:
      plotData(Data)

    if type(hmodel.obsModel) == bnpy.obsmodel.GaussObsCompSet:
      bnpy.viz.GaussViz.plotGauss2DFromHModel(hmodel)
    elif args.dataName.count('Bars') > 0:
      pylab.close(figHandle)
      Data = loadData(jobpath)
      bnpy.viz.BarsViz.plotBarsFromHModel(hmodel, Data=Data, doShowNow=False)
    else:
      raise NotImplementedError('TODO')
    if args.savefilename is not None:
      pylab.show(block=False)
      pylab.savefig(args.savefilename % (taskid))
  
  if args.savefilename is None:
    pylab.show(block=True)
  
if __name__ == "__main__":
  main()

