'''
PlotComps.py

Executable for plotting learned parameters for each component

Usage
-------
python -m bnpy.viz.PlotComps dataName aModelName obsModelName algName [opts]

'''
from matplotlib import pylab
import numpy as np
import argparse
import os
import sys
import bnpy
import bnpy.ioutil.BNPYArgParser as BNPYArgParser


def main():
  args = parse_args()
  jobpath, taskids = parse_jobpath_and_taskids(args)

  for taskid in taskids:
    taskpath = os.path.join(jobpath, taskid)
    hmodel = bnpy.ioutil.ModelReader.load_model(taskpath, "Best")
    plotModelInNewFigure(jobpath, hmodel, args)
    if args.savefilename is not None:
      pylab.show(block=False)
      pylab.savefig(args.savefilename % (taskid))
  
  if args.savefilename is None:
    pylab.show(block=True)

        
def plotModelInNewFigure(jobpath, hmodel, args):
  figHandle = pylab.figure()
  if args.doPlotData:
    Data = loadData(jobpath)
    plotData(Data)

  if hmodel.getObsModelName().count('ZMGauss') and hmodel.obsModel.D > 2:
    bnpy.viz.GaussViz.plotCovMatFromHModel(hmodel)
  elif hmodel.getObsModelName().count('Gauss'):
    bnpy.viz.GaussViz.plotGauss2DFromHModel(hmodel)
  elif args.dataName.count('Bars') > 0:
    pylab.close(figHandle)
    Data = loadData(jobpath)
    bnpy.viz.BarsViz.plotBarsFromHModel(hmodel, Data=Data, doShowNow=False)
  else:
    raise NotImplementedError('TODO')

def plotData(Data, nObsPlot=5000):
  ''' Plot data items, at most nObsPlot distinct points (for quick rendering)
  '''
  if type(Data) == bnpy.data.XData:
    PRNG = np.random.RandomState(nObsPlot)
    pIDs = PRNG.permutation(Data.nObs)[:nObsPlot]
    pylab.plot(Data.X[pIDs,0], Data.X[pIDs,1], 'k.')  

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
  
  
def parse_args():
  ''' Parse cmd line arguments
  '''
  parser = argparse.ArgumentParser() 
   
  BNPYArgParser.addRequiredVizArgsToParser(parser)
  BNPYArgParser.addStandardVizArgsToParser(parser)
  parser.add_argument('--doPlotData', action='store_true', default=False,
        help="if present, also plot training data")
  args = parser.parse_args()
  return args

def parse_jobpath_and_taskids(args):
  rootpath = os.path.join(os.environ['BNPYOUTDIR'], args.dataName, 
                              args.allocModelName, args.obsModelName)
  jobpath = os.path.join(rootpath, args.algNames, args.jobnames)
  if not os.path.exists(jobpath):
    raise ValueError("No such path: %s" % (jobpath))
  taskids = BNPYArgParser.parse_task_ids(jobpath, args.taskids)

  # Verify that the intended savefile will work as expected!
  if args.savefilename is not None:
    if args.savefilename.count('%') and len(taskids) > 1:
      try:
        args.savefilename % ('1')
      except TypeError:
        raise ValueError("Missing or bad format string in savefilename %s" %  
                        (args.savefilename)
                      )  
  return jobpath, taskids

  
if __name__ == "__main__":
  main()

