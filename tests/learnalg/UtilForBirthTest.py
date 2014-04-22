''' 
Shared resources for testing BirthMove functionality
'''
import numpy as np

import bnpy

outArgs = dict(traceEvery=1, saveEvery=0, printEvery=1,
               customFuncPath=None, customFuncArgs=None)

algArgs = dict(doFullPassBeforeMstep=0,
            convergeSigFig=6,
            startLap=0,
            doMemoizeLocalParams=1,
            nCoordAscentItersLP=10,
            convThrLP=0.01,
            nHardItersLP=1,
            doDocTopicFracLP=0,
            doOnlySomeDocsLP=1,
            doShowSeriousWarningsOnly=1)

birthArgs = dict(
               Kfresh=10, Kmax=25, 
               creationRoutine='randexamples', 
               cleanupMinSize=25,
               expandAdjustSuffStats=0,
               expandOrder='expandThenRefine',
               refineNumIters=10,
               cleanupDeleteEmpty=0,
               cleanupDeleteToImprove=0,
               birthRetainExtraMass=0,
               birthVerifyELBOIncrease=0,
               fracLapsBirth=0.9,
               birthPerLap=1,
               birthBatchFrac=0.0,
               birthBatchLapLimit=10,
               doVizBirth=0,
               ### TargetPlanner
               targetSelectName='uniform',
               ### TargetDataSampler
               targetMinSize=0,
               targetMaxSize=30,
               targetCompFrac=0.1,
               targetMinWordsPerDoc=0,
               targetMinKLPerDoc=0,
             )

BarsName = None
BarsData = None
def getBarsData(name=None):
  global BarsData, BarsName
  if BarsData is None or name != BarsName:
    BarsName = name
    if name == 'BarsK10V900':
      import BarsK10V900
      BarsData = BarsK10V900.get_data(nDocTotal=500, nWordsPerDoc=300)
      # NOTE: tests for donotchangeorigmodel are very sensitive to nDocTotal
      #  be careful when we change this! 
      #   nDocTotal=300, for example, will cause tests to fail
    else:
      import BarsK6V9
      BarsData = BarsK6V9.get_data(nDocTotal=100)
  return BarsData

def viz_bars_and_wait_for_key_press(Info):
  from matplotlib import pylab
  from bnpy.viz import BarsViz
  BarsViz.plotBarsFromHModel( Info['model'], doShowNow=False)
  pylab.show(block=False)
  try: 
    _ = raw_input('Press any key to continue >>')
    pylab.close()
  except KeyboardInterrupt:
    sys.exit(-1)


def loadData(name, **kwargs):
  datamod = __import__(name, fromlist=[])
  return datamod.get_data(**kwargs)

def MakeModelWithTrueTopics(Data, alpha0=5.0, gamma=0.5):
  ''' Create new model.
  '''
  aDict = dict(alpha0=alpha0, gamma=gamma)
  oDict = {'lambda':0.1}
  hmodel = bnpy.HModel.CreateEntireModel('VB', 'HDPModel', 'Mult', 
                                          aDict, oDict, Data)
  hmodel.init_global_params(Data, initname='trueparams')
  LP = hmodel.calc_local_params(Data)
  SS = hmodel.get_global_suff_stats(Data, LP)
  hmodel.update_global_params(SS)
  LP = hmodel.calc_local_params(Data)
  SS = hmodel.get_global_suff_stats(Data, LP)
  return hmodel, SS, LP

def MakeModelWithTrueTopicsButMissingOne(Data, kmissing=0):
  ''' Create new model.
  '''
  aDict = dict(alpha0=5.0, gamma=0.5)
  oDict = {'lambda':0.1}
  hmodel = bnpy.HModel.CreateEntireModel('VB', 'HDPModel', 'Mult', 
                                          aDict, oDict, Data)
  hmodel.init_global_params(Data, initname='trueparams')
  LP = hmodel.calc_local_params(Data)
  SS = hmodel.get_global_suff_stats(Data, LP)
  # Remove the comp from SS and the model itself
  SS.removeComp(kmissing)
  hmodel.update_global_params(SS)

  # Perform local/summary/global updates so everything is at desired scale
  LP = hmodel.calc_local_params(Data)
  SS = hmodel.get_global_suff_stats(Data, LP)
  hmodel.update_global_params(SS)
  return hmodel, SS, LP

def MakeModelWithFiveTopics(Data):
  ''' Create new model.
  '''
  aDict = dict(alpha0=5.0, gamma=0.5)
  oDict = {'lambda':0.1}
  hmodel = bnpy.HModel.CreateEntireModel('VB', 'HDPModel', 'Mult', 
                                          aDict, oDict, Data)
  hmodel.init_global_params(Data, K=5, initname='randexamples',
                                    seed=0)
  LP = hmodel.calc_local_params(Data)
  SS = hmodel.get_global_suff_stats(Data, LP)
  hmodel.update_global_params(SS)
  return hmodel, SS, LP

def MakeModelWithOneTopic(Data):
  ''' Create new model.
  '''
  aDict = dict(alpha0=5.0, gamma=0.5)
  oDict = {'lambda':0.1}
  hmodel = bnpy.HModel.CreateEntireModel('VB', 'HDPModel', 'Mult', 
                                          aDict, oDict, Data)
  hmodel.init_global_params(Data, K=1, initname='randomfromprior',
                                    seed=0)
  LP = hmodel.calc_local_params(Data)
  SS = hmodel.get_global_suff_stats(Data, LP)
  hmodel.update_global_params(SS)
  return hmodel, SS, LP

