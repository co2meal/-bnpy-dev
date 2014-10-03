''' 
Shared resources for testing HDP functionality
'''
import numpy as np
import bnpy

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

def viz_bars_and_wait_for_key_press(model):
  from matplotlib import pylab
  from bnpy.viz import BarsViz
  BarsViz.plotBarsFromHModel(model, doShowNow=False)
  pylab.show(block=False)
  try: 
    _ = raw_input('Press any key to continue >>')
    pylab.close()
  except KeyboardInterrupt:
    sys.exit(-1)

def loadData(name, **kwargs):
  datamod = __import__(name, fromlist=[])
  return datamod.get_data(**kwargs)

def MakeModelWithTrueTopics(Data, alpha0=5.0, gamma=0.5, 
                                  aModel='HDPModel', nSteps=5):
  ''' Create new model.
  '''
  aDict = dict(alpha0=alpha0, gamma=gamma)
  oDict = {'lambda':0.1}
  hmodel = bnpy.HModel.CreateEntireModel('VB', aModel, 'Mult', 
                                          aDict, oDict, Data)
  hmodel.init_global_params(Data, initname='trueparams')
  for _ in range(nSteps):
    LP = hmodel.calc_local_params(Data)
    SS = hmodel.get_global_suff_stats(Data, LP)
    hmodel.update_global_params(SS)
  return hmodel, SS