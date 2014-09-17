''' 
Shared resources for testing DeleteMove functionality
'''
import numpy as np

import bnpy

kwargs = dict()

########################################################### Load data
########################################################### 
BarsName = None
BarsData = None
def getBarsData(name=None):
  global BarsData, BarsName
  if BarsData is None or name != BarsName:
    BarsName = name
    if name == 'BarsK10V900':
      import BarsK10V900
      BarsData = BarsK10V900.get_data(nDocTotal=500, nWordsPerDoc=300)
    elif name == 'BarsK50V2500':
      import BarsK50V2500
      BarsData = BarsK50V2500.get_data(nDocTotal=800, nWordsPerDoc=300)
    else:
      import BarsK6V9
      BarsData = BarsK6V9.get_data(nDocTotal=100)
  return BarsData

def loadData(name, **kwargs):
  datamod = __import__(name, fromlist=[])
  return datamod.get_data(**kwargs)


########################################################### Make models
########################################################### 

def MakeModelWithTrueTopics(Data):
  ''' Create new model.
  '''
  aDict = dict(alpha0=5.0, gamma=0.5)
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

########################################################### Scale verification
########################################################### 

def verify_suffstats_at_desired_scale( SS, nDoc=0, word_count=0):
  if hasattr(SS, 'nDoc'):
    assert SS.nDoc == nDoc
    assert np.allclose(SS.WordCounts.sum(), word_count)
    assert np.allclose(SS.N.sum(), word_count)

def verify_obsmodel_at_desired_scale( obsModel, word_count=0):
  priorsum = obsModel.obsPrior.lamvec.sum()
  lamsum = 0
  for k in range(obsModel.K):
    lamsum += obsModel.comp[k].lamvec.sum() - priorsum
  print lamsum, word_count
  assert np.allclose(lamsum, word_count)

########################################################### Visual debugging
########################################################### 
def viz_bars_and_wait_for_key_press(model):
  from matplotlib import pylab
  from bnpy.viz import BarsViz
  BarsViz.plotBarsFromHModel( model, doShowNow=False)
  pylab.show(block=False)
  try: 
    _ = raw_input('Press any key to continue >>')
    pylab.close()
  except KeyboardInterrupt:
    sys.exit(-1)
