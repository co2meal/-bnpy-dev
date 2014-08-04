''' 
Shared resources for testing BirthMove functionality
'''
import numpy as np

import bnpy

kwargs = dict(randstate=np.random.RandomState(0),
               Kfresh=10, Kmax=25, 
               targetMaxSize=100,
               targetMinWordsPerDoc=0,
               targetMinKLPerDoc=0,
               creationRoutine='randexamples', 
               cleanupMinSize=25,
               expandOrder='expandThenRefine',
               refineNumIters=10,
               cleanupDeleteEmpty=1,
               cleanupDeleteToImprove=0,
               birthRetainExtraMass=0,
               birthVerifyELBOIncrease=0,
             )

def loadData(name, **kwargs):
  datamod = __import__(name, fromlist=[])
  return datamod.get_data(**kwargs)

def MakeModelWithTrueComps(Data):
  ''' Create new model.
  '''
  aDict = dict(alpha0=2.0, truncType='z')
  oDict = {'ECovMat':'eye', 'kappa':1e-4, 'dF':0, 'sF':1.0}
  hmodel = bnpy.HModel.CreateEntireModel('VB', 'DPMixModel', 'Gauss', 
                                          aDict, oDict, Data)
  hmodel.init_global_params(Data, initname='truelabels')
  LP = hmodel.calc_local_params(Data)
  SS = hmodel.get_global_suff_stats(Data, LP)
  hmodel.update_global_params(SS)
  LP = hmodel.calc_local_params(Data)
  SS = hmodel.get_global_suff_stats(Data, LP)
  return hmodel, SS, LP

"""
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
"""