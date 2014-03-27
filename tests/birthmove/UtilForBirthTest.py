''' 
Shared resources for testing BirthMove functionality
'''
import numpy as np

import bnpy

kwargs = dict(creationroutine='randexamples', cleanupMinSize=25,
               expandorder='expandThenRefine', refineNumIters=10,
               Kfresh=10, Kmax=25, randstate=np.random.RandomState(0),
               cleanupDeleteEmpty=1, cleanupDeleteToImprove=0,
               birthRetainExtraMass=0,
               birthVerifyELBOIncrease=0,
             )

BarsName = None
BarsData = None
def getBarsData(name=None):
  global BarsData, BarsName
  if BarsData is None or name != BarsName:
    BarsName = name
    if name == 'BarsK10V900':
      import BarsK10V900
      BarsData = BarsK10V900.get_data(nDocTotal=300, nWordsPerDoc=300)
    else:
      import BarsK6V9
      BarsData = BarsK6V9.get_data(nDocTotal=100)
  return BarsData

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


def verify_expanded_obsmodel_bigger_than_original( xobsModel, obsModel):
  priorvec = obsModel.obsPrior.lamvec
  xpriorvec = xobsModel.obsPrior.lamvec
  assert np.allclose(priorvec, xpriorvec)
  for k in range(obsModel.K):
    lamsum = obsModel.comp[k].lamvec - priorvec
    xlamsum = xobsModel.comp[k].lamvec - xpriorvec
    assert np.all( xlamsum >= lamsum )
