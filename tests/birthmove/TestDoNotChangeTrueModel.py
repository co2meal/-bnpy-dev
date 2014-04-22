'''
TestBirthsDoNotChangeTrueModel.py

Goal
-------
We exercise the birthmove module, providing automatic tests verifying that for various toy datasets, the birthmoves (with deletes enabled) will not add any additional "junk" topics when provided an input model matching the "true" generative model.
'''

import numpy as np
import unittest

import bnpy
BirthMove = bnpy.birthmove.BirthMove
BirthProposalError = bnpy.birthmove.BirthProposalError

import UtilForBirthTest as U

def runBirthAndVerifyNoChange(bigmodel, bigSS, bigData, **kwargs):
  Data = bigData.get_random_sample(kwargs['targetMaxSize'], 
                                   randstate=kwargs['randstate'],
                                   )
  newModel, newSS, Info = BirthMove.run_birth_move(
                                              bigmodel, bigSS, Data, **kwargs)

  U.verify_suffstats_at_desired_scale( newSS, nDoc=bigData.nDoc,
                                       word_count=bigData.word_count.sum())
  print 'TargetData: %d docs' % (Data.nDoc)
  print 'newModel.K = %d' % (newModel.obsModel.K)
  print Info['msg']
  np.set_printoptions(precision=2, suppress=True, linewidth=120)

  didPass = True
  msg = ''
  if Info['didAddNew']:
    #U.viz_bars_and_wait_for_key_press(newModel)
    didPass = False
    msg = 'added new comp.'
    freshLP = newModel.calc_local_params(Data)
    freshSS = newModel.get_global_suff_stats(Data, freshLP)
    print freshSS.N
    newM, newELBO = runCoordAscent(newModel, bigData, nIters=10)
    curM, curELBO = runCoordAscent(bigmodel, bigData, nIters=10)
  elif not id(newModel) == id(bigmodel):
    didPass = False
    msg = 'model id changed'
  elif not np.allclose(newSS.N, bigSS.N):
    didPass = False
    msg = 'SuffStats contents changed. Should be exactly the same!'
  Info['model'] = newModel
  Info['SS'] = newSS
  return didPass, msg, Info

def runCoordAscent(model, Data, nIters=10):
  model = model.copy()
  elbo = np.zeros(nIters)
  for iterid in range(nIters):
    LP = model.calc_local_params(Data)
    SS = model.get_global_suff_stats(Data, LP, doPrecompEntropy=True)
    elbo[iterid] = model.calc_evidence(SS=SS)
    model.update_global_params(SS)
  return model, elbo

def createAltNewModel(bigModel, bigData, freshSS):
  newModel = bigModel.copy()
  bigLP = newModel.calc_local_params(bigData)
  bigSS = newModel.get_global_suff_stats(bigData, bigLP)
  Kextra = freshSS.K - bigSS.K
  adjustedInsert = newModel.allocModel.insertEmptyCompsIntoSuffStatBag
  xbigSS, AI, RI = adjustedInsert(bigSS, Kextra)
  
  xbigSS += freshSS
  newModel.update_global_params(xbigSS)
  return newModel, xbigSS

class TestBarsK6V9(unittest.TestCase):

  def shortDescription(self):
    return None

  def setUp(self):
    self.dataName = 'BarsK6V9'
    mykwargs = dict(**U.kwargs)
    mykwargs['randstate'] = np.random.RandomState(123)
    mykwargs['Kfresh'] = 5
    mykwargs['targetMaxSize'] = 100
    mykwargs['targetMinWordsPerDoc'] = 0
    mykwargs['creationRoutine'] = 'randexamples'
    mykwargs['refineNumIters'] = 20
    mykwargs['cleanupDeleteEmpty'] = 1
    mykwargs['cleanupDeleteToImprove'] = 1
    mykwargs['birthVerifyELBOIncrease'] = 1
    mykwargs['birthRetainExtraMass'] = 0
    self.kwargs = mykwargs

  def test_no_change_after_birth(self):
    Data = U.getBarsData(self.dataName)
    model, SS, LP = U.MakeModelWithTrueTopics(Data, aModel='HDPModel2')  

    for trial in range(10):
      didPass, msg, Info = runBirthAndVerifyNoChange(model, SS, 
                                                      Data, **self.kwargs)
      print msg
      assert didPass


class TestBarsK10V900(TestBarsK6V9):

  def setUp(self):
    self.dataName = 'BarsK10V900'
    mykwargs = dict(**U.kwargs)
    mykwargs['randstate'] = np.random.RandomState(123)
    mykwargs['Kfresh'] = 5
    mykwargs['targetMaxSize'] = 100
    mykwargs['targetMinWordsPerDoc'] = 0
    mykwargs['creationRoutine'] = 'randexamples'
    mykwargs['refineNumIters'] = 20
    mykwargs['cleanupDeleteEmpty'] = 1
    mykwargs['cleanupDeleteToImprove'] = 1
    mykwargs['birthVerifyELBOIncrease'] = 1
    mykwargs['birthRetainExtraMass'] = 0
    self.kwargs = mykwargs


class TestBarsK10V900_findmissingtopics(
                                    TestBarsK6V9):

  def setUp(self):
    self.dataName = 'BarsK10V900'
    mykwargs = dict(**U.kwargs)
    mykwargs['randstate'] = np.random.RandomState(123)
    mykwargs['Kfresh'] = 5
    mykwargs['targetMaxSize'] = 100
    mykwargs['targetMinWordsPerDoc'] = 0
    mykwargs['creationRoutine'] = 'randexamples'
    mykwargs['refineNumIters'] = 20
    mykwargs['cleanupDeleteEmpty'] = 1
    mykwargs['cleanupDeleteToImprove'] = 1
    mykwargs['birthVerifyELBOIncrease'] = 1
    mykwargs['birthRetainExtraMass'] = 0
    self.kwargs = mykwargs


class TestBarsK10V900_findmissingtopics_adjusted(
                                    TestBarsK6V9):

  def setUp(self):
    self.dataName = 'BarsK10V900'
    mykwargs = dict(**U.kwargs)
    mykwargs['randstate'] = np.random.RandomState(123)
    mykwargs['Kfresh'] = 5
    mykwargs['targetMaxSize'] = 100
    mykwargs['targetMinWordsPerDoc'] = 0
    mykwargs['creationRoutine'] = 'randexamples'
    mykwargs['refineNumIters'] = 20
    mykwargs['cleanupDeleteEmpty'] = 1
    mykwargs['cleanupDeleteToImprove'] = 1
    mykwargs['birthVerifyELBOIncrease'] = 1
    mykwargs['birthRetainExtraMass'] = 0
    mykwargs['expandAdjustSuffStats'] = 1
    self.kwargs = mykwargs


class TestBarsK10V900_findmissingtopics_adjusted_nodelete(
                                    TestBarsK6V9):

  def setUp(self):
    self.dataName = 'BarsK10V900'
    mykwargs = dict(**U.kwargs)
    mykwargs['randstate'] = np.random.RandomState(123)
    mykwargs['Kfresh'] = 2
    mykwargs['targetMaxSize'] = 100
    mykwargs['targetMinWordsPerDoc'] = 0
    mykwargs['creationRoutine'] = 'randexamples'
    mykwargs['refineNumIters'] = 20
    mykwargs['cleanupDeleteEmpty'] = 0
    mykwargs['cleanupDeleteToImprove'] = 0
    mykwargs['birthVerifyELBOIncrease'] = 1
    mykwargs['birthRetainExtraMass'] = 0
    mykwargs['expandAdjustSuffStats'] = 1
    self.kwargs = mykwargs
