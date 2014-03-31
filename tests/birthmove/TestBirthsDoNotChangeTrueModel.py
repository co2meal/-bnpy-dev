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
  print 'TargetData: %d docs' % (Data.nDoc)
  newModel, newSS, Info = BirthMove.run_birth_move(
                                              bigmodel, bigSS, Data, **kwargs)
  print 'newModel.K = %d' % (newModel.obsModel.K)
  print Info['msg']
  didPass = True
  msg = ''
  if Info['didAddNew']:
    didPass = False
    msg = 'added new comp'
  if not id(newModel) == id(bigmodel):
    didPass = False
    msg = 'model id changed'
  if not np.allclose(newSS.N, bigSS.N):
    didPass = False
    msg = 'SuffStats contents changed. Should be exactly the same!'
  Info['model'] = newModel
  Info['SS'] = newSS
  return didPass, msg, Info
  

class TestNoChange_BarsK6V9(unittest.TestCase):

  def shortDescription(self):
    return None

  def setUp(self):
    self.dataName = 'BarsK6V9'
    mykwargs = dict(**U.kwargs)
    mykwargs['Kfresh'] = 5
    mykwargs['targetMaxSize'] = 100
    mykwargs['creationroutine'] = 'randexamples'
    mykwargs['refineNumIters'] = 20
    mykwargs['cleanupDeleteEmpty'] = 1
    mykwargs['cleanupDeleteToImprove'] = 1
    mykwargs['birthRetainExtraMass'] = 0
    self.kwargs = mykwargs

  def test_no_change_after_birth(self):
    Data = U.getBarsData(self.dataName)
    model, SS, LP = U.MakeModelWithTrueTopics(Data)  

    for trial in range(10):
      didPass, msg, Info = runBirthAndVerifyNoChange(model, SS, 
                                                      Data, **self.kwargs)
      print msg
      assert didPass


class TestNoChange_BarsK10V900(TestNoChange_BarsK6V9):

  def setUp(self):
    self.dataName = 'BarsK10V900'
    mykwargs = dict(**U.kwargs)
    mykwargs['Kfresh'] = 5
    mykwargs['targetMaxSize'] = 300
    mykwargs['creationroutine'] = 'randexamples'
    mykwargs['refineNumIters'] = 20
    mykwargs['cleanupDeleteEmpty'] = 1
    mykwargs['cleanupDeleteToImprove'] = 1
    mykwargs['birthRetainExtraMass'] = 0
    self.kwargs = mykwargs


class TestNoChange_BarsK10V900_findmissingtopics(
                                    TestNoChange_BarsK6V9):

  def setUp(self):
    self.dataName = 'BarsK10V900'
    mykwargs = dict(**U.kwargs)
    mykwargs['Kfresh'] = 10
    mykwargs['targetMaxSize'] = 300
    mykwargs['creationroutine'] = 'findmissingtopics'
    mykwargs['refineNumIters'] = 20
    mykwargs['cleanupDeleteEmpty'] = 1
    mykwargs['cleanupDeleteToImprove'] = 1
    mykwargs['birthRetainExtraMass'] = 0
    self.kwargs = mykwargs
