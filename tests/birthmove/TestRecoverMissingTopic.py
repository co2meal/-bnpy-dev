'''
TestBirthsDoRecoverMissingTopic.py

Goal
-------
We exercise the birthmove module, verifying that for various toy datasets, when we provide an input model that is *almost* the true generative model, but just missing one component, that the births can recover this missing component.
'''

import numpy as np
import sys
import unittest

import bnpy
BirthMove = bnpy.birthmove.BirthMove
BirthProposalError = bnpy.birthmove.BirthProposalError

import UtilForBirthTest as U

def runBirthTargetedAtMissingCompAndVerifyChange(bigmodel, bigSS, 
                                                bigData, kmissing=0, **kwargs):
  candidates = bigData.TrueParams['alphaPi'][:, kmissing] > 0.5
  candidates = np.flatnonzero(candidates)
  targetData = bigData.get_random_sample(kwargs['targetMaxSize'], 
                                   randstate=kwargs['randstate'],
                                   candidates=candidates,
                                   )
  bnpy.viz.BarsViz.plotExampleBarsDocs(targetData, doShowNow=False)

  print 'TargetData: %d docs' % (targetData.nDoc)
  newModel, newSS, Info = BirthMove.run_birth_move(bigmodel, bigSS, 
                                                   targetData, **kwargs)
  print 'newModel.K = %d' % (newModel.obsModel.K)
  didPass = True
  msg = Info['msg']
  if not Info['didAddNew']:
    didPass = False
    #msg = 'failed to add new comp.'
  elif id(newModel) == id(bigmodel):
    didPass = False
  elif newSS.K == bigSS.K:
    didPass = False
    #msg = 'SuffStats contents did not change number of comps.'
  elif newModel.obsModel.K != newSS.K:
    didPass = False
    #msg = 'SuffStats and model should have same num components!'

  Info['model'] = newModel
  Info['SS'] = newSS
  return didPass, msg, Info



class Test_BarsK6V9(unittest.TestCase):

  def shortDescription(self):
    return None

  def setUp(self):
    self.dataName = 'BarsK6V9'
    mykwargs = dict(**U.kwargs)
    mykwargs['randstate'] = np.random.RandomState(3692468)
    mykwargs['Kfresh'] = 5
    mykwargs['targetMaxSize'] = 100
    mykwargs['creationroutine'] = 'randexamples'
    mykwargs['refineNumIters'] = 10
    mykwargs['cleanupDeleteEmpty'] = 1
    mykwargs['cleanupDeleteToImprove'] = 1
    mykwargs['birthVerifyELBOIncrease'] = 1
    mykwargs['birthRetainExtraMass'] = 0
    mykwargs['expandAdjustSuffStats'] = 1
    mykwargs['doVizBirth'] = 1
    self.kwargs = mykwargs

  def test_birth_targets_missing_comp_does_add_new(self):
    print ''
    Data = U.getBarsData(self.dataName)
    for kmissing in range(Data.TrueParams['K']):
      for _ in range(3):
        self.verify_birth_targets_missing_comp_does_add_new(Data, kmissing)

  def verify_birth_targets_missing_comp_does_add_new(self, Data, kmissing=0):
    model, SS = U.MakeModelWithTrueTopicsButMissingOne(Data, 
                                                 aModel='HDPModel2',
                                                 kmissing=kmissing,
                                                 )
    didPass, msg, Info = runBirthTargetedAtMissingCompAndVerifyChange(
                                                 model, SS, Data, 
                                                 kmissing=kmissing,
                                                 **self.kwargs
                                                 )
    argstring = ' '.join(sys.argv)
    #if argstring.count('nocapture') > 0:
    #  print Info['SS'].sumLogPiActive
    #  U.viz_bars_and_wait_for_key_press(Info['model'])
    print msg
    #assert didPass

class Test_BarsK10V900(Test_BarsK6V9):

  def setUp(self):
    self.dataName = 'BarsK10V900'
    mykwargs = dict(**U.kwargs)
    mykwargs['randstate'] = np.random.RandomState(2468)
    mykwargs['Kfresh'] = 5
    mykwargs['targetMaxSize'] = 100
    mykwargs['creationroutine'] = 'randexamples'
    mykwargs['refineNumIters'] = 10
    mykwargs['cleanupDeleteEmpty'] = 1
    mykwargs['cleanupDeleteToImprove'] = 1
    mykwargs['birthVerifyELBOIncrease'] = 1
    mykwargs['birthRetainExtraMass'] = 0
    mykwargs['expandAdjustSuffStats'] = 1
    mykwargs['doVizBirth'] = 1
    self.kwargs = mykwargs


class Test_BarsK10V900_findmissing(Test_BarsK6V9):

  def setUp(self):
    self.dataName = 'BarsK10V900'
    mykwargs = dict(**U.kwargs)
    mykwargs['randstate'] = np.random.RandomState(2468)
    mykwargs['Kfresh'] = 5
    mykwargs['targetMaxSize'] = 100
    mykwargs['creationroutine'] = 'findmissingtopics'
    mykwargs['refineNumIters'] = 10
    mykwargs['cleanupDeleteEmpty'] = 1
    mykwargs['cleanupDeleteToImprove'] = 1
    mykwargs['birthVerifyELBOIncrease'] = 1
    mykwargs['birthRetainExtraMass'] = 0
    mykwargs['expandAdjustSuffStats'] = 1
    mykwargs['doVizBirth'] = 1
    self.kwargs = mykwargs


"""
class Test_BarsK10V900_findmissingtopics(Test_BarsK6V9):

  def setUp(self):
    self.dataName = 'BarsK10V900'
    mykwargs = dict(**U.kwargs)
    mykwargs['Kfresh'] = 5
    mykwargs['targetMaxSize'] = 100
    mykwargs['creationroutine'] = 'findmissingtopics'
    mykwargs['refineNumIters'] = 10
    mykwargs['cleanupDeleteEmpty'] = 1
    mykwargs['cleanupDeleteToImprove'] = 1
    mykwargs['birthVerifyELBOIncrease'] = 1
    mykwargs['birthRetainExtraMass'] = 0
    self.kwargs = mykwargs



class Test_BarsK10V900_findmissingtopics_adjusted(Test_BarsK6V9):

  def setUp(self):
    self.dataName = 'BarsK10V900'
    mykwargs = dict(**U.kwargs)
    mykwargs['Kfresh'] = 5
    mykwargs['targetMaxSize'] = 100
    mykwargs['creationroutine'] = 'findmissingtopics'
    mykwargs['expandAdjustSuffStats'] = 1
    mykwargs['refineNumIters'] = 10
    mykwargs['cleanupDeleteEmpty'] = 1
    mykwargs['cleanupDeleteToImprove'] = 1
    mykwargs['birthVerifyELBOIncrease'] = 1
    mykwargs['birthRetainExtraMass'] = 0
    self.kwargs = mykwargs
"""
