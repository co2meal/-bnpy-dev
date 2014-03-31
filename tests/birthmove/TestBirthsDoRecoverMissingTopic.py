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
  candidates = bigData.TrueParams['alphaPi'][:, kmissing] > 0.05
  candidates = np.flatnonzero(candidates)
  Data = bigData.get_random_sample(kwargs['targetMaxSize'], 
                                   randstate=kwargs['randstate'],
                                   candidates=candidates,
                                   )
  print 'TargetData: %d docs' % (Data.nDoc)
  newModel, newSS, Info = BirthMove.run_birth_move(
                                              bigmodel, bigSS, Data, **kwargs)
  print 'newModel.K = %d' % (newModel.obsModel.K)
  print Info['msg']
  didPass = True
  msg = ''
  if not Info['didAddNew']:
    didPass = False
    msg = 'failed to add new comp.'
  if id(newModel) == id(bigmodel):
    didPass = False
    msg = 'model id did not change.'
  if newSS.K == bigSS.K:
    didPass = False
    msg = 'SuffStats contents did not change number of comps.'
  if newModel.obsModel.K != newSS.K:
    didPass = False
    msg = 'SuffStats and model should have same num components!'

  Info['model'] = newModel
  Info['SS'] = newSS
  return didPass, msg, Info



class Test_BarsK6V9(unittest.TestCase):

  def shortDescription(self):
    return None

  def setUp(self):
    self.dataName = 'BarsK6V9'
    mykwargs = dict(**U.kwargs)
    mykwargs['Kfresh'] = 5
    mykwargs['targetMaxSize'] = 100
    mykwargs['creationroutine'] = 'randexamples'
    mykwargs['refineNumIters'] = 10
    mykwargs['cleanupDeleteEmpty'] = 1
    mykwargs['cleanupDeleteToImprove'] = 1
    mykwargs['birthVerifyELBOIncrease'] = 1
    mykwargs['birthRetainExtraMass'] = 0
    self.kwargs = mykwargs

  def test_birth_targets_missing_comp_does_add_new(self):
    Data = U.getBarsData(self.dataName)
    for kmissing in range(Data.TrueParams['K']):
      self.verify_birth_targets_missing_comp_does_add_new(Data, kmissing)

  def verify_birth_targets_missing_comp_does_add_new(self, Data, kmissing=0):
    model, SS, LP = U.MakeModelWithTrueTopicsButMissingOne(Data, 
                                                 kmissing=kmissing
                                                 )      
    didPass, msg, Info = runBirthTargetedAtMissingCompAndVerifyChange(
                                                 model, SS, Data, 
                                                 kmissing=kmissing,
                                                 **self.kwargs
                                                 )
    argstring = ' '.join(sys.argv)
    if argstring.count('nocapture') > 0:
      viz_bars_and_wait_for_key_press(Info)
    print msg
    assert didPass

def viz_bars_and_wait_for_key_press(Info):
  from matplotlib import pylab
  from bnpy.viz import BarsViz
  #topics = np.exp(Info['model'].obsModel.getElogphiMatrix())
  #pylab.imshow(topics, aspect=topics.shape[1]/topics.shape[0])
  BarsViz.plotBarsFromHModel( Info['model'], doShowNow=False)
  pylab.show(block=False)
  try: 
    _ = raw_input('Press any key to continue >>')
    pylab.close()
  except KeyboardInterrupt:
    sys.exit(-1)


class Test_BarsK10V900(Test_BarsK6V9):

  def setUp(self):
    self.dataName = 'BarsK10V900'
    mykwargs = dict(**U.kwargs)
    mykwargs['Kfresh'] = 5
    mykwargs['targetMaxSize'] = 100
    mykwargs['creationroutine'] = 'randexamples'
    mykwargs['refineNumIters'] = 10
    mykwargs['cleanupDeleteEmpty'] = 1
    mykwargs['cleanupDeleteToImprove'] = 1
    mykwargs['birthVerifyELBOIncrease'] = 1
    mykwargs['birthRetainExtraMass'] = 0
    self.kwargs = mykwargs


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
