'''
TestBirthsDoKeepAddingWhenStartedAtK1.py

Goal
-------
We exercise the birthmove module, verifying that births add useful components when started from a suboptimal model for various real-world datasets.

Specifically, we provide an input model with just K=1 component, and verify that if we repeatedly sample data at random and runa  birth move, that none of these moves are rejected.
'''

import numpy as np
import sys
import os
import unittest

import bnpy
TargetDataSampler = bnpy.birthmove.TargetDataSampler
BirthMove = bnpy.birthmove.BirthMove
BirthProposalError = bnpy.birthmove.BirthProposalError

import UtilForBirthTest as U

def runBirthTargetedAtRandomAndVerifyChange(bigmodel, bigSS, 
                                                bigData, **kwargs):

  U.verify_suffstats_at_desired_scale( bigSS, nDoc=bigData.nDoc,
                                       word_count=bigData.word_count.sum())
  U.verify_obsmodel_at_desired_scale( bigmodel.obsModel, 
                                       word_count=bigData.word_count.sum())

  Data = TargetDataSampler.sample_target_data(bigData, **kwargs)
  assert Data.nDoc <= kwargs['targetMaxSize']
  newModel, newSS, Info = BirthMove.run_birth_move(
                                              bigmodel, bigSS, Data, **kwargs)
  print 'TargetData: %d docs' % (Data.nDoc)
  print 'newModel K = %d' % (newModel.obsModel.K)
  print Info['msg']
  didPass = True
  msg = ''
  if not Info['didAddNew']:
    didPass = False
    msg = 'failed to add new comp.'
  elif id(newModel) == id(bigmodel):
    didPass = False
    msg = 'model id did not change.'
  elif newSS.K == bigSS.K:
    didPass = False
    msg = 'SuffStats contents did not change number of comps.'
  elif newModel.obsModel.K != newSS.K:
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
    mykwargs['randstate'] = np.random.RandomState(123)
    mykwargs['Kfresh'] = 5
    mykwargs['targetMinWordsPerDoc'] = 0
    mykwargs['targetMaxSize'] = 100
    mykwargs['creationroutine'] = 'randexamples'
    mykwargs['refineNumIters'] = 10
    mykwargs['cleanupDeleteEmpty'] = 1
    mykwargs['cleanupDeleteToImprove'] = 1
    mykwargs['birthVerifyELBOIncrease'] = 1
    mykwargs['birthRetainExtraMass'] = 0
    mykwargs['expandAdjustSuffStats'] = 1

    self.kwargs = mykwargs
    self.Kexpected = np.minimum(6, mykwargs['Kmax'])

  def test_birth_does_add_new(self):
    Data = U.loadData(self.dataName)
    model, SS, LP = U.MakeModelWithOneTopic(Data)

    while SS.K < self.Kexpected:
      if SS.K > 1:
        LP = model.calc_local_params(Data)
        SS = model.get_global_suff_stats(Data, LP)
        model.update_global_params(SS)

      didPass, msg, Info = runBirthTargetedAtRandomAndVerifyChange(
                                                 model, SS, Data, 
                                                 **self.kwargs
                                                 )
      print msg
      assert didPass
      model = Info['model']
      SS = Info['SS']
      


class Test_NIPS(Test_BarsK6V9):

  def setUp(self):
    self.dataName = 'NIPSCorpus'
    os.environ['BNPYDATADIR'] = '/data/NIPS/'
    if not os.path.exists(os.environ['BNPYDATADIR']):
      os.environ['BNPYDATADIR'] = '/data/liv/liv-x/topic_models/data/nips/'
    sys.path.append(os.environ['BNPYDATADIR'])

    mykwargs = dict(**U.kwargs)
    mykwargs['Kfresh'] = 5
    mykwargs['targetMinWordsPerDoc'] = 100
    mykwargs['targetMaxSize'] = 100
    mykwargs['creationroutine'] = 'findmissingtopics'
    mykwargs['expandAdjustSuffStats'] = 1
    mykwargs['refineNumIters'] = 10
    mykwargs['cleanupDeleteEmpty'] = 1
    mykwargs['cleanupDeleteToImprove'] = 1
    mykwargs['birthVerifyELBOIncrease'] = 1
    mykwargs['birthRetainExtraMass'] = 0
    self.kwargs = mykwargs

    self.Kexpected = np.minimum(20, mykwargs['Kmax'])

class Test_huffpost(Test_BarsK6V9):
  ''' 
      Notes
      --------
      For this dataset, it is very tricky to add useful topics
        via the 'findmissingtopics' creationroutine.
  '''

  def setUp(self):
    self.dataName = 'huffpost'
    os.environ['BNPYDATADIR'] = '/data/huffpost/'
    sys.path.append(os.environ['BNPYDATADIR'])

    mykwargs = dict(**U.kwargs)
    mykwargs['Kfresh'] = 10
    mykwargs['targetMinWordsPerDoc'] = 100
    mykwargs['targetMaxSize'] = 300
    mykwargs['creationroutine'] = 'findmissingtopics'
    mykwargs['refineNumIters'] = 50
    mykwargs['cleanupDeleteEmpty'] = 1
    mykwargs['cleanupDeleteToImprove'] = 1
    mykwargs['birthVerifyELBOIncrease'] = 1
    mykwargs['birthRetainExtraMass'] = 0
    self.kwargs = mykwargs

    self.Kexpected = np.minimum(20, mykwargs['Kmax'])
