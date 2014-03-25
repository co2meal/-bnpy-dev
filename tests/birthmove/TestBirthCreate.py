'''
Unit tests for BirthCreate.py

Verifies that births produce valid models with expected new components.

'''
import numpy as np
import unittest

import bnpy
BirthRefine = bnpy.birthmove.BirthRefine
BirthCreate = bnpy.birthmove.BirthCreate
BirthProposalError = bnpy.birthmove.BirthProposalError

import UtilForBirthTest as U

class TestBirthCreate(unittest.TestCase):
  def shortDescription(self):
    return None

  def setUp(self):
    pass

  def test_create_model_with_new_comps__does_create_Kfresh_topics(self):
    ''' freshModel and freshSS must have exactly Kfresh topics
    '''
    BarsData = U.getBarsData()
    model, SS, LP = U.MakeModelWithOneTopic(BarsData)
    freshModel, freshSS = BirthCreate.create_model_with_new_comps(
                                          model, SS, BarsData, **U.kwargs)
    assert freshModel.obsModel.K == freshSS.K
    assert freshSS.K > SS.K
    assert freshSS.K == U.kwargs['Kfresh']

  def test_create_model_with_new_comps__suffstats_have_target_scale(self):
    ''' freshSS must have scale exactly consistent with target dataset
    '''
    BarsData = U.getBarsData()
    model, SS, LP = U.MakeModelWithOneTopic(BarsData)
    freshModel, freshSS = BirthCreate.create_model_with_new_comps(
                                          model, SS, BarsData, **U.kwargs)
    assert freshSS.nDoc == BarsData.nDoc
    assert np.allclose(freshSS.WordCounts.sum(), BarsData.word_count.sum())      

  def test_create_model_with_new_comps__obsmodel_does_not_have_target_scale(self):
    ''' freshModel will generally have scale inconsistent with target data,
          since topic-word parameters are created "from scratch"
    '''
    BarsData = U.getBarsData()
    model, SS, LP = U.MakeModelWithOneTopic(BarsData)
    freshModel, freshSS = BirthCreate.create_model_with_new_comps(
                                          model, SS, BarsData, **U.kwargs)
    lamsum = 0
    priorsum = freshModel.obsModel.obsPrior.lamvec.sum()
    for k in xrange(freshSS.K):
      lamsum += freshModel.obsModel.comp[k].lamvec.sum() - priorsum
    assert not np.allclose(lamsum, BarsData.word_count.sum() + freshSS.K*priorsum)

  def test_create_model_with_new_comps__cleanupMinSize_raises_error(self):
    BarsData = U.getBarsData()
    model, SS, LP = U.MakeModelWithOneTopic(BarsData)
    mykwargs = dict(**U.kwargs)
    mykwargs['cleanupDeleteEmpty'] = 1
    mykwargs['cleanupMinSize'] = 10000
    with self.assertRaises(BirthProposalError):
      freshModel, freshSS = BirthCreate.create_model_with_new_comps(
                                          model, SS, BarsData, **mykwargs)
