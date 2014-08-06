'''
Unit tests for BirthRefine.py

Verifies that births produce valid models with expected new components.

'''

import numpy as np
import unittest

import bnpy
BirthRefine = bnpy.birthmove.BirthRefine
BirthCreate = bnpy.birthmove.BirthCreate
BirthProposalError = bnpy.birthmove.BirthProposalError

import UtilForBirthTest as U

class TestBirthRefineBarsK6V9(unittest.TestCase):

  def shortDescription(self):
    return None

  def setUp(self):
    pass

  ######################################################### refine_model
  #########################################################
  def test_expand_then_refine__has_correct_size_K(self):
    Data = U.getBarsData()
    model, SS, LP = U.MakeModelWithOneTopic(Data)  
    freshModel, freshSS = BirthCreate.create_model_with_new_comps(
                                          model, SS, Data, **U.kwargs)
    Knew = freshSS.K

    mykwargs = dict(**U.kwargs)
    mykwargs['refineNumIters'] = 1
    mykwargs['cleanupDeleteEmpty'] = 0
    mykwargs['cleanupDeleteToImprove'] = 0
    xbigModel, xbigSS, freshSS, AI, RI = BirthRefine.expand_then_refine(
                              freshModel, freshSS, Data,
                              model, SS, **mykwargs)
    assert xbigSS.K == SS.K + Knew

  def test_expand_then_refine__verify_suffstats_scale(self):
    Data = U.getBarsData()
    model, SS, LP = U.MakeModelWithOneTopic(Data)  
    freshModel, freshSS = BirthCreate.create_model_with_new_comps(
                                          model, SS, Data, **U.kwargs)
    mykwargs = dict(**U.kwargs)
    mykwargs['refineNumIters'] = 5
    mykwargs['cleanupDeleteEmpty'] = 1
    mykwargs['cleanupDeleteToImprove'] = 0
    xbigModel, xbigSS, xfreshSS, AI, RI = BirthRefine.expand_then_refine(
                              freshModel, freshSS, Data,
                              model, SS, **mykwargs)

    U.verify_suffstats_at_desired_scale(xfreshSS, 
                                          nDoc=Data.nDoc,
                                          word_count=Data.word_count.sum()
                                       )
    U.verify_suffstats_at_desired_scale(xbigSS, 
                                          nDoc=2*Data.nDoc,
                                          word_count=2*Data.word_count.sum()
                                       )

  def test_expand_then_refine__verify_obsmodel_scale(self):
    Data = U.getBarsData()
    model, SS, LP = U.MakeModelWithOneTopic(Data)  
    freshModel, freshSS = BirthCreate.create_model_with_new_comps(
                                          model, SS, Data, **U.kwargs)
    mykwargs = dict(**U.kwargs)
    mykwargs['refineNumIters'] = 5
    mykwargs['cleanupDeleteEmpty'] = 1
    mykwargs['cleanupDeleteToImprove'] = 0
    xbigModel, xbigSS, xfreshSS, AI, RI = BirthRefine.expand_then_refine(
                              freshModel, freshSS, Data,
                              model, SS, **mykwargs)
    U.verify_obsmodel_at_desired_scale(xbigModel.obsModel,
                                          word_count=2*Data.word_count.sum()
                                      )
    U.verify_expanded_obsmodel_bigger_than_original(xbigModel.obsModel,
                                     model.obsModel,
                                    )

  def test_expand_then_refine__K1_delete__verify_scale(self):
    Data = U.getBarsData()
    model, SS, LP = U.MakeModelWithOneTopic(Data)  
    freshModel, freshSS = BirthCreate.create_model_with_new_comps(
                                          model, SS, Data, **U.kwargs)
    mykwargs = dict(**U.kwargs)
    mykwargs['refineNumIters'] = 2
    mykwargs['cleanupDeleteEmpty'] = 1
    mykwargs['cleanupDeleteToImprove'] = 1
    xbigModel, xbigSS, xfreshSS, AI, RI = BirthRefine.expand_then_refine(
                              freshModel, freshSS, Data,
                              model, SS, **mykwargs)
    assert xbigSS.K < model.obsModel.K + freshSS.K
    U.verify_suffstats_at_desired_scale(xfreshSS, 
                                          nDoc=Data.nDoc,
                                          word_count=Data.word_count.sum()
                                       )
    U.verify_suffstats_at_desired_scale(xbigSS, 
                                          nDoc=2*Data.nDoc,
                                          word_count=2*Data.word_count.sum()
                                       )
    U.verify_obsmodel_at_desired_scale(xbigModel.obsModel,
                                          word_count=2*Data.word_count.sum()
                                      )

  def test_expand_then_refine__K5_delete__verify_scale(self):
    Data = U.getBarsData()
    model, SS, LP = U.MakeModelWithFiveTopics(Data)  
    freshModel, freshSS = BirthCreate.create_model_with_new_comps(
                                          model, SS, Data, **U.kwargs)
    mykwargs = dict(**U.kwargs)
    mykwargs['refineNumIters'] = 2
    mykwargs['cleanupDeleteEmpty'] = 1
    mykwargs['cleanupDeleteToImprove'] = 1
    xbigModel, xbigSS, xfreshSS, AI, RI = BirthRefine.expand_then_refine(
                              freshModel, freshSS, Data,
                              model, SS, **mykwargs)
    assert xbigSS.K < model.obsModel.K + freshSS.K
    U.verify_suffstats_at_desired_scale(xfreshSS, 
                                          nDoc=Data.nDoc,
                                          word_count=Data.word_count.sum()
                                       )
    U.verify_suffstats_at_desired_scale(xbigSS, 
                                          nDoc=2*Data.nDoc,
                                          word_count=2*Data.word_count.sum()
                                       )
    U.verify_obsmodel_at_desired_scale(xbigModel.obsModel,
                                          word_count=2*Data.word_count.sum()
                                      )

  def test_expand_then_refine__KTrue_delete__raises_true_model(self):
    Data = U.BarsData
    model, SS, LP = U.MakeModelWithTrueTopics(Data)  
    freshModel, freshSS = BirthCreate.create_model_with_new_comps(
                                          model, SS, Data, **U.kwargs)
    mykwargs = dict(**U.kwargs)
    mykwargs['refineNumIters'] = 10
    mykwargs['cleanupDeleteEmpty'] = 1
    mykwargs['cleanupDeleteToImprove'] = 1
    with self.assertRaises(BirthProposalError):
      xbigModel, xbigSS, xfreshSS, AI, RI = BirthRefine.expand_then_refine(
                              freshModel, freshSS, Data,
                              model, SS, **mykwargs)


class TestBirthRefineBarsK10V900(TestBirthRefineBarsK6V9):
  ''' Same as above, but with different dataset!
  '''
  def shortDescription(self):
    return None

  def setUp(self):
    U.getBarsData('BarsK10V900')

  def test_data_size(self):
    assert U.BarsData.TrueParams['K'] == 10
    assert U.BarsData.nDoc > 100
