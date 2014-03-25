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

class TestBirthRefine(unittest.TestCase):

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
    xbigModel, xbigSS, freshSS = BirthRefine.expand_then_refine(
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
    xbigModel, xbigSS, xfreshSS = BirthRefine.expand_then_refine(
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
    xbigModel, xbigSS, xfreshSS = BirthRefine.expand_then_refine(
                              freshModel, freshSS, Data,
                              model, SS, **mykwargs)
    U.verify_obsmodel_at_desired_scale(xbigModel.obsModel,
                                          word_count=2*Data.word_count.sum()
                                      )
    U.verify_expanded_obsmodel_bigger_than_original(xbigModel.obsModel,
                                     model.obsModel,
                                    )

  def test_expand_then_refine__delete__verify_suffstats_scale(self):
    Data = U.getBarsData()
    model, SS, LP = U.MakeModelWithOneTopic(Data)  
    freshModel, freshSS = BirthCreate.create_model_with_new_comps(
                                          model, SS, Data, **U.kwargs)
    mykwargs = dict(**U.kwargs)
    mykwargs['refineNumIters'] = 2
    mykwargs['cleanupDeleteEmpty'] = 1
    mykwargs['cleanupDeleteToImprove'] = 1
    xbigModel, xbigSS, xfreshSS = BirthRefine.expand_then_refine(
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
"""
  def test_expand_then_refine__delete_returns_to_true_model(self):
    Data = U.BarsData
    model, SS, LP = U.MakeModelWithTrueTopics(Data)  
    freshModel, freshSS = BirthCreate.create_model_with_new_comps(
                                          model, SS, Data, **U.kwargs)
    mykwargs = dict(**U.kwargs)
    mykwargs['creationroutine'] = 'randomfromprior'
    mykwargs['refineNumIters'] = 1
    mykwargs['cleanupDeleteEmpty'] = 1
    mykwargs['cleanupDeleteToImprove'] = 1
    with self.assertRaises(BirthProposalError):
      newModel, newSS = BirthRefine.expand_then_refine(
                              freshModel, freshSS, Data,
                              model, SS, **mykwargs)

  def test_expand_then_refine__orig_obsmodel_does_not_change(self):
    Data = U.BarsData
    model, SS, LP = U.MakeModelWithTrueTopics(Data)  
    freshModel, freshSS = BirthCreate.create_model_with_new_comps(
                                          model, SS, Data, **U.kwargs)
    mykwargs = dict(**U.kwargs)
    mykwargs['refineNumIters'] = 5
    mykwargs['cleanupDeleteEmpty'] = 1
    mykwargs['cleanupDeleteToImprove'] = 0
    newModel, newSS = BirthRefine.expand_then_refine(
                              freshModel, freshSS, Data,
                              model, SS, **mykwargs)
    assert id(model.obsModel) != id(newModel.obsModel)
    for k in range(SS.K):
      origvec = model.obsModel.comp[k].lamvec
      newvec = newModel.obsModel.comp[k].lamvec
      print k
      print origvec[:10]
      print newvec[:10]
      assert np.allclose(origvec, newvec)

"""
