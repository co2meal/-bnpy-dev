'''
Unit tests for BirthRefine.py

Verifies that births produce valid models with expected new components.

'''

import numpy as np
import unittest

import bnpy
BirthMove = bnpy.birthmove.BirthMove
BirthProposalError = bnpy.birthmove.BirthProposalError

import UtilForBirthTest as U



######################################################### Bars K6V9
#########################################################
class TestBirthMove(unittest.TestCase):

  def shortDescription(self):
    return None

  def setUp(self):
    U.getBarsData('BarsK6V9')

  def test_run_birth_move__verify_num_comps_and_scale(self):
    Data = U.getBarsData()
    model, SS, LP = U.MakeModelWithOneTopic(Data)  
    mykwargs = dict(**U.kwargs)
    mykwargs['Kfresh'] = 5
    mykwargs['refineNumIters'] = 2
    mykwargs['cleanupDeleteEmpty'] = 0
    mykwargs['cleanupDeleteToImprove'] = 0
    mykwargs['birthRetainExtraMass'] = 0
    newModel, newSS, Info = BirthMove.run_birth_move(
                                          model, SS, Data, **mykwargs)
    
    assert Info['didAddNew']
    assert newModel.allocModel.K == 6
    assert newModel.obsModel.K == 6
    assert newSS.K == 6
    U.verify_suffstats_at_desired_scale(newSS, 
                                          nDoc=Data.nDoc,
                                          word_count=Data.word_count.sum()
                                       )
    U.verify_obsmodel_at_desired_scale(newModel.obsModel,
                                          word_count=2*Data.word_count.sum()
                                      )

  def test_run_birth_move__RetainExtraMass__verify_num_comps_and_scale(self):
    Data = U.getBarsData()
    model, SS, LP = U.MakeModelWithOneTopic(Data)  
    mykwargs = dict(**U.kwargs)
    mykwargs['Kfresh'] = 5
    mykwargs['refineNumIters'] = 2
    mykwargs['cleanupDeleteEmpty'] = 0
    mykwargs['cleanupDeleteToImprove'] = 0
    mykwargs['birthRetainExtraMass'] = 1
    newModel, newSS, Info = BirthMove.run_birth_move(
                                          model, SS, Data, **mykwargs)
    
    assert Info['didAddNew']
    assert newModel.allocModel.K == 6
    assert newModel.obsModel.K == 6
    assert newSS.K == 6
    U.verify_suffstats_at_desired_scale(newSS, 
                                          nDoc=2*Data.nDoc,
                                          word_count=2*Data.word_count.sum()
                                       )
    U.verify_obsmodel_at_desired_scale(newModel.obsModel,
                                          word_count=2*Data.word_count.sum()
                                      )

  def test_run_birth_move__checkELBOImprove__verify_num_comps_and_scale(self):
    Data = U.getBarsData()
    model, SS, LP = U.MakeModelWithOneTopic(Data)  
    mykwargs = dict(**U.kwargs)
    mykwargs['Kfresh'] = 5
    mykwargs['refineNumIters'] = 2
    mykwargs['cleanupDeleteEmpty'] = 1
    mykwargs['cleanupDeleteToImprove'] = 1
    mykwargs['birthRetainExtraMass'] = 0
    mykwargs['birthVerifyELBOIncrease'] = 1
    newModel, newSS, Info = BirthMove.run_birth_move(
                                          model, SS, Data, **mykwargs)
    
    assert newSS.K > SS.K
    assert Info['didAddNew']
    U.verify_suffstats_at_desired_scale(newSS, 
                                        nDoc=Data.nDoc,
                                        word_count=Data.word_count.sum()
                                       )
    U.verify_obsmodel_at_desired_scale(newModel.obsModel,
                                        word_count=2*Data.word_count.sum()
                                       )


  def test_run_birth_move__Ktrue__verify_checkELBOImprove_rejects_simple(self):
    Data = U.getBarsData()
    model, SS, LP = U.MakeModelWithTrueTopics(Data)  
    mykwargs = dict(**U.kwargs)
    mykwargs['Kfresh'] = 5
    mykwargs['refineNumIters'] = 1
    mykwargs['cleanupDeleteEmpty'] = 0
    mykwargs['cleanupDeleteToImprove'] = 0
    mykwargs['birthRetainExtraMass'] = 0
    mykwargs['birthVerifyELBOIncrease'] = 1
    newModel, newSS, Info = BirthMove.run_birth_move(
                                          model, SS, Data, **mykwargs)
    
    assert newSS.K == SS.K
    assert not Info['didAddNew']


  def test_run_birth_move__Ktrue__verify_checkELBOImprove_rejects_refined(self):
    Data = U.getBarsData()
    model, SS, LP = U.MakeModelWithTrueTopics(Data)  
    mykwargs = dict(**U.kwargs)
    mykwargs['Kfresh'] = 5
    mykwargs['refineNumIters'] = 25
    mykwargs['cleanupDeleteEmpty'] = 1
    mykwargs['cleanupDeleteToImprove'] = 1
    mykwargs['birthRetainExtraMass'] = 0
    mykwargs['birthVerifyELBOIncrease'] = 1
    newModel, newSS, Info = BirthMove.run_birth_move(
                                          model, SS, Data, **mykwargs)
    
    assert newSS.K == SS.K
    assert not Info['didAddNew']


  def test_run_birth_move__SS_with_merges(self):
    ''' Verify that when input suff stats have ELBO/Merge fields,
          the birth moves do not alter these original values
          and only insert zeros for relevant new components
    '''
    Data = U.getBarsData()
    model, SS, LP = U.MakeModelWithFiveTopics(Data)

    LP = model.calc_local_params(Data)
    SS = model.get_global_suff_stats(Data, LP, doPrecompEntropy=True,
                                               doPrecompMergeEntropy=True)    
    assert SS.hasMergeTerms()
    assert not np.allclose(SS.getMergeTerm('ElogpZ'), 0)
    mykwargs = dict(**U.kwargs)
    mykwargs['Kfresh'] = 5
    mykwargs['refineNumIters'] = 3
    mykwargs['cleanupDeleteEmpty'] = 1
    mykwargs['cleanupDeleteToImprove'] = 1
    mykwargs['birthRetainExtraMass'] = 0
    mykwargs['birthVerifyELBOIncrease'] = 0
    newModel, newSS, Info = BirthMove.run_birth_move(
                                          model, SS, Data, **mykwargs)
    assert Info['didAddNew']
    assert newSS.hasELBOTerms()
    print  newSS.getELBOTerm('ElogqPiConst')
    assert newSS.hasMergeTerms()
    assert newSS.K == newModel.obsModel.K
    assert newSS._MergeTerms.K == newSS.K
    verify_original_comp_elbo_fields_intact(newSS, SS, SS.K)
    verify_new_comp_elbo_fields_zero(newSS, SS.K)

    verify_original_comp_merge_fields_intact(newSS, SS, SS.K)
    verify_new_comp_merge_fields_zero(newSS, SS.K)


def verify_original_comp_elbo_fields_intact(aSS, bSS, Korig):
  ''' aSS : Korig+Knew comps
      bSS : Korig comps
  '''
  aKeys = aSS._ELBOTerms._FieldDims
  bKeys = bSS._ELBOTerms._FieldDims
  assert len(aKeys) == len(bKeys)
  for key in aKeys:
    print key
    a = aSS.getELBOTerm(key)
    b = bSS.getELBOTerm(key)
    dims = aSS._ELBOTerms._FieldDims[key]
    print ' ', a
    print ' ', b
    if dims is None:
      assert a == b
    else:
      if dims[0] == 'K':
        assert a.shape[0] > b.shape[0]

      if len(dims) == 2:
        if dims == ('K','K'):
          assert np.allclose(a[:Korig, :Korig], b)
        elif dims[0] == 'K':
          assert np.allclose(a[:Korig], b)
      elif len(dims) == 1:
        assert np.allclose(a[:Korig], b)


def verify_original_comp_merge_fields_intact(aSS, bSS, Korig):
  ''' aSS : Korig+Knew comps
      bSS : Korig comps
  '''
  aKeys = aSS._MergeTerms._FieldDims
  bKeys = bSS._MergeTerms._FieldDims
  assert len(aKeys) == len(bKeys)
  for key in aKeys:
    print key
    a = aSS.getMergeTerm(key)
    b = bSS.getMergeTerm(key)
    dims = bSS._MergeTerms._FieldDims[key]
    print ' ', a
    print ' ', b
    if dims is None:
      assert a == b
    else:
      if len(dims) == 2:
        if dims == ('K','K'):
          assert np.allclose(a[:Korig, :Korig], b)
        elif dims[0] == 'K':
          assert np.allclose(a[:Korig], b)
      elif len(dims) == 1:
        assert np.allclose(a[:Korig], b)

def verify_new_comp_merge_fields_zero(aSS, Korig):
  ''' aSS : Korig+Knew comps
      bSS : Korig comps
  '''
  aKeys = aSS._MergeTerms._FieldDims
  for key in aKeys:
    print key
    a = aSS.getMergeTerm(key)
    dims = aSS._MergeTerms._FieldDims[key]
    print ' ', a
    if dims is not None:
      if len(dims) == 2:
        if dims == ('K','K'):
          assert np.allclose(a[Korig:, :], 0)
        elif dims[0] == 'K':
          assert np.allclose(a[Korig:], 0)
      elif len(dims) == 1:
        assert np.allclose(a[Korig:], 0)


def verify_new_comp_elbo_fields_zero(aSS, Korig):
  ''' aSS : Korig+Knew comps
      bSS : Korig comps
  '''
  aKeys = aSS._ELBOTerms._FieldDims
  for key in aKeys:
    print key
    a = aSS.getELBOTerm(key)
    dims = aSS._ELBOTerms._FieldDims[key]
    print ' ', a
    if dims is not None:
      if len(dims) == 2:
        if dims == ('K','K'):
          assert np.allclose(a[Korig:, :], 0)
        elif dims[0] == 'K':
          assert np.allclose(a[Korig:], 0)
      elif len(dims) == 1:
        assert np.allclose(a[Korig:], 0)
