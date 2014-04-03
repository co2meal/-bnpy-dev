'''
Unit tests for BirthCreate.py

Verifies that births produce valid models with expected new components.

'''
import numpy as np
from scipy.special import digamma
import unittest

import bnpy
DMove = bnpy.deletemove.DeleteMoveBagOfWords

import UtilForDeleteTest as DU

class TestBarsK6V9(unittest.TestCase):
  def shortDescription(self):
    return None

  def setUp(self):
    self.dataName = 'BarsK6V9'
    mykwargs = dict(**DU.kwargs)
    self.kwargs = mykwargs

  def test_construct_LP_with_comp_removed__verify_consistent(self):
    ''' Verifies that newLP exactly represents the provided dataset
          * word_variational should sum to one in each row
          * DocTopicCount should sum to nWordPerDoc in each row
          * theta should have total sum equal to nWordPerDoc + alpha
    '''
    Data = DU.getBarsData(self.dataName)
    nWordPerDoc = Data.to_sparse_docword_matrix().toarray().sum(axis=1)
    model, SS, LP = DU.MakeModelWithTrueTopics(Data)
    LP = model.calc_local_params(Data)
    
    newLP = DMove.construct_LP_with_comp_removed(Data, model, 
                                                  compID=0, LP=None,
                                                  betamethod='current')
    assert np.allclose( newLP['word_variational'].sum(axis=1), 1.0)

    assert np.allclose( newLP['DocTopicCount'].sum(axis=1), nWordPerDoc)

    print newLP['theta_u']+ newLP['theta'].sum(axis=1)[:10]
    print nWordPerDoc[:10] + model.allocModel.gamma
    assert np.allclose( newLP['theta_u'] + newLP['theta'].sum(axis=1),
                        nWordPerDoc + model.allocModel.gamma)

    assert np.allclose( newLP['digammasumTheta'],
                        digamma(nWordPerDoc + model.allocModel.gamma))

    assert 'E_logPi_u' in newLP

  def test_propose_modelAndSS_with_comp_removed__viaLP__verify_scale(self):
    Data = DU.getBarsData(self.dataName)
    nWordPerDoc = Data.to_sparse_docword_matrix().toarray().sum(axis=1)
    wc = Data.word_count.sum()

    model, SS, LP = DU.MakeModelWithTrueTopics(Data)
    LP = model.calc_local_params(Data)
    
    newLP = DMove.construct_LP_with_comp_removed(Data, model, 
                                                  compID=0, LP=None,
                                                  betamethod='current')
    newModel, newSS = DMove.propose_modelAndSS_with_comp_removed__viaLP(
                                            Data, model, compID=0,  
                                                      newLP=newLP)
    DU.verify_obsmodel_at_desired_scale(newModel.obsModel,
                                              word_count=wc)
    DU.verify_suffstats_at_desired_scale(newSS, nDoc=Data.nDoc, word_count=wc)

"""
  def test_create_model_with_new_comps__does_create_Kfresh_topics(self):
    ''' freshModel and freshSS must have exactly Kfresh topics
    '''
    BarsData = U.getBarsData(self.dataName)
    model, SS, LP = U.MakeModelWithOneTopic(BarsData)
    freshModel, freshSS = BirthCreate.create_model_with_new_comps(
                                          model, SS, BarsData, **self.kwargs)
    assert freshModel.obsModel.K == freshSS.K
    assert freshSS.K > SS.K
    assert freshSS.K == U.kwargs['Kfresh']

  def test_create_model_with_new_comps__verify_scale(self):
    ''' freshSS must have scale exactly consistent with target dataset
    '''
    BarsData = U.getBarsData(self.dataName)
    model, SS, LP = U.MakeModelWithOneTopic(BarsData)
    freshModel, freshSS = BirthCreate.create_model_with_new_comps(
                                          model, SS, BarsData, **self.kwargs)
    assert freshSS.nDoc == BarsData.nDoc
    assert np.allclose(freshSS.WordCounts.sum(), BarsData.word_count.sum())      
"""
