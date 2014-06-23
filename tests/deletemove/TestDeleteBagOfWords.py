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

  def test_construct_LP_with_comps_removed__verify_consistent(self):
    ''' Verifies that newLP exactly represents the provided dataset
          * word_variational should sum to one in each row
          * DocTopicCount should sum to nWordPerDoc in each row
          * theta should have total sum equal to nWordPerDoc + alpha
    '''
    Data = DU.getBarsData(self.dataName)
    nWordPerDoc = Data.to_sparse_docword_matrix().toarray().sum(axis=1)
    model, SS, LP = DU.MakeModelWithTrueTopics(Data)
    LP = model.calc_local_params(Data)
    
    newLP = DMove.construct_LP_with_comps_removed(Data, model, 
                                                  compIDs=0, LP=None,
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
    

  def test_propose_model_and_SS_with_comps_removed__viaLP__verify_scale(self):
    Data = DU.getBarsData(self.dataName)
    nWordPerDoc = Data.to_sparse_docword_matrix().toarray().sum(axis=1)
    wc = Data.word_count.sum()

    model, SS, LP = DU.MakeModelWithTrueTopics(Data)
    LP = model.calc_local_params(Data)
    

    for betamethod in ['current', 'prior']:

      newLP = DMove.construct_LP_with_comps_removed(Data, model, 
                                                  compIDs=0, LP=None,
                                                  betamethod=betamethod)
      newModel, newSS = DMove.propose_model_and_SS_with_comps_removed__viaLP(
                                            Data, model, compIDs=0,  
                                                      newLP=newLP)
      DU.verify_obsmodel_at_desired_scale(newModel.obsModel,
                                              word_count=wc)
      DU.verify_suffstats_at_desired_scale(newSS, nDoc=Data.nDoc, word_count=wc)
      assert newModel.obsModel.K == SS.K - 1
      assert newModel.allocModel.K == SS.K - 1
      assert newSS.K == SS.K - 1



  def test_many_comps_removed__viaLP__verify_scale(self):
    Data = DU.getBarsData(self.dataName)
    nWordPerDoc = Data.to_sparse_docword_matrix().toarray().sum(axis=1)
    wc = Data.word_count.sum()

    model, SS, LP = DU.MakeModelWithTrueTopics(Data)
    LP = model.calc_local_params(Data)
    
    cIDs = [1, 3, 5]
    for method in ['current', 'prior']:

      newLP = DMove.construct_LP_with_comps_removed(Data, model, 
                                                  compIDs=cIDs, LP=None,
                                                  betamethod='current')
      newModel, newSS = DMove.propose_model_and_SS_with_comps_removed__viaLP(
                                            Data, model, compIDs=cIDs,  
                                                      newLP=newLP)
      DU.verify_obsmodel_at_desired_scale(newModel.obsModel,
                                              word_count=wc)
      DU.verify_suffstats_at_desired_scale(newSS, nDoc=Data.nDoc, word_count=wc)
      assert newModel.obsModel.K == SS.K - len(cIDs)
      assert newModel.allocModel.K == SS.K - len(cIDs)
      assert newSS.K == SS.K - len(cIDs)



  def test_run_delete_move__verify_scale(self):
    Data = DU.getBarsData(self.dataName)
    nWordPerDoc = Data.to_sparse_docword_matrix().toarray().sum(axis=1)
    wc = Data.word_count.sum()

    model, SS, LP = DU.MakeModelWithTrueTopics(Data)
    LP = model.calc_local_params(Data)
    SS = model.get_global_suff_stats(Data, LP, doPrecompEntropy=True)

    cIDs = [1, 3, 5]
    for betamethod in ['current', 'prior']:

      newModel, newSS, Info = DMove.run_delete_move(Data, model, SS, LP,
                                          compIDs=cIDs, 
                                          betamethod=betamethod)
      DU.verify_obsmodel_at_desired_scale(newModel.obsModel,
                                              word_count=wc)
      DU.verify_suffstats_at_desired_scale(newSS, nDoc=Data.nDoc, word_count=wc)
      assert Info['didRemoveComps'] == 0