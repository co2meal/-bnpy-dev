'''
Unit tests for MergeMove.py for HDPTopicModels

Verification merging works as expected and produces valid models.


'''
import numpy as np
import unittest

import bnpy
from scipy.special import digamma

class TestMergeMove(unittest.TestCase):
  def shortDescription(self):
    return None

  def setUp(self):
    self.MakeData()
    self.MakeModelWithTrueComps()

  def MakeModelWithTrueComps(self):
    aDict = dict(alpha0=1.0, gamma=0.5)
    oDict = {'lambda':0.05}
    self.hmodel = bnpy.HModel.CreateEntireModel('VB', 'HDPModel', 'Mult',
                                             aDict, oDict, self.Data)
    LP = dict(word_variational = self.trueResp, 
              E_logPi=self.trueElogPi, alphaPi=self.trueAlphPi,
              DocTopicCount=self.trueDocTopicC)
    SS = self.hmodel.get_global_suff_stats(self.Data, LP)
    self.hmodel.update_global_params(SS)
    for iterid in range(3):
      LP = self.hmodel.calc_local_params(self.Data)
      flagDict = dict(doPrecompEntropy=True, doPrecompMergeEntropy=True)
      self.SS = self.hmodel.get_global_suff_stats(self.Data, LP, **flagDict)
      self.hmodel.update_global_params(self.SS)

  def MakeData(self, K=3, D=100):
    ''' Simple 3 component data on 6 word vocabulary
        
    '''
    TopicWord = np.zeros((3,6))
    TopicWord[0] = [0.48, 0.48, 0.01, 0.01, 0.01, 0.01]
    TopicWord[1] = [0.01, 0.01, 0.48, 0.48, 0.01, 0.01]
    TopicWord[2] = [0.01, 0.01, 0.01, 0.01, 0.48, 0.48]
    docTopicParamVec = 0.1 * np.ones(3)
    Data = bnpy.data.WordsData.genToyData(TopicWordProbs=TopicWord,
                      docTopicParamVec=docTopicParamVec,
                      nDocTotal=D,
                      nWordsPerDoc=25, seed=123)
    # "Make up" the right local params for this data
    self.trueResp = np.zeros( (Data.nObs, 3))
    for vv in range(6): 
      mask = Data.word_id == vv
      if vv < 2:
        self.trueResp[mask,0] = 1.0
      elif vv < 4:
        self.trueResp[mask,1] = 1.0
      else:
        self.trueResp[mask,2] = 1.0
    epsPad = 1.001 * np.ones((D,1))
    self.trueAlphPi = np.hstack([Data.true_td.T * 1000, epsPad])
    assert np.shape(self.trueAlphPi) == (D,K+1)
    self.trueElogPi = digamma(self.trueAlphPi) - digamma(self.trueAlphPi.sum(axis=1))[:,np.newaxis]
    assert self.trueElogPi.shape == (D,K+1)
    self.trueDocTopicC = np.zeros((D, 3))
    for dd in range(D):
      start,stop = Data.doc_range[dd,:]
      self.trueDocTopicC[dd,:] = np.dot(Data.word_count[start:stop],        
                                        self.trueResp[start:stop,:]
                                        )
    assert np.allclose(self.trueDocTopicC.sum(), Data.word_count.sum())
    self.Data = Data

  def verify_selected_component_ids(self, kA, kB):
    assert kA < self.SS.K
    assert kB < self.SS.K
    assert kA < kB

  def test_model_matches_ground_truth_as_precheck(self):
    ''' Before learning can proceed, need to ensure the model
          is able to learn ground truth.
    '''
    np.set_printoptions(precision=3,suppress=True)
    for k in range(self.hmodel.obsModel.K):
      logtopicWordHat = self.hmodel.obsModel.comp[k].Elogphi
      topicWordHat = np.exp(logtopicWordHat)
      print topicWordHat
      print self.Data.true_tw[k]
      diffVec = np.abs(topicWordHat - self.Data.true_tw[k])
      print diffVec
      print ' '
      assert np.max(diffVec) < 0.04

