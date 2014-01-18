'''
Unit tests for MergeMove.py for HDPTopicModels

Verification merging works as expected and produces valid models.


'''
import numpy as np
import unittest

import bnpy
from bnpy.learnalg import MergeMove
from scipy.special import digamma
import copy

class TestMergeHDP(unittest.TestCase):
  def shortDescription(self):
    return None

  def setUp(self):
    self.MakeData()
    self.MakeModelWithTrueComps()
    self.MakeModelWithDuplicatedComps()

  ######################################################### Make Data
  #########################################################
  def MakeData(self, K=4, D=2500, nWordsPerDoc=50):
    ''' Simple 4 component data on 6 word vocabulary
        
    '''
    TopicWord = np.zeros((K,6))
    TopicWord[0] = [0.48, 0.48, 0.01, 0.01, 0.01, 0.01]
    TopicWord[1] = [0.01, 0.01, 0.48, 0.48, 0.01, 0.01]
    TopicWord[2] = [0.01, 0.01, 0.01, 0.01, 0.48, 0.48]
    TopicWord[3] = [0.01, 0.33, 0.01, 0.32, 0.01, 0.32]
    docTopicParamVec = 0.1 * np.ones(4)
    Data = bnpy.data.WordsData.genToyData(TopicWordProbs=TopicWord,
                      docTopicParamVec=docTopicParamVec,
                      nDocTotal=D,
                      nWordsPerDoc=nWordsPerDoc, seed=123)
    # "Make up" the right local params for this data
    self.trueResp = np.zeros( (Data.nObs, K))
    docID = 0
    for nn in range(Data.nObs):
      if nn >= Data.doc_range[docID,1]:
        docID += 1
      topicProbs = Data.true_td[:,docID] 
      self.trueResp[nn,:] = topicProbs * TopicWord[:,Data.word_id[nn]]
    self.trueResp /= np.sum(self.trueResp,axis=1)[:,np.newaxis]
    assert np.allclose(1.0,np.sum(self.trueResp,axis=1))
    self.Data = Data

  ######################################################### Make Model
  #########################################################
  def MakeModelWithTrueComps(self):
    aDict = dict(alpha0=1.0, gamma=0.1)
    oDict = {'lambda':0.05}
    self.hmodel = bnpy.HModel.CreateEntireModel('VB', 'HDPModel', 'Mult',
                                             aDict, oDict, self.Data)
    LP = self.getTrueLP()
    SS = self.hmodel.get_global_suff_stats(self.Data, LP)
    self.hmodel.update_global_params(SS)

  def MakeModelWithDuplicatedComps(self):
    aDict = dict(alpha0=1.0, gamma=0.1)
    oDict = {'lambda':0.05}
    self.dupModel = bnpy.HModel.CreateEntireModel('VB', 'HDPModel', 'Mult',
                                             aDict, oDict, self.Data)
    dupLP = self.getDupLP()
    dupSS = self.dupModel.get_global_suff_stats(self.Data, dupLP)
    self.dupModel.update_global_params(dupSS)

  def getTrueLP(self):
    return self.getLPfromResp(self.trueResp)

  def getDupLP(self):
    Data = self.Data
    K = self.trueResp.shape[1]
    dupResp = np.zeros((Data.nObs, 2*K))
    dupResp[:Data.nObs/2,:K] = self.trueResp[:Data.nObs/2]
    dupResp[Data.nObs/2:,K:] = self.trueResp[Data.nObs/2:]
    return self.getLPfromResp(dupResp)

  def getLPfromResp(self, Resp, smoothMass=0.001):
    Data = self.Data
    D = Data.nDoc
    K = Resp.shape[1]
    # DocTopicCount matrix : D x K matrix
    DocTopicC = np.zeros((D, K))
    for dd in range(D):
      start,stop = Data.doc_range[dd,:]
      DocTopicC[dd,:] = np.dot(Data.word_count[start:stop],        
                               Resp[start:stop,:]
                               )
    assert np.allclose(DocTopicC.sum(), Data.word_count.sum())
    # Alpha and ElogPi : D x K+1 matrices
    padCol = smoothMass * np.ones((D,1))
    alph = np.hstack( [DocTopicC + smoothMass, padCol])    
    ElogPi = digamma(alph) - digamma(alph.sum(axis=1))[:,np.newaxis]
    assert ElogPi.shape == (D,K+1)
    return dict(word_variational =Resp, 
              E_logPi=ElogPi, alphaPi=alph,
              DocTopicCount=DocTopicC)    

  def getSuffStatsPrepForMerge(self, hmodel):
    ''' With merge flats ENABLED,
          run Estep, calc suff stats, then do an Mstep
    '''
    LP = hmodel.calc_local_params(self.Data)
    flagDict = dict(doPrecompEntropy=True, doPrecompMergeEntropy=True)
    SS = hmodel.get_global_suff_stats(self.Data, LP, **flagDict)
    hmodel.update_global_params(SS)
    return LP, SS

  ######################################################### Test many moves
  #########################################################
  def test_run_many_merge_moves_trueModel_random(self):
    LP, SS = self.getSuffStatsPrepForMerge(self.hmodel)
    PRNG = np.random.RandomState(0)
    mergeKwArgs = dict(mergename='random')
    a, b, c, MTracker = MergeMove.run_many_merge_moves(self.hmodel, 
                               self.Data, SS,
                               nMergeTrials=100, randstate=PRNG,
                               **mergeKwArgs)
    assert MTracker.nTrial == SS.K * (SS.K-1)/2
    assert MTracker.nSuccess == 0

  def test_run_many_merge_moves_dupModel_random(self):
    LP, SS = self.getSuffStatsPrepForMerge(self.dupModel)
    PRNG = np.random.RandomState(0)
    mergeKwArgs = dict(mergename='random')
    a, b, c, MTracker = MergeMove.run_many_merge_moves(self.dupModel, 
                               self.Data, SS,
                               nMergeTrials=100, randstate=PRNG,
                               **mergeKwArgs)
    assert MTracker.nSuccess == 4
    assert (0,4) in MTracker.acceptedOrigIDs
    assert (1,5) in MTracker.acceptedOrigIDs
    assert (2,6) in MTracker.acceptedOrigIDs
    assert (3,7) in MTracker.acceptedOrigIDs

  def test_run_many_merge_moves_dupModel_marglik(self):
    LP, SS = self.getSuffStatsPrepForMerge(self.dupModel)
    PRNG = np.random.RandomState(456)
    mergeKwArgs = dict(mergename='marglik')
    a, b, c, MTracker = MergeMove.run_many_merge_moves(self.dupModel, 
                               self.Data, SS,
                               nMergeTrials=100, randstate=PRNG,
                               **mergeKwArgs)
    for msg in MTracker.InfoLog:
      print msg
    assert MTracker.nSuccess == 4
    assert MTracker.nTrial == 4
    assert (0,4) in MTracker.acceptedOrigIDs
    assert (1,5) in MTracker.acceptedOrigIDs
    assert (2,6) in MTracker.acceptedOrigIDs
    assert (3,7) in MTracker.acceptedOrigIDs