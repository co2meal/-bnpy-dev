'''
Unit tests for LocalStepBagOfWords.py function calcLocalDocParams()

Test Strategy
------------
Consider simple set of 2 topics over 10 vocabulary words
  topic A covers words 0,1,2,3,4,5,6
  topic B covers words       3,4,5,6,7,8,9

We should be able to correctly infer local doc-specific params that
  place a doc with only topic A words into topic A
  place a doc with only topic B words into topic B
  place a doc with both A and B words into topics A and B evenly

'''
import sys
import numpy as np
import unittest

from bnpy.allocmodel.admix import LocalStepBagOfWords 
from bnpy.data import WordsData

PRNG = np.random.RandomState(0)

######### Create topics
topicA = [1., 1., 1., 1., 1., 1., 1., 0., 0., 0.]
topicB = [0., 0., 0., 1., 1., 1., 1., 1., 1., 1.]
topics = np.vstack([topicA, topicB]) + 1e-5
topics /= np.sum(topics,axis=1)[:,np.newaxis]
topicPrior = np.asarray([0.1, 0.1])

######### Create documents
word_id_A = np.arange(0, 7)
word_id_B = np.arange(3, 10)
word_id_C = np.arange(0, 10)
word_ct_A = 50 * np.ones_like(word_id_A)
word_ct_B = 50 * np.ones_like(word_id_B)
word_ct_C = 50 * np.ones_like(word_id_C)
doc_range = np.vstack([[0, 7],[7, 14], [14, 24]])

# Add fourth doc with 75%, 25% ratio
word_id_D = np.arange(0, 10)
word_ct_D = PRNG.multinomial(300, topics[0]) \
            + PRNG.multinomial(100, topics[1])

word_id_E = np.arange(0, 10)
word_ct_E = PRNG.multinomial(100, topics[0]) \
            + PRNG.multinomial(300, topics[1])

wid = np.hstack([word_id_A, word_id_B, word_id_C, word_id_D, word_id_E])
wct = np.hstack([word_ct_A, word_ct_B, word_ct_C, word_ct_D, word_ct_E])
doc_range = np.vstack([doc_range, [24, 34], [34, 44]])

Data = WordsData(word_id=wid, word_count=wct,
                           doc_range=doc_range,
                           vocab_size=10)

class TestLocalStepBagOfWords(unittest.TestCase):

  def shortDescription(self):
    return None

  def setUp(self):
    '''
    '''
    self.LocalStepFunc = LocalStepBagOfWords.calcLocalDocParams_forloopoverdocs
    self.kwargs = dict()

  def test_localstep__with_pure_docs(self):
    Eloglik = np.log(topics).T[Data.word_id]
    LP = dict(E_logsoftev_WordsData=Eloglik)
    LP = self.LocalStepFunc(Data, LP, topicPrior, **self.kwargs)
    assert 'theta' in LP
    # Make sure first document is all about topic A
    theta1 = LP['theta'][0]
    assert theta1[0] > np.sum(word_ct_A)
    assert theta1[1] < topicPrior[1] + 0.5
    assert np.allclose(np.sum(theta1), np.sum(word_ct_A) + np.sum(topicPrior))
    # Make sure second document is all about topic B
    theta2 = LP['theta'][1]
    assert theta2[0] < topicPrior[0] + 0.5
    assert theta2[1] > np.sum(word_ct_B)
    assert np.allclose(np.sum(theta2), np.sum(word_ct_B) + np.sum(topicPrior))

  def test_localstep__with_mixture_docs(self):
    Eloglik = np.log(topics).T[Data.word_id]
    LP = dict(E_logsoftev_WordsData=Eloglik)
    LP = self.LocalStepFunc(Data, LP, topicPrior, **self.kwargs)
    print LP['theta']
    # Make sure third document is mixture of topics A and B
    theta3 = LP['theta'][2]
    assert theta3[0] > np.sum(word_ct_C)/2
    assert theta3[1] > np.sum(word_ct_C)/2
    assert np.allclose(np.sum(theta3), np.sum(word_ct_C) + np.sum(topicPrior))
    # Fourth doc should be nearly 75% A, 25% B
    theta4 = LP['theta'][3]
    assert theta4[0] > 300 - 10
    assert theta4[1] > 100 - 10
    # Fifth doc should be nearly 75% B, 25% A
    theta5 = LP['theta'][4]
    assert theta5[1] > 300 - 10
    assert theta5[0] > 100 - 10
    assert np.allclose(np.sum(theta5), np.sum(word_ct_E) + np.sum(topicPrior))

  def test_localstep__matches_ref_impl(self):
    if not hasattr(self, 'RefFunc'):
      return True
    Eloglik = np.log(topics).T[Data.word_id]
    LP = dict(E_logsoftev_WordsData=Eloglik)
    LP = self.LocalStepFunc(Data, LP, topicPrior, **self.kwargs)

    Eloglik = np.log(topics).T[Data.word_id]
    refLP = dict(E_logsoftev_WordsData=Eloglik)
    refLP = self.RefFunc(Data, refLP, topicPrior)
    assert np.allclose( LP['theta'], refLP['theta'])

class TestLocalStepBoW_nObs2nDoc_fast(TestLocalStepBagOfWords):
  def setUp(self):
    self.RefFunc = LocalStepBagOfWords.calcLocalDocParams_forloopoverdocs
    self.LocalStepFunc = LocalStepBagOfWords.calcLocalDocParams_vectorized
    self.kwargs = dict(do_nObs2nDoc_fast=True)

class TestLocalStepBoW_vectorized(TestLocalStepBagOfWords):
  def setUp(self):
    self.RefFunc = LocalStepBagOfWords.calcLocalDocParams_forloopoverdocs
    self.LocalStepFunc = LocalStepBagOfWords.calcLocalDocParams_vectorized
    self.kwargs = dict(do_nObs2nDoc_fast=False)