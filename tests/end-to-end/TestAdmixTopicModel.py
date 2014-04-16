'''
Unit-tests for full learning for topic models
'''
import TestGenericModel
import bnpy

import unittest

class TestAdmixTopicModel(TestGenericModel.TestGenericModel):
  __test__ = True

  def setUp(self):
    self.Data = bnpy.data.WordsData.CreateToyDataSimple(nDoc=25, nWordsPerDoc=50, vocab_size=100)
    self.allocModelName = 'AdmixModel'
    self.obsModelName = 'Mult'  
    self.kwargs = dict(nLap=30, K=5, alpha0=1)
    self.kwargs['lambda'] = 1
    self.kwargs['doMemoizeLocalParams'] = 1

    self.mustRetainLPAcrossLapsForGuarantees = True
    self.learnAlgs = ['VB', 'moVB', 'soVB']
