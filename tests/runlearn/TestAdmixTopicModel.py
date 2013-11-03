'''
Unit-tests for full learning for topic models
'''
import TestGenericModel
import bnpy

import unittest

class TestAdmixTopicModel(TestGenericModel.TestGenericModel):

  def setUp(self):
    self.__test__ = True
    self.LearnAlgs = set(['VB'])

    self.Data = bnpy.data.WordsData.makeRandomData(nDoc=25, nWordsPerDoc=50, vocab_size=100)
    self.allocModelName = 'AdmixModel'
    self.obsModelName = 'Mult'  
    self.kwargs = dict(nLap=30, K=5, alpha0=1)
    self.kwargs['lambda'] = 1