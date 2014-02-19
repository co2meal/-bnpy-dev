'''
Unit-tests for full learning for topic models
'''
import TestGenericModel
import bnpy

import unittest

class TestHDPFullHard(TestGenericModel.TestGenericModel):
  __test__ = True

  def setUp(self):
    self.Data = bnpy.data.WordsData.CreateToyDataSimple(nDoc=25, nWordsPerDoc=50, vocab_size=100)
    self.allocModelName = 'HDPFullHard'
    self.obsModelName = 'Mult'  
    self.kwargs = dict(nLap=30, K=5, alpha0=1)
    self.kwargs['lambda'] = 1
    self.kwargs['doMemoizeLocalParams'] = 1
    self.kwargs['doFullPassBeforeMstep'] = 1
    self.kwargs['convergeSigFig'] = 25 # dont converge early!

    self.mustRetainLPAcrossLapsForGuarantees = True