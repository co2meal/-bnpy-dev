'''
Unit tests for LearnAlg.py

Verification that the verify_evidence check works as expected
'''
import numpy as np
import unittest

from bnpy.learn import LearnAlg

class TestLearnAlg(unittest.TestCase):
  def shortDescription(self):
    pass

  def setUp(self):
    self.THR=1e-3
    self.learnAlg = LearnAlg('/tmp/', dict(convergeTHR=self.THR))

  def test_verify_evidence_small_pos_vals(self, x=1.0e-5):
    assert not self.learnAlg.verify_evidence( evBound=x+self.THR+0.0001, prevBound=x)
    assert self.learnAlg.verify_evidence( evBound=x + self.THR, prevBound=x)
    assert self.learnAlg.verify_evidence( evBound=x + self.THR/10.0, prevBound=x)

  def test_verify_evidence_big_pos_vals(self, x=1.0e5):
    THR = np.max( x * self.THR, self.THR)
    assert not self.learnAlg.verify_evidence( evBound=x+THR+0.0001, prevBound=x)
    assert self.learnAlg.verify_evidence( evBound=x+THR, prevBound=x)
    assert self.learnAlg.verify_evidence( evBound=x+THR/100.0, prevBound=x)

  def test_verify_evidence_big_neg_vals(self, x=-12345678.9011):
    THR = np.max( np.abs(x) * self.THR, self.THR)
    assert not self.learnAlg.verify_evidence( evBound=x+THR+0.0001, prevBound=x)
    assert self.learnAlg.verify_evidence( evBound=x+THR-1e-9, prevBound=x)
    assert self.learnAlg.verify_evidence( evBound=x+THR/100.0, prevBound=x)

  def test_verify_evidence_small_neg_vals(self, x=-9.18e-5):
    THR = self.THR
    assert not self.learnAlg.verify_evidence( evBound=x+THR+0.0001, prevBound=x)
    assert self.learnAlg.verify_evidence( evBound=x+THR-1e-9, prevBound=x)
    assert self.learnAlg.verify_evidence( evBound=x+THR/100.0, prevBound=x)
