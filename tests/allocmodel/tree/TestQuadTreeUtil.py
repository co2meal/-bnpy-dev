import numpy as np
import unittest
import sys

import bnpy.allocmodel.tree.QuadTreeUtil as QTU

class TestQuadTreeUtil_5Nodes(unittest.TestCase):
  def setUp(self):
    PiInit = np.asarray([0.3, 0.7])
    PiMat = np.asarray([[0.6, 0.4],
							[0.2, 0.8]])
    SoftEv = 0.5 * np.ones((5,2))
    logSoftEv = np.log(SoftEv)
    self.PiInit = PiInit
    self.PiMat = PiMat
    self.logSoftEv = logSoftEv
    self.SoftEv = SoftEv

  def test_sumproduct_equals_bruteforce(self):
    print ''
    resp, respPair, logPrObs = QTU.calcRespBySumProduct(
									   self.PiInit, self.PiMat, self.logSoftEv)
    resp2, respPair2, logPrObs2 = QTU.calcRespByBruteForce(
										  self.PiInit, self.PiMat, self.logSoftEv)
    print '============================= resp'
    print '------------- SumProduct'
    print resp
    print '------------- BruteForce'
    print resp2
    assert np.allclose(resp, resp2)

    print '============================= respPair'
    print '------------- SumProduct'
    print respPair[1:3]
    print '------------- BruteForce'
    print respPair2[1:3]
    assert np.allclose(respPair, respPair2)

    print '============================= logPrObs'
    print '------------- SumProduct'
    print logPrObs
    print '------------- BruteForce'
    print logPrObs2
    assert np.allclose(logPrObs, logPrObs2)


class TestQuadTreeUtil_5NodesNonUniformLik(TestQuadTreeUtil_5Nodes):
  def setUp(self):
    PiInit = np.asarray([0.3, 0.7])
    PiMat = np.asarray([[0.6, 0.4],
							[0.2, 0.8]])
    PRNG = np.random.RandomState(0)
    p = PRNG.rand( 5, 1)
    SoftEv = np.hstack([p, 1-p])
    logSoftEv = np.log(SoftEv)
    self.PiInit = PiInit
    self.PiMat = PiMat
    self.logSoftEv = logSoftEv
    self.SoftEv = SoftEv

class TestQuadTreeUtil_4States(TestQuadTreeUtil_5Nodes):
  def setUp(self):
    PiInit = np.asarray([0.3, 0.1, 0.4, 0.2])
    PiMat = np.asarray([[0.1, 0.5, 0.2, 0.2],
							[0.2, 0.1, 0.3, 0.4],
							[0.4, 0.2, 0.1, 0.3],
							[0.2, 0.2, 0.5, 0.1]])
    SoftEv = 0.25 * np.ones((5,4))
    logSoftEv = np.log(SoftEv)
    self.PiInit = PiInit
    self.PiMat = PiMat
    self.logSoftEv = logSoftEv
    self.SoftEv = SoftEv

class TestQuadTreeUtil_4StatesNonUniformLik(TestQuadTreeUtil_5Nodes):
    def setUp(self):
        PiInit = np.asarray([0.3, 0.1, 0.4, 0.2])
        PiMat = np.asarray([[0.1, 0.5, 0.2, 0.2],
                            [0.2, 0.1, 0.3, 0.4],
                            [0.4, 0.2, 0.1, 0.3],
                            [0.2, 0.2, 0.5, 0.1]])
        SoftEv = np.asarray([[0.2, 0.2, 0.35, 0.25],
                             [0.3, 0.4, 0.2, 0.1], 
                             [0.25, 0.1, 0.15, 0.5], 
                             [0.23, 0.37, 0.3, 0.1], 
                             [0.12, 0.28, 0.4, 0.2]])
        assert np.allclose(1.0, np.sum(SoftEv, axis=1))
        logSoftEv = np.log(SoftEv)
        self.PiInit = PiInit
        self.PiMat = PiMat
        self.logSoftEv = logSoftEv
        self.SoftEv = SoftEv


class TestQuadTreeUtil_21Nodes(TestQuadTreeUtil_5Nodes):

	def setUp(self):
		PiInit = np.asarray([0.3, 0.7])
		PiMat = np.asarray([[0.6, 0.4],
							[0.2, 0.8]])
		SoftEv = 0.5 * np.ones((21,2))
		logSoftEv = np.log(SoftEv)
		self.PiInit = PiInit
		self.PiMat = PiMat
		self.logSoftEv = logSoftEv
		self.SoftEv = SoftEv


class TestQuadTreeUtil_21NodesNonUniformLik(TestQuadTreeUtil_5Nodes):
  def setUp(self):
    PiInit = np.asarray([0.3, 0.7])
    PiMat = np.asarray([[0.6, 0.4],
							[0.2, 0.8]])
    PRNG = np.random.RandomState(0)
    SoftEv = PRNG.rand( 21, 2)
    SoftEv /= SoftEv.sum(axis=1)[:,np.newaxis]
    assert np.allclose(1.0, np.sum(SoftEv, axis=1))

    logSoftEv = np.log(SoftEv)
    self.PiInit = PiInit
    self.PiMat = PiMat
    self.logSoftEv = logSoftEv
    self.SoftEv = SoftEv
