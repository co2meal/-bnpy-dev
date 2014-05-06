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
		print '============================= logPrObs'
		print '------------- SumProduct'
		print logPrObs
		print '------------- BruteForce'
		print logPrObs2

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

class TestQuadTreeUtil_21Nodes(unittest.TestCase):

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

	def test_sumproduct_equals_bruteforce(self):
		print ''
		print 'This might take a few minutes...'
		resp, respPair, logPrObs = QTU.calcRespBySumProduct(
										self.PiInit, self.PiMat, self.logSoftEv)
		resp2, respPair2, logPrObs2 = QTU.calcRespByBruteForce(
										self.PiInit, self.PiMat, self.logSoftEv)
		print '============================= resp'
		print '------------- SumProduct'
		print resp
		print '------------- BruteForce'
		print resp2
		print '============================= logPrObs'
		print '------------- SumProduct'
		print logPrObs
		print '------------- BruteForce'
		print logPrObs2

class TestQuadTreeUtil_NonuniformLik(TestQuadTreeUtil_5Nodes):
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
        logSoftEv = np.log(SoftEv)
        self.PiInit = PiInit
        self.PiMat = PiMat
        self.logSoftEv = logSoftEv
        self.SoftEv = SoftEv