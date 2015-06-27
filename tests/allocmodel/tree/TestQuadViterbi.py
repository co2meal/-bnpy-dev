import numpy as np
import unittest
import sys


import bnpy.allocmodel.tree.HMTQuadViterbi as HMTV

class TestQuadViterbi_NonuniformLik21(unittest.TestCase):
	def setUp(self):
		PiInit = np.asarray([0.3, 0.7])
		PiMat = np.asarray([[0.6, 0.4],
							[0.2, 0.8]])
		PRNG = np.random.RandomState(0)
		p = PRNG.rand(21,1)
		SoftEv = np.hstack([p, 1-p])
		logSoftEv = np.log(SoftEv)
		self.PiInit = PiInit
		self.PiMat = PiMat
		self.logSoftEv = logSoftEv
		self.SoftEv = SoftEv

	def test_viterbi_equals_bruteforce(self):
	    print ''
	    encoding = HMTV.ViterbiAlg(self.PiInit, self.PiMat, self.logSoftEv)
	    encoding2 = HMTV.findEncodingByBruteForce(self.PiInit, self.PiMat, self.logSoftEv)
	    print '------------- SumProduct'
	    print encoding
	    print '------------- BruteForce'
	    print encoding2
	    assert np.allclose(encoding, encoding2)