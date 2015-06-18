import numpy as np
import unittest
import sys

import bnpy.allocmodel.tree.BinTreeUtil as BTU

class TestBinTreeUtil_3Nodes(unittest.TestCase):

  def setUp(self):
    PiInit = np.asarray([0.3, 0.7])
    PiMat = np.asarray([[[0.6, 0.4],
                         [0.2, 0.8]],
                        [[0.7, 0.2],
                         [0.5, 0.5]]])
    SoftEv = np.asarray([[0.5, 0.5],
                         [0.5, 0.5],
                         [0.5, 0.5]])
    logSoftEv = np.log(SoftEv)
    self.PiInit = PiInit
    self.PiMat = PiMat
    self.logSoftEv = logSoftEv
    self.SoftEv = SoftEv

  '''
  def test_upward_pass__SingleNode(self):
    print ''
    umsg = BTU.UpwardPass(self.PiInit, self.PiMat, self.SoftEv[0][np.newaxis,:])
    print umsg
    assert not np.any(np.isnan(umsg))

  def test_upward_pass(self):
    print ''
    umsg = BTU.UpwardPass(self.PiInit, self.PiMat, self.SoftEv)
    print umsg
    assert not np.any(np.isnan(umsg))

  def test_downward_pass(self):
    print ''
    umsg = np.ones_like(self.SoftEv)
    dmsg, margPrObs = BTU.DownwardPass(self.PiInit, self.PiMat, self.SoftEv,
                                       umsg)
    print dmsg
    assert not np.any(np.isnan(dmsg))
  '''

  def test_sumproduct_equals_bruteforce(self):
    print ''
    resp, respPair, logPrObs = BTU.SumProductAlg_BinTree(
                     self.PiInit, self.PiMat, self.logSoftEv)
    resp2, respPair2, logPrObs2 = BTU.calcRespByBruteForce(
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


class TestBinTreeUtil_7Nodes(TestBinTreeUtil_3Nodes):

  def setUp(self):
    PiInit = np.asarray([0.3, 0.7])
    PiMat = np.asarray([[[0.6, 0.4],
                         [0.2, 0.8]],
                         [[0.7, 0.3],
                         [0.5, 0.5]]])
    SoftEv = 0.5 * np.ones((7,2))
    logSoftEv = np.log(SoftEv)
    self.PiInit = PiInit
    self.PiMat = PiMat
    self.logSoftEv = logSoftEv
    self.SoftEv = SoftEv

class TestBinTreeUtil_3NodesNonUniformLik(TestBinTreeUtil_3Nodes):
  ''' N = 3, K = 2, non-uniform likelihood
  '''
  def setUp(self):
    PiInit = np.asarray([0.4, 0.6])
    PiMat = np.asarray([[[0.6, 0.4],
                         [0.2, 0.8]],
                         [[0.7, 0.3],
                         [0.5, 0.5]]])
    SoftEv = np.asarray([[0.25, 0.75],
                         [0.6, 0.4], 
                         [0.55, 0.45]])
    logSoftEv = np.log(SoftEv)
    self.PiInit = PiInit
    self.PiMat = PiMat
    self.logSoftEv = logSoftEv
    self.SoftEv = SoftEv

class TestBinTreeUtil_7NodesNonUniformLik(TestBinTreeUtil_3Nodes):
  def setUp(self):
    PiInit = np.asarray([0.3, 0.7])
    PiMat = np.asarray([[[0.6, 0.4],
                         [0.2, 0.8]],
                         [[0.7, 0.3],
                         [0.5, 0.5]]])
    PRNG = np.random.RandomState(0)
    p = PRNG.rand(7,1)
    SoftEv = np.hstack([p, 1-p])
    assert np.allclose(SoftEv.sum(axis=1), 1)

    logSoftEv = np.log(SoftEv)
    self.PiInit = PiInit
    self.PiMat = PiMat
    self.logSoftEv = logSoftEv
    self.SoftEv = SoftEv

class TestBinTreeUtil_4StatesNonUniformLik(TestBinTreeUtil_3Nodes):
  def setUp(self):
    PiInit = np.asarray([0.3, 0.1, 0.4, 0.2])
    PiMat = np.asarray([[[0.1, 0.5, 0.2, 0.2], 
                         [0.2, 0.1, 0.3, 0.4],
                         [0.4, 0.2, 0.1, 0.3], 
                         [0.2, 0.2, 0.5, 0.1]],
                        [[0.1, 0.5, 0.2, 0.2], 
                         [0.2, 0.1, 0.3, 0.4],
                         [0.4, 0.2, 0.1, 0.3], 
                         [0.2, 0.2, 0.5, 0.1]]])

    PRNG = np.random.RandomState(0)
    SoftEv = PRNG.rand(7,4)
    SoftEv /= SoftEv.sum(axis=1)[:,np.newaxis]
    logSoftEv = np.log(SoftEv)
    self.PiInit = PiInit
    self.PiMat = PiMat
    self.logSoftEv = logSoftEv
    self.SoftEv = SoftEv