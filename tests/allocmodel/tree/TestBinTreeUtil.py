import numpy as np
import unittest
import sys

import bnpy.allocmodel.tree.BinTreeUtil as BTU

class TestBinTreeUtil_3Nodes(unittest.TestCase):

  def setUp(self):
    PiInit = np.asarray([0.3, 0.7])
    PiMat = np.asarray([[0.6, 0.4],
                        [0.2, 0.8]])
    SoftEv = np.asarray([[0.5, 0.5],
                         [0.5, 0.5],
                         [0.5, 0.5]])
    logSoftEv = np.log(SoftEv)
    self.PiInit = PiInit
    self.PiMat = PiMat
    self.logSoftEv = logSoftEv
    self.SoftEv = SoftEv


  def test_upward_pass__SingleNode(self):
    print ''
    umsg = BTU.UpwardPass(self.PiInit, self.PiMat, self.SoftEv[0][np.newaxis,:])
    print umsg

  def test_upward_pass(self):
    print ''
    umsg = BTU.UpwardPass(self.PiInit, self.PiMat, self.SoftEv)
    print umsg

  def test_downward_pass(self):
    print ''
    dmsg, margPrObs = BTU.DownwardPass(self.PiInit, self.PiMat, self.SoftEv)
    print dmsg

  def test_sumproduct_equals_bruteforce(self):
    print ''
    resp, respPair, logPrObs = BTU.calcRespBySumProduct(
                                      self.PiInit, self.PiMat, self.logSoftEv)
    resp2, respPair2, logPrObs2 = BTU.calcRespByBruteForce(
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



class TestBinTreeUtil_7Nodes(TestBinTreeUtil_3Nodes):

  def setUp(self):
    PiInit = np.asarray([0.3, 0.7])
    PiMat = np.asarray([[0.6, 0.4],
                        [0.2, 0.8]])
    SoftEv = 0.5 * np.ones((7,2))
    logSoftEv = np.log(SoftEv)
    self.PiInit = PiInit
    self.PiMat = PiMat
    self.logSoftEv = logSoftEv
    self.SoftEv = SoftEv

  def test_get_child_indices(self):
    print ''
    print BTU.get_child_indices(0, 0)

class TestBinTreeUtil_4States(TestBinTreeUtil_3Nodes):
  def setUp(self):
    PiInit = np.asarray([0.3, 0.1, 0.4, 0.2])
    PiMat = np.asarray([[0.1, 0.5, 0.2, 0.2],
                        [0.2, 0.1, 0.3, 0.4],
                        [0.4, 0.2, 0.1, 0.3],
                        [0.2, 0.2, 0.5, 0.1]])
    SoftEv = 0.25 * np.ones((7,4))
    logSoftEv = np.log(SoftEv)
    self.PiInit = PiInit
    self.PiMat = PiMat
    self.logSoftEv = logSoftEv
    self.SoftEv = SoftEv

class TestBinTreeUtil_NonuniformLik(TestBinTreeUtil_3Nodes):
    def setUp(self):
        PiInit = np.asarray([0.3, 0.7])
        PiMat = np.asarray([[0.6, 0.4],
                            [0.2, 0.8]])
        SoftEv = np.asarray([[0.8, 0.2],
                             [0.3, 0.7],
                             [0.42, 0.58]])
        logSoftEv = np.log(SoftEv)
        self.PiInit = PiInit
        self.PiMat = PiMat
        self.logSoftEv = logSoftEv
        self.SoftEv = SoftEv