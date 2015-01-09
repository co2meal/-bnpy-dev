import unittest
import scipy.io
import numpy as np
np.set_printoptions(precision=2, suppress=1)

import bnpy


respPair = np.asarray([
 [ .01, .02, .03, .14],
 [ .11, .02, .13, .04],
 [ .01, .12, .03, .04],
 [ .11, .02, .13, .04],
 ])

class TestMergeEntropy(unittest.TestCase):

  def setUp(self):
    self.respPair = respPair.copy()
    self.sigma = respPair / respPair.sum(axis=1)[:,np.newaxis]
    self.H_KxK = self.respPair * np.log(self.sigma)
    self.H_total = np.sum(self.respPair * np.log(self.sigma))
    
    # Merge 1and2 into A
    self.ArespPair = respPair[1:, 1:].copy() 
    self.ArespPair[0,0] += respPair[0,0]
    self.ArespPair[:,0] += respPair[1:,0]
    self.ArespPair[0, :] += respPair[0, 1:]
    self.Asigma = self.ArespPair / self.ArespPair.sum(axis=1)[:,np.newaxis]

    # Merge 3and4 into B
    self.BrespPair = respPair[:-1, :-1].copy()
    self.BrespPair[-1, -1] += respPair[-1, -1]
    self.BrespPair[-1, :] += respPair[-1, :-1]
    self.BrespPair[:, -1] += respPair[:-1, -1]
    self.Bsigma = self.BrespPair / self.BrespPair.sum(axis=1)[:,np.newaxis]

  def test_print_A(self):
    print ''
    print self.respPair
    print ''
    print self.ArespPair

  def test_print_B(self):
    print ''
    print self.respPair
    print ''
    print self.BrespPair

  def test_sums_to_one(self):
    print ''
    print self.respPair
    print self.respPair.sum()
    assert np.allclose(1.0, np.sum(self.respPair))
    assert np.allclose(1.0, np.sum(self.sigma, axis=1))

    assert np.allclose(1.0, np.sum(self.ArespPair))
    assert np.allclose(1.0, np.sum(self.Asigma, axis=1))

  def test_entropy(self):
    print ''
    H_orig = -1 * np.sum( self.respPair * np.log(self.sigma))
    H_A = -1 * np.sum( self.ArespPair * np.log(self.Asigma))
    H_B = -1 * np.sum( self.BrespPair * np.log(self.Bsigma))
    print H_orig, ' >= ', H_A
    assert H_orig >= H_A
    assert H_orig >= H_B

    G_orig = -1 * np.sum( self.respPair * np.log(self.respPair))
    G_A = -1 * np.sum( self.ArespPair * np.log(self.ArespPair))
    G_B = -1 * np.sum( self.BrespPair * np.log(self.BrespPair))
    assert G_B >= H_B
    assert G_orig >= H_orig
    assert G_A >= H_A
    print G_A, H_A

