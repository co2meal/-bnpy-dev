'''
'''
from bnpy.distr import WishartDistr
import numpy as np
import copy

class TestWishart(object):
  def setUp(self):
    self.v = 4
    self.invW = np.eye(2)
    self.distr = WishartDistr(v=self.v, invW=self.invW)
    
  def test_dimension(self):
    assert self.distr.D == self.invW.shape[0]
    
  def test_cholinvW(self):
    cholinvW = self.distr.cholinvW()
    assert np.allclose(np.dot(cholinvW, cholinvW.T), self.distr.invW)
  
  def test_expected_covariance_matrix(self):
    CovMat = self.distr.ECovMat()
    MyCovMat = self.invW / (self.v - self.distr.D - 1)
    print MyCovMat, CovMat
    assert np.allclose(MyCovMat, CovMat)
    
  def test_post_update_stochastic(self, rho=0.375):
    distrA = copy.copy(self.distr)
    distrB = WishartDistr(distrA.v, invW=np.eye(distrA.D) )        
    self.distr.post_update_soVB(rho, distrB)
    assert self.distr.v == rho*distrA.v + (1-rho)*distrB.v
    assert np.allclose(self.distr.invW, rho*distrA.invW + (1-rho)*distrB.invW)