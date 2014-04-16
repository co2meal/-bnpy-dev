'''
Unit-tests for full learning for zero-mean, full-covariance Gaussian models
'''
import TestGenericModel
import bnpy
import numpy as np
import unittest

class TestMixZMGaussModel(TestGenericModel.TestGenericModel):
  __test__ = True

  def setUp(self):
    PRNG = np.random.RandomState(333)
    X = PRNG.randn(1000, 3)
    self.Data = bnpy.data.XData(X)
    self.allocModelName = 'MixModel'
    self.obsModelName = 'ZMGauss'  
    self.kwargs = dict(nLap=30, K=4, alpha0=0.5)
    self.kwargs['smatname'] = 'eye'
    self.kwargs['sF'] = 0.01

    self.learnAlgs = ['EM', 'VB', 'moVB', 'soVB']
