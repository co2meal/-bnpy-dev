'''
Unit-tests for full learning for full-mean, full-covariance Gaussian models
'''
import TestGenericModel
import bnpy
import numpy as np
import unittest

class TestMixGaussModel(TestGenericModel.TestGenericModel):

  def setUp(self):
    self.__test__ = True

    PRNG = np.random.RandomState(333)
    X = PRNG.randn(1000, 3)
    self.Data = bnpy.data.XData(X)
    self.allocModelName = 'MixModel'
    self.obsModelName = 'Gauss'  
    self.kwargs = dict(nLap=30, K=3, alpha0=1)
    self.kwargs['smatname'] = 'eye'