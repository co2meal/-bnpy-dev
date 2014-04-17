'''
Unit-tests for full learning for zero-mean, full-covariance Gaussian models
'''
import numpy as np
import unittest

import bnpy
from AbstractEndToEndTest import AbstractEndToEndTest
import Util

class TestSimple(AbstractEndToEndTest):
  __test__ = True

  def setUp(self):
    PRNG = np.random.RandomState(333)
    X = PRNG.randn(1000, 3)
    self.Data = bnpy.data.XData(X)
    self.allocModelName = 'MixModel'
    self.obsModelName = 'ZMGauss'  
    self.kwargs = dict(nLap=30, K=4, alpha0=1.0)
    self.kwargs['smatname'] = 'eye'
    self.kwargs['sF'] = 0.01

    self.learnAlgs = ['EM', 'VB', 'moVB', 'soVB']

class TestK2(AbstractEndToEndTest):
  __test__ = True

  def setUp(self):
    self.K = 2
    B = 20
    Sigma = np.zeros((2,2,2))
    Sigma[0] = np.asarray([[B,0], [0,1./B]])
    Sigma[1] = np.asarray([[1./B,0], [0,B]])    
    Nk = 1000
    X = Util.MakeZMGaussData(Sigma, Nk)
    
    self.Data = bnpy.data.XData(X)
    self.TrueParams = dict(w=0.5*np.ones(self.K), Sigma=Sigma, K=self.K)

    # Pass if w is within 0.01 of wtrue    
    self.ProxFunc = dict(Sigma=Util.CovMatProxFunc,
                          w=Util.ProbVectorProxFunc)

    self.allocModelName = 'MixModel'
    self.obsModelName = 'ZMGauss'  
    self.kwargs = dict(nLap=30, K=self.K, alpha0=1.0)
    self.kwargs['ECovMat'] = 'eye'
    self.kwargs['sF'] = 0.01
    self.learnAlgs = ['EM']

    self.fromScratchArgs = dict(nLap=50, K=self.K, initname='randexamples',
                                alpha0=1.0, sF=0.01, ECovMat='eye')


class TestStarCovarK5(AbstractEndToEndTest):
  __test__ = True

  def setUp(self):
    self.K = 5
    import StarCovarK5
    self.Data = StarCovarK5.get_data(nObsTotal=10000)
    
    self.TrueParams = dict(w=StarCovarK5.w,
                           Sigma=StarCovarK5.Sigma,
                           K=5)

    self.ProxFunc = dict(Sigma=Util.CovMatProxFunc,
                          w=Util.ProbVectorProxFunc)

    self.allocModelName = 'MixModel'
    self.obsModelName = 'ZMGauss'  
    self.kwargs = dict(nLap=30, K=self.K, alpha0=1.0)
    self.kwargs['ECovMat'] = 'eye'
    self.kwargs['sF'] = 0.01
    self.learnAlgs = ['EM']

    self.fromScratchArgs = dict(nLap=50, K=self.K, initname='randexamples',
                                alpha0=1.0, sF=0.01, ECovMat='eye')
    self.fromScratchTrials = 10
    self.fromScratchSuccessRate = 0.5