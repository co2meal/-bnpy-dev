'''
Unit-tests for full learning for full-mean, full-covariance Gaussian models
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
    self.obsModelName = 'Gauss'  
    self.kwargs = dict(nLap=30, K=3, alpha0=1)
    self.kwargs['smatname'] = 'eye'

    self.learnAlgs = ['EM', 'VB', 'moVB', 'soVB']


class TestK2(AbstractEndToEndTest):
  __test__ = True

  def setUp(self):
    self.K = 2
    B = 20
    Mu = np.eye(2)
    Sigma = np.zeros((2,2,2))
    Sigma[0] = np.asarray([[B,0], [0,1./B]])
    Sigma[1] = np.asarray([[1./B,0], [0,B]])    
    Nk = 1000
    X = Util.MakeGaussData(Mu, Sigma, Nk)

    L = np.zeros_like(Sigma)
    for k in xrange(self.K):
      L[k] = np.linalg.inv(Sigma[k])    

    self.Data = bnpy.data.XData(X)
    self.TrueParams = dict(w=0.5*np.ones(self.K), K=self.K, m=Mu, L=L)

    self.ProxFunc = dict(L=Util.CovMatProxFunc,
                         m=Util.VectorProxFunc,
                         w=Util.ProbVectorProxFunc)

    self.allocModelName = 'MixModel'
    self.obsModelName = 'Gauss'  
    self.kwargs = dict(nLap=30, K=self.K, alpha0=1.0)
    self.kwargs['ECovMat'] = 'eye'
    self.kwargs['sF'] = 0.01
    self.learnAlgs = ['EM']

    self.fromScratchArgs = dict(nLap=50, K=self.K, initname='randexamples',
                                alpha0=1.0, sF=0.01, ECovMat='eye')
