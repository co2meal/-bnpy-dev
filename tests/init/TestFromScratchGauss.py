'''
Unit tests for FromScratchGauss.py
'''
import unittest
import numpy as np
from bnpy.data import XData
from bnpy import HModel

class TestFromScratchGauss(unittest.TestCase):
  def shortDescription(self):
    return None

  def setUp(self):
    PRNG = np.random.RandomState(867)
    X = PRNG.randn(100,2)
    self.Data = XData(X=X)

    aPDict = dict(alpha0=1.0)
    oPDict = dict(min_covar=1e-9)
    self.hmodel = HModel.InitFromData('EM','MixModel','ZMGauss', aPDict, oPDict, self.Data)

  def test_viable_init(self, K=7):
    ''' Verify hmodel after init can be used to perform E-step
    '''
    for initname in ['randexamples', 'randsoftpartition']:
      initParams = dict(initname=initname, seed=0, K=K)
      self.hmodel.init_global_params(self.Data, **initParams)
      LP = self.hmodel.calc_local_params(self.Data)
      resp = LP['resp']
      assert np.all(np.logical_and(resp >=0, resp <= 1.0))

  def test_consistent_random_seed(self, K=7):
    hmodel = self.hmodel 
    for initname in ['randexamples', 'randsoftpartition']:
      initParams = dict(initname=initname, seed=0, K=K)
      hmodel2 = self.hmodel.copy()
      hmodel.init_global_params(self.Data, **initParams)
      hmodel2.init_global_params(self.Data, **initParams)
      assert np.allclose(hmodel.allocModel.w, hmodel2.allocModel.w)
      assert np.allclose(hmodel.obsModel.comp[0].Sigma, hmodel2.obsModel.comp[0].Sigma)
      assert np.allclose(hmodel.obsModel.comp[K-1].Sigma, hmodel2.obsModel.comp[K-1].Sigma)
