'''
Unit tests for ModelReader.py
'''
from bnpy.ioutil import ModelWriter
from bnpy import HModel
from bnpy.data import XData
import numpy as np
import unittest

class TestModelReader(unittest.TestCase):
  def shortDescription(self):
    return None

  def setUp(self):
    PRNG = np.random.RandomState(867)
    X = PRNG.randn(100,2)
    self.Data = XData(X=X)

    aPDict = dict(alpha0=0.0)
    oPDict = dict(min_covar=1e-9)
    self.hmodel = HModel.InitFromData('EM','MixModel','ZMGauss', aPDict, oPDict, self.Data)
    initParams = dict(initname='randexamples', seed=0, K=5)
    self.hmodel.init_global_params(self.Data, **initParams)


  def test_save_model(self):
    prefix = 'Best'
    ModelWriter.save_model(self.hmodel, '/tmp/', prefix)
    

