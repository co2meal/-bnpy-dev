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

    aPDict = dict(alpha0=0.0)
    oPDict = dict(min_covar=1e-9)
    self.hmodel = HModel.InitFromData('EM','MixModel','ZMGauss', aPDict, oPDict, self.Data)

  def test_viable_init(self, K=7):
    ''' Verify hmodel after init can be used to perform E-step
    '''
    modelB = self.hmodel.copy()
    
    initParams = dict(initname='randexamples', seed=0, K=K)
    modelB.init_global_params(self.Data, **initParams)
    ModelWriter.save_model(modelB, '/tmp/')

    initSavedParams = dict(initname='/tmp/')
    self.hmodel.init_global_params(self.Data, **initParams)
    assert self.hmodel.K == K
    keysA = self.hmodel.allocModel.to_dict()
    keysB = modelB.allocModel.to_dict()
    assert len(keysA) == len(keysB)
