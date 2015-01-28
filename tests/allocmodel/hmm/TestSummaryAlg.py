import unittest
import numpy as np

from bnpy.allocmodel.hmm.HMMUtil import FwdAlg_py, BwdAlg_py, SummaryAlg_py
from bnpy.allocmodel.hmm.HMMUtil import SummaryAlg_cpp

from bnpy.init.FromTruth import convertLPFromHardToSoft

class TestSummaryAlg_K4T2(unittest.TestCase):

  def shortDescription(self):
    return None

  def setUp(self, K=4, T=2):
    initPi = 1.0/K * np.ones(K)
    transPi = 1.0/K * np.ones((K,K))
    SoftEv = 10 * np.ones((T,K)) + np.random.rand(T,K)
    self._setUpFromParams(initPi, transPi, SoftEv)

  def _setUpFromParams(self, initPi, transPi, SoftEv):
    fMsg, margPrObs = FwdAlg_py(initPi, transPi, SoftEv)
    bMsg = BwdAlg_py(initPi, transPi, SoftEv, margPrObs)

    self.initPi = initPi
    self.transPi = transPi
    self.SoftEv = SoftEv
    self.fMsg = fMsg
    self.bMsg = bMsg
    self.margPrObs = margPrObs
    self.K = initPi.size
    self.T = SoftEv.shape[0]

  def test_python_equals_cpp(self):
    ''' Test both versions of C++ and python, verify same value returned
    '''
    print ''
    print '-------- python'
    T1, H1 = SummaryAlg_py(self.initPi, self.transPi, self.SoftEv,
                          self.margPrObs, self.fMsg, self.bMsg)
    if self.K < 5:
      print H1
    else:
      print H1[:5, :5]

    print '-------- cpp'
    T2, H2 = SummaryAlg_cpp(self.initPi, self.transPi, self.SoftEv,
                          self.margPrObs, self.fMsg, self.bMsg)
    if self.K < 5:
      print H2
    else:
      print H2[:5, :5]

    assert np.allclose(T1, T2)
    assert np.allclose(H1, H2)


class TestSummaryAlg_K4T100(TestSummaryAlg_K4T2):

  def setUp(self, K=4, T=100):
    parent = super(type(self), self)
    parent.setUp(K, T)


class TestSummaryAlg_K22T55(TestSummaryAlg_K4T2):

  def setUp(self, K=22, T=55):
    parent = super(type(self), self)
    parent.setUp(K, T)


class TestSummaryAlg_ToyData(TestSummaryAlg_K4T2):

  def setUp(self):
    T = 3000
    import DDToyHMM
    Data = DDToyHMM.get_data(seed=0, seqLens=(T))

    initPi = DDToyHMM.initPi
    transPi = DDToyHMM.transPi
    LP = dict(Z=Data.TrueParams['Z'])
    LP = convertLPFromHardToSoft(LP, Data)
    Keff = LP['resp'].shape[1]

    K = initPi.size
    SoftEv = np.zeros((T,K))
    SoftEv[:, :Keff] = LP['resp']
    SoftEv += 0.05
    self._setUpFromParams(initPi, transPi, SoftEv)
