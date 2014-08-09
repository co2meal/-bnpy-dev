import numpy as np
import unittest
import timeit

from bnpy.allocmodel.seq import HMMUtil

FwdSetupCode = "\n\
import numpy as np \n\
from bnpy.allocmodel.seq import HMMUtil \n\
PRNG = np.random.RandomState(K*T % 100000) \n\
initPi = PRNG.rand(K) \n\
transPi = 3 * np.eye(K) + PRNG.rand(K,K) \n\
SoftEv = PRNG.rand(T, K)"

BwdSetupCode = "\n\
import numpy as np \n\
from bnpy.allocmodel.seq import HMMUtil \n\
PRNG = np.random.RandomState(K*T % 100000) \n\
initPi = PRNG.rand(K) \n\
transPi = 3 * np.eye(K) + PRNG.rand(K,K) \n\
SoftEv = PRNG.rand(T, K) \n\
fmsg, m = HMMUtil.FwdAlg_py(initPi, transPi, SoftEv)"

def SetupFwdArgs(K, T):
  PRNG = np.random.RandomState(K*T)
  initPi = PRNG.rand(K)
  transPi = 3 * np.eye(K) + PRNG.rand(K,K)
  SoftEv = PRNG.rand(T, K)
  return initPi, transPi, SoftEv


class TestFwdBwd(unittest.TestCase):

  def shortDescription(self):
    pass

  def test_correctness(self, K=9, T=17, verbose=1):
    ''' Verify FwdBwdAlg produces proper marginal distributions
    '''
    initPi, transPi, SoftEv = SetupFwdArgs(K, T)
    resp, respPair, logp = HMMUtil.FwdBwdAlg(initPi, transPi, np.log(SoftEv))
    assert np.allclose(np.sum(resp, axis=1), 1.0)
    for t in xrange(1, T):
      assert np.allclose(np.sum(respPair[t]), 1.0)


class TestFwd(unittest.TestCase):

  def test_correctness(self, K=9, T=17, verbose=1):
    ''' Verify FwdAlg produces matching results for both py and cpp versions
    '''
    if verbose: print ''
    PRNG = np.random.RandomState(K*T)

    initPi = np.ones(K, dtype=np.float)
    transPi = 3 * np.eye(K) + PRNG.rand(K, K)

    SoftEv = PRNG.rand(T, K)

    fmsgF, margPrObsF = HMMUtil.FwdAlg_cpp(initPi, transPi, SoftEv)
    fmsg, margPrObs = HMMUtil.FwdAlg_py(initPi, transPi, SoftEv)

    if verbose:
      for t in xrange(fmsg.shape[0]):
        print '           .........'
        print ' '.join(['% .6f' % (x) for x in fmsg[t]])
        print ' '.join(['% .6f' % (x) for x in fmsgF[t]])

    # Verify both algorithms produce the same output
    assert np.allclose(fmsg, fmsgF)
    assert np.allclose(margPrObs, margPrObsF)
    assert np.allclose(np.sum(fmsg, axis=1), 1.0)

  def test_correctness__manyTK(self):
    for T in [1, 2, 4, 10, 100, 1000]:
      for K in [1, 2, 4, 33]:
        self.test_correctness(K=K, T=T, verbose=0)

  def test_speed(self, K=10, T=100, nTrials=100, doFirstLineBlank=True):
    ''' Run time trial for particular input (average over many runs).
    '''
    if doFirstLineBlank:
      print ''
    global FwdSetupCode

    SetupCode = FwdSetupCode.replace('K', str(K))
    SetupCode = SetupCode.replace('T', str(T))

    cTimer = timeit.Timer("HMMUtil.FwdAlg_cpp(initPi, transPi, SoftEv)",
                          SetupCode)
    pyTimer = timeit.Timer("HMMUtil.FwdAlg_py(initPi, transPi, SoftEv)",
                           SetupCode)

    cElapsed = cTimer.timeit(number=nTrials) / nTrials
    pyElapsed = pyTimer.timeit(number=nTrials) / nTrials

    print ' K %5d | T %5d | py % 7.5f sec | c % 7.5f sec | speedup %7.2f' \
          % (K, T, pyElapsed, cElapsed, pyElapsed / cElapsed)


  def test_speed_vs_K(self):
    ''' Execute many time trials at various values of K, to determine trends
    '''
    print ''
    T = 1000
    for K in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
      self.test_speed(K=K, T=T, doFirstLineBlank=0)

  def test_speed_vs_T(self):
    ''' Execute many time trials at various values of T, to determine trends
    '''
    print ''
    K = 64
    for T in [16, 32, 64, 128, 256, 1024, 2048, 4096, 8192, 16384]:
      self.test_speed(K=K, T=T, doFirstLineBlank=0)


class TestBwd(unittest.TestCase):

  def test_correctness(self, K=8, T=44, verbose=1):
    if verbose: print ''
    PRNG = np.random.RandomState(K*T)
    initPi = np.ones(K, dtype=np.float)
    transPi = 3 * np.eye(K) + PRNG.rand(K,K)
    SoftEv = PRNG.rand(T, K)

    fmsg, margPrObs = HMMUtil.FwdAlg_py(initPi, transPi, SoftEv)

    bmsg = HMMUtil.BwdAlg_py(initPi, transPi, SoftEv, margPrObs)
    bmsgF = HMMUtil.BwdAlg_cpp(initPi, transPi, SoftEv, margPrObs)

    if verbose:
      for t in xrange(bmsg.shape[0]):
        print '           .........'
        print ' '.join(['% .6f' % (x) for x in bmsg[t]])
        print ' '.join(['% .6f' % (x) for x in bmsgF[t]])

    # Verify both algorithms produce the same output
    assert np.allclose(bmsg, bmsgF)
    assert not np.any(np.isnan(bmsg))
    assert not np.any(np.isnan(bmsgF))
    assert np.allclose(bmsg[-1,:], 1.0)

  def test_correctness__manyTK(self):
    for T in [1, 2, 4, 10, 100, 1000]:
      for K in [1, 2, 4, 33]:
        self.test_correctness(K=K, T=T, verbose=0)

  def test_speed(self, K=10, T=100, nTrials=100, doFirstLineBlank=True):
    ''' Run timed test to determine how much faster c++ version is than python
    '''
    if doFirstLineBlank == 1:
      print ''
    global BwdSetupCode

    SetupCode = BwdSetupCode.replace('K', str(K))
    SetupCode = SetupCode.replace('T', str(T))
    cTimer = timeit.Timer("HMMUtil.BwdAlg_cpp(initPi, transPi, SoftEv, m)",
                          SetupCode)
    pyTimer = timeit.Timer("HMMUtil.BwdAlg_py(initPi, transPi, SoftEv, m)",
                           SetupCode)

    cElapsed = cTimer.timeit(number=nTrials) / nTrials
    pyElapsed = pyTimer.timeit(number=nTrials) / nTrials

    print ' K %5d | T %5d | py % 7.5f sec | c % 7.5f sec | speedup %7.2f' \
          % (K, T, pyElapsed, cElapsed, pyElapsed / cElapsed)


  def test_speed_vs_K(self):
    ''' Execute many time trials at various values of K, to determine trends
    '''
    print ''
    T = 1000
    for K in [2, 4, 8, 16, 32, 64, 128]:
      self.test_speed(K=K, T=T, doFirstLineBlank=0)

  def test_speed_vs_T(self):
    ''' Execute many time trials at various values of T, to determine trends
    '''
    print ''
    K = 64
    for T in [16, 32, 64, 128, 256, 1024, 2048, 4096]:
      self.test_speed(K=K, T=T, doFirstLineBlank=0)
