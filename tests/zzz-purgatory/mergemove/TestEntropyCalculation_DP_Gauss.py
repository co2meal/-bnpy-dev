import numpy as np
from TestPlanner_DP_Gauss import DPGaussTestBase

class TestEntropyCalc(DPGaussTestBase):
  __test__ = True

  def setUp(self):
    self.MakeData()
    self.MakeModelWithTrueComps()

  def MakeData(self, N=10000):
    import AsteriskK8
    PRNG = np.random.RandomState(0)

    Data = AsteriskK8.get_data(nObsTotal=N, seed=425)

    self.Data = Data

    self.TrueResp = np.zeros( (N,8))
    self.DupResp = np.zeros( (N,8*2))
    for n in range(Data.nObs):
      k = Data.TrueLabels[n]
      self.TrueResp[n, k] = 1
      self.DupResp[n, k] = 0.5
      self.DupResp[n, k+8] = 0.5

  def test_entropy_calculation(self):
    print ''
    LP = self.hmodel.calc_local_params(self.Data)

    flags = dict(doPrecompEntropy=1, doPrecompMergeEntropy=1)
    flags2 = dict(doPrecompEntropy=1, doPrecompMergeEntropy=2)

    SS = self.hmodel.get_global_suff_stats(self.Data, LP, **flags)
    SS2 = self.hmodel.get_global_suff_stats(self.Data, LP, **flags2)

    origH = -1 * SS.getELBOTerm('ElogqZ').sum()

    for kA in xrange(SS.K):
      for kB in xrange(kA+1, SS.K):

        exactSS = SS.copy()
        boundSS = SS2.copy()
        exactSS.mergeComps(kA, kB)
        boundSS.mergeComps(kA, kB)

        exactH = -1*exactSS.getELBOTerm('ElogqZ').sum()
        boundH = -1*boundSS.getELBOTerm('ElogqZ').sum()

        assert origH >= exactH
        assert exactH >= boundH


  def test_entropy_calc__ManyNonOverlappingPairs(self):
    ''' Try to calculate entropy for merge of (0,1) and (2,3) and (4,5)
    '''
    print ''
    LP = self.hmodel.calc_local_params(self.Data)

    flags = dict(doPrecompEntropy=1, doPrecompMergeEntropy=1)
    flags2 = dict(doPrecompEntropy=1, doPrecompMergeEntropy=2)

    SS = self.hmodel.get_global_suff_stats(self.Data, LP, **flags)
    SS2 = self.hmodel.get_global_suff_stats(self.Data, LP, **flags2)

    origH = -1 * SS.getELBOTerm('ElogqZ').sum()

    exactSS = SS.copy()
    boundSS = SS2.copy()

    for mPair in [ (0,1), (2,3), (4,5)]:
      kA = mPair[0]
      kB = mPair[1]

      exactSS.mergeComps(kA, kB)
      boundSS.mergeComps(kA, kB)

      exactH = -1*exactSS.getELBOTerm('ElogqZ').sum()
      boundH = -1*boundSS.getELBOTerm('ElogqZ').sum()

      print '%.6e' % (origH)
      print '%.6e' % (exactH)
      print '%.6e' % (boundH)

      assert origH >= exactH
      assert exactH >= boundH

  def test_entropy_calc__ManyOverlappingPairs(self):
    ''' Try to calculate entropy for merge of (0,1) over and over again
    '''
    print ''
    LP = self.hmodel.calc_local_params(self.Data)

    flags2 = dict(doPrecompEntropy=1, doPrecompMergeEntropy=2)
    SS2 = self.hmodel.get_global_suff_stats(self.Data, LP, **flags2)

    origH = -1 * SS2.getELBOTerm('ElogqZ').sum()

    boundSS = SS2.copy()
    for pos, mPair in enumerate([ (0,1), (0,1), (0,1), (0,1), (0,1)]):
      kA = mPair[0]
      kB = mPair[1]

      LP['resp'][:, kA] += LP['resp'][:, kB]
      LP['resp'] = np.delete(LP['resp'], kB, axis=1)
      SS = self.hmodel.get_global_suff_stats(self.Data, LP, doPrecompEntropy=1)

      boundSS.mergeComps(kA, kB)
      exactH = -1*SS.getELBOTerm('ElogqZ').sum()
      boundH = -1*boundSS.getELBOTerm('ElogqZ').sum()

      print SS.K, boundSS.K
      print '%.6e' % (exactH)
      print '%.6e' % (boundH)
      assert exactH >= boundH
