import numpy as np
import unittest
import sys
import copy

import bnpy

from bnpy.mergemove import OptimizerMultiwayMerge as OMM

class TestSoftMergeEntropy(unittest.TestCase):

  def test_calcSoftMergeEntropy_ManyTrials(self, K=4, nTrial=5):
    PRNG = np.random.RandomState(0)
    for kdel in xrange(K):
      for trial in xrange(nTrial):
        xalph = PRNG.rand(K)
        xalph[kdel] = 0
        xalph = xalph / np.sum(xalph)
        self.test_calcSoftMergeEntropy(K, kdel, xalph)

  def test_calcSoftMergeEntropy(self, K=6, kdel=0, xalph=None):
    print ''
    if xalph is None:
      xalph = 1.0/(K-1) * np.ones(K)
      xalph[kdel] = 0
    print xalph

    SS = bnpy.suffstats.SuffStatBag(K=K, D=1)
    SS.setField('N', 10 * np.ones(K), dims='K')
    SS.setELBOTerm('ElogqZ', -33 * np.ones(K), dims='K')
    SS.setMergeTerm('ElogqZ', 42 * np.ones(K), dims='K')

    ## Option 1: calc the entropy gap via DPMixModel's built-in method
    PriorSpec = dict(truncType='z', alpha0=5)
    aModel = bnpy.allocmodel.DPMixModel('VB', PriorSpec)
    ELBOgap = aModel.calcSoftMergeEntropyGap(SS, kdel, xalph)

    ## Option 2: calc the entropy gap pedantically,
    ##  by calculating the entropy *before* and *after*, and taking the diff
    propSS = SS.copy()
    propSS.multiMergeComps(kdel, xalph)
    Hbefore = -1 * SS.getELBOTerm('ElogqZ').sum()
    Hafter = -1 * propSS.getELBOTerm('ElogqZ').sum()
    ELBOgap2 = Hafter - Hbefore

    print ELBOgap
    print ELBOgap2
    assert np.allclose(ELBOgap, ELBOgap2)

########################################################### DPMix SoftMergeGap
###########################################################
class TestSoftMergeGap_DPMixModel(unittest.TestCase):
  def test_calcSoftMergeGap_ManyTrials(self, K=4, nTrial=5):
    PRNG = np.random.RandomState(0)
    for kdel in xrange(K):
      for trial in xrange(nTrial):
        xalph = PRNG.rand(K)
        xalph[kdel] = 0
        xalph = xalph / np.sum(xalph)
        self.test_calcSoftMergeGap(K, kdel, xalph)

  def test_calcSoftMergeGap(self, K=6, kdel=0, xalph=None):
    print ''
    if xalph is None:
      xalph = 1.0/(K-1) * np.ones(K)
      xalph[kdel] = 0
    print xalph

    SS = bnpy.suffstats.SuffStatBag(K=K, D=1)
    SS.setField('N', 10 * np.ones(K), dims='K')
    PriorSpec = dict(truncType='z', alpha0=5)
    aModel = bnpy.allocmodel.DPMixModel('VB', PriorSpec)
    aModel.update_global_params(SS)

    ## Option 1: calc the ELBO gap via DPMixModel's built-in method
    ELBOgap = aModel.calcSoftMergeGap(SS, kdel, xalph)

    ## Option 2: calc the entropy gap pedantically,
    ##  by calculating the entropy *before* and *after*, and taking the diff
    propSS = SS.copy()
    propSS.multiMergeComps(kdel, xalph)
    propModel = bnpy.allocmodel.DPMixModel('VB', PriorSpec)
    propModel.update_global_params(propSS)

    ELBObefore = aModel.E_logpZ(SS) \
                 + aModel.E_logpV() \
                 - aModel.E_logqV()
    ELBOafter = propModel.E_logpZ(propSS) \
                + propModel.E_logpV() \
                - propModel.E_logqV()
    ELBOgap2 = ELBOafter - ELBObefore

    print ELBOgap
    print ELBOgap2
    assert np.allclose(ELBOgap, ELBOgap2)

########################################################### Gauss SoftMergeGap
###########################################################
class TestSoftMergeGap_Gauss(unittest.TestCase):
  def test_calcSoftMergeGap_ManyTrials(self, K=4, nTrial=5):
    PRNG = np.random.RandomState(0)
    for kdel in xrange(K):
      for trial in xrange(nTrial):
        xalph = PRNG.rand(K)
        xalph[kdel] = 0
        xalph = xalph / np.sum(xalph)
        self.test_calcSoftMergeGap(K, kdel, xalph, PRNG=PRNG)

  def test_calcSoftMergeGap(self, K=6, kdel=0, xalph=None, D=1,
                                  PRNG=np.random.RandomState(0)):
    print ''
    if xalph is None:
      xalph = 1.0/(K-1) * np.ones(K)
      xalph[kdel] = 0
    print xalph

    PriorSpec = dict(ECovMat='eye', nu=D+1.234, sF=1.0, kappa=1e-4)
    oModel = bnpy.obsmodel.GaussObsModel('VB', D=D, **PriorSpec)

    X = PRNG.randn(10, D)
    R = PRNG.rand(10, K)
    R /= R.sum(axis=1)[:,np.newaxis]

    Data = bnpy.data.XData(X)
    LP = dict(resp=R)
    SS = oModel.get_global_suff_stats(Data, None, LP)
    oModel.update_global_params(SS)


    ## Option 1: calc the ELBO gap via DPMixModel's built-in method
    ELBOgap = oModel.calcSoftMergeGap(SS, kdel, xalph)

    ## Option 2: calc the entropy gap pedantically,
    ##  by calculating the entropy *before* and *after*, and taking the diff
    propSS = SS.copy()
    propSS.multiMergeComps(kdel, xalph)
    propModel = bnpy.obsmodel.GaussObsModel('VB', D=D, **PriorSpec)
    propModel.update_global_params(propSS)

    ELBObefore = oModel.calcELBO_Memoized(SS) 
    ELBOafter = propModel.calcELBO_Memoized(propSS)
    ELBOgap2 = ELBOafter - ELBObefore

    print ELBOgap
    print ELBOgap2
    assert np.allclose(ELBOgap, ELBOgap2)




########################################################### Optimizer
###########################################################

class TestOptimizer(unittest.TestCase):
  
  def MakeData(self):
    import StandardNormalK1
    return StandardNormalK1.get_data()
    
  def GetK(self):
    return 5

  def setUp(self):
    ''' Create allocmodel and obsmodel that need a merge
    '''
    Data = self.MakeData()
    K = self.GetK()
    model, _, _ = bnpy.run(Data, 'DPMixModel', 'Gauss', 'VB',
                      K=K, initname='randexamples',
                      nLap=25, printEvery=0, saveEvery=0)
    LP = model.calc_local_params(Data)
    SS = model.get_global_suff_stats(Data, LP)
    model.update_global_params(SS)

    aFunc = model.allocModel.calcSoftMergeGap_alph
    oFunc = model.obsModel.calcSoftMergeGap_alph

    self.model = model
    self.SS = SS
    self.aFunc = aFunc
    self.oFunc = oFunc

  def test__find_optimum(self, nTrial=10, PRNG=np.random.RandomState(0)):
    ''' Verify that the alph returned by find_optimum is a local minimum
    '''
    SS = self.SS
    kdel = 0
    alph, f, Info = OMM.find_optimum(SS, kdel, self.aFunc, self.oFunc)

    fopt = OMM.objFunc_alph(alph, SS, kdel, self.aFunc, self.oFunc)
    print '%.5e [optimal]' % (fopt)
    
    for kk in xrange(SS.K - 1):
      alph = np.zeros(SS.K-1)
      alph[kk] = 1
      f = OMM.objFunc_alph(alph, SS, kdel, self.aFunc, self.oFunc)
      print '%.5e [all-to-one %d]' % (f, kk)
      assert f >= fopt

    for nn in xrange(nTrial):
      alph = PRNG.rand(SS.K - 1)
      alph /= np.sum(alph)

      f = OMM.objFunc_alph(alph, SS, kdel, self.aFunc, self.oFunc)
      print '%.5e [rand trial %d]' % (f, nn)
      assert f >= fopt


  def test__find_optimum__sensitivity_to_init(self, 
                                              PRNG=np.random.RandomState(0)):
    ''' Determine if many local optima exist for single instance of find_optimum
    '''
    SS = self.SS
    for kdel in xrange(SS.K):
      print '------------------------------------- kdel %d N %.1f' % (kdel, SS.N[kdel])
      alphPrev = None
      for trial in xrange(4):
        initalph = PRNG.rand(SS.K-1)
        initalph /= initalph.sum()
        alph, f, Info = OMM.find_optimum(SS, kdel, self.aFunc, self.oFunc,
                                             initalph=initalph)
        msg = '%.2e' % (f)
        if alphPrev is None:
          alphPrev = alph.copy()
        else:
          if not np.allclose(alph, alphPrev, atol=0.001):
            msg += '**'
        alphStr = ' '.join(['%.3f' % (x) for x in alph])
        print '%s %s' % (alphStr, msg)
      print ' '

class TestOptimizer_AsteriskK13(TestOptimizer):
  def MakeData(self):
    import AsteriskK8
    return AsteriskK8.get_data()

  def GetK(self):
    return 13


"""
  def test__calcMergeGap__allocmodel(self):
    ''' For the allocmodel, calc merge ELBO two ways, make sure they both agree
        method A : make orig and merge models, call calcELBO() on both, and calc diff
        method B : call the calcMergeGap() method directly on orig model
    '''
    SS = self.SS
    aM = self.model.allocModel
    SS.setELBOTerm('ElogqZ', np.zeros(SS.K), dims=('K'))

    kdel = 0
    alph = np.asarray([.5, .1, .2,.2])

    alphx = np.zeros(SS.K)
    alphx[:kdel] = alph[:kdel]
    alphx[kdel+1:] = alph[kdel:]

    propSS = SS.copy()
    propSS.multiMergeComps(kdel, alphx)
    propM = copy.deepcopy(aM)
    propM.update_global_params(propSS)

    elboBEFORE = aM.calc_evidence(None, SS, None)
    elboAFTER = propM.calc_evidence(None, propSS, None)
    aELBODelta = elboAFTER - elboBEFORE

    aELBODelta2 = aM.calcMergeGap_NonEntropy(SS, kdel, alphx)
    print aELBODelta
    print aELBODelta2
    assert np.allclose(aELBODelta, aELBODelta2)

  def test__calcMergeGap__obsmodel(self):
    ''' For the obsmodel, calculate merge ELBO two ways, make sure they both agree
        method A : make orig and merge models, call calcELBO() on both, and calc diff
        method B : call the calcMergeELBO() method directly on orig model
    '''
    SS = self.SS
    obsM = self.model.obsModel

    kdel = 0
    alph = np.asarray([.5, .2, .1, .2])

    alphx = np.zeros(SS.K)
    alphx[:kdel] = alph[:kdel]
    alphx[kdel+1:] = alph[kdel:]

    # now actually try the merge
    propSS = SS.copy()
    propSS.multiMergeComps(kdel, alphx)
    propM = copy.deepcopy(obsM)
    propM.updatePost(propSS)

    elboBEFORE = obsM.calcELBO_Memoized(SS, doFast=0)
    elboAFTER = propM.calcELBO_Memoized(propSS, doFast=0)
    oELBODelta = elboAFTER - elboBEFORE
    oELBODelta2 = obsM.calcMergeGap(SS, kdel, alphx)
    print oELBODelta
    print oELBODelta2
    assert np.allclose(oELBODelta, oELBODelta2)
"""