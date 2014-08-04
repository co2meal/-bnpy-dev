import numpy as np
import unittest
import sys
import copy

import bnpy
import StandardNormalK1

from bnpy.mergemove import OptimizerMultiwayMerge as OMM

class TestOptimizerMultiwayMerge(unittest.TestCase):
  
  def setUp(self):
    ''' Create allocmodel and obsmodel that need a merge
    '''
    Data = StandardNormalK1.get_data()
    model, _, _ = bnpy.run(Data, 'DPMixModel', 'Gauss', 'VB',
                      K=5, initname='randexamples',
                      nLap=25, printEvery=0, saveEvery=0)
    LP = model.calc_local_params(Data)
    SS = model.get_global_suff_stats(Data, LP)
    model.update_global_params(SS)

    aFunc = model.allocModel.calcMergeGap_alph
    oFunc = model.obsModel.calcMergeGap_alph

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


  def test__find_optimum__sensitivity_to_init(self, PRNG=np.random.RandomState(0)):
    ''' Determine if many local optima exist for single instance of find_optimum
    '''
    SS = self.SS
    for kdel in xrange(SS.K):
      alphPrev = None
      for trial in xrange(10):
        initalph = PRNG.rand(SS.K-1)
        initalph /= initalph.sum()
        alph, f, Info = OMM.find_optimum(SS, kdel, self.aFunc, self.oFunc,
                                             initalph=initalph)
        msg = ''
        if alphPrev is None:
          alphPrev = alph.copy()
        else:
          if not np.allclose(alph, alphPrev, atol=0.001):
            msg = '**'
        alphStr = ' '.join(['%.4f' % (x) for x in alph])
        print '%s %s' % (alphStr, msg)
      print ' '

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

