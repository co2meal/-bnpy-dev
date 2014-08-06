'''
TestDPMixPEHardMerge.py

Verify that the following methods for calculating the ELBO yield the same results
* calc_evidence(beforeModel) - calc_evidence(beforeModel)
* beforeModel.calcHardMergeGap_AllPairs(SS)[kA, kB]
* beforeModel.calcHardMergeGap(SS, kA, kB)

Note: this is all done by **ignoring** the entropy.
This allows us to verify that we can accurately track non-entropy terms.
'''

import numpy as np
import unittest
import copy

import bnpy

class TestHardMerge(unittest.TestCase):

  def setUp(self):
    K = 5

    SS = bnpy.suffstats.SuffStatBag(K=5, D=1)
    #SS.setField('N', np.arange(K), dims='K')
    PRNG = np.random.RandomState(0)
    Nvec = PRNG.rand(K) * 80
    SS.setField('N', Nvec, dims='K')

    ## Set Entropy to zero, so it doesnt factor into calculations
    SS.setELBOTerm('ElogqZ', np.zeros(K), dims='K')
    SS.setMergeTerm('ElogqZ', np.zeros((K,K)), dims=('K', 'K'))

    ## Update beforeModel so it is current with the sufficient stats
    ## (this makes necessary slack terms vanish!)
    beforeModel = bnpy.allocmodel.DPMixPE('VB', dict(gamma0=5))
    beforeModel.update_global_params(SS)

    self.beforeModel = beforeModel
    self.beforeSS = SS

  def test_calcHardMergeGap(self):
    print ''
    beforeModel = self.beforeModel
    beforeSS = self.beforeSS
    beforeK = self.beforeSS.K

    ## Calculate ELBO of original, "before" model
    beforeELBO = beforeModel.calc_evidence(None, beforeSS, None)

    ## Method 1: Calc gap via call to "AllPairs" method
    GapMat = beforeModel.calcHardMergeGap_AllPairs(beforeSS)

    ## Allocate storage for "after" model
    afterModel = copy.deepcopy(beforeModel)
    for kA in xrange(beforeK):
      for kB in xrange(kA+1, beforeK):
        print '%d, %d' % (kA, kB)

        ## Method 2: Calc gap via call to "calcHardMergeGap" method
        gapAB = beforeModel.calcHardMergeGap(beforeSS, kA, kB)

        ## Method 3: Direct construction of merge model 
        ## followed by explicit gap calculation (after - before)
        afterSS = beforeSS.copy()
        afterSS.mergeComps(kA, kB)
        afterModel.update_global_params(afterSS)
        assert afterModel.K == beforeModel.K - 1

        afterELBO = afterModel.calc_evidence(None, afterSS, None)
        gapDirect = afterELBO - beforeELBO

        ## Verify that all three methods produce the same gap
        print '  %.4f %.4f' % (gapAB, gapDirect)
        assert np.allclose(gapAB, gapDirect)
        assert np.allclose(gapAB, GapMat[kA, kB])
