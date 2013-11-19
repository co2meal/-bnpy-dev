'''
Unit-tests for ParamBag
'''

from bnpy.suffstats.SuffStatBag import SuffStatBag
import numpy as np
import unittest

class TestSuffStatBag(unittest.TestCase):
  def shortDescription(self):
    return None
  
  def test_ampFactor(self, K=2, D=2):
    SS = self.makeSuffStatBagAndFillWithOnes(K,D)
    SS.applyAmpFactor(3.0)
    assert np.allclose(SS.s, 3.0)
    assert np.allclose(SS.N,  3.0*np.ones(K))   
    assert np.allclose(SS.x,  3.0*np.ones((K,D)))   
    assert np.allclose(SS.xxT,  3.0*np.ones((K,D,D)))   
    assert SS.hasAmpFactor()
    assert SS.ampF == 3.0

  def makeSuffStatBagAndFillWithOnes(self, K, D):
    SS = SuffStatBag(K=K, D=D)
    s = 1.0
    N = np.ones(K)
    x = np.ones((K,D))
    xxT = np.ones((K,D,D))
    SS.setField('s', s)
    SS.setField('N', N, dims='K')
    SS.setField('x', x, dims=('K','D'))
    SS.setField('xxT', xxT, dims=('K','D','D'))
    return SS

  def addELBOtoSuffStatBag(self, SS, K):
    SS.setELBOTerm('Elogz', np.ones(K), dims='K')
    SS.setELBOTerm('Econst', 1.0, dims=None)
    SS.setMergeTerm('Elogz', 2*np.ones((K,K)), dims=('K','K'))
    return SS

  def getExpectedMergedFields(self, K, D, kA=0):
    s = 2.0
    N = np.ones( K-1)
    N[kA] = 2
    x = np.ones((K-1, D))
    x[kA] = 2
    xxT = np.ones((K-1,D,D))
    xxT[kA] = 2
    return s, N, x, xxT

  def test_mergeComps_K2_D3_noELBO(self, K=2, D=3):
    SS = self.makeSuffStatBagAndFillWithOnes(K=K, D=D)
    SS.mergeComps(0, 1)

    assert SS.K == K - 1
    assert np.allclose(SS.s, 2)
    assert np.allclose(SS.N, 2)
    assert np.allclose(SS.x, 2*np.ones(D))
    assert np.allclose(SS.xxT, 2*np.ones((D,D)))


  def test_mergeComps_K2_D3_withELBO(self, K=2, D=3):
    SS = self.makeSuffStatBagAndFillWithOnes(K, D)
    self.addELBOtoSuffStatBag(SS, K)
    SS.mergeComps(0, 1)

    assert SS.K == K - 1
    assert np.allclose(SS.s, 2)
    assert np.allclose(SS.N, 2)
    assert np.allclose(SS.x, 2*np.ones(D))
    assert np.allclose(SS.xxT, 2*np.ones((D,D)))
    
    assert np.allclose(SS.getELBOTerm('Elogz'), 2.0)
    assert np.allclose(SS.getELBOTerm('Econst'), 1.0)
    assert SS._ELBOTerms.K == K - 1
    assert SS._MergeTerms.K == K - 1


  def test_mergeComps_K5_D3_withELBO_kA0(self, K=5, D=3):
    SS = self.makeSuffStatBagAndFillWithOnes(K, D)
    self.addELBOtoSuffStatBag(SS, K)
    SS.mergeComps(0, 1)
    s, N, x, xxT = self.getExpectedMergedFields(K, D)

    assert SS.K == K - 1
    assert SS._ELBOTerms.K == K - 1
    assert SS._MergeTerms.K == K - 1
    
    assert np.allclose(SS.s, s)
    assert np.allclose(SS.N, N)
    assert np.allclose(SS.x, x)
    assert np.allclose(SS.xxT, xxT)
    
    assert np.allclose(SS.getELBOTerm('Elogz'), [2., 1, 1, 1])
    assert np.allclose(SS.getELBOTerm('Econst'), 1.0)
    assert np.all(np.isnan(SS._MergeTerms.Elogz[0,1:]))
    assert np.all(np.isnan(SS._MergeTerms.Elogz[:0,0]))


  def test_mergeComps_K5_D3_withELBO_kA3(self, K=5, D=3):
    SS = self.makeSuffStatBagAndFillWithOnes(K, D)
    self.addELBOtoSuffStatBag(SS, K)
    SS.mergeComps(3, 4)
    s, N, x, xxT = self.getExpectedMergedFields(K, D, kA=3)
    
    assert SS.K == K - 1
    assert SS._ELBOTerms.K == K - 1
    assert SS._MergeTerms.K == K - 1
    
    assert np.allclose(SS.s, s)
    assert np.allclose(SS.N, N)
    assert np.allclose(SS.x, x)
    assert np.allclose(SS.xxT, xxT)
    
    assert np.allclose(SS.getELBOTerm('Elogz'), [1., 1, 1, 2.])
    assert np.allclose(SS.getELBOTerm('Econst'), 1.0)
    assert np.all(np.isnan(SS._MergeTerms.Elogz[3,4:]))
    assert np.all(np.isnan(SS._MergeTerms.Elogz[:3,3]))

  def test_mergeComps_K5_D3_withELBO_back2back(self, K=5, D=3):
    SS = self.makeSuffStatBagAndFillWithOnes(K, D)
    self.addELBOtoSuffStatBag(SS, K)
    SS.mergeComps(3, 4)
    SS.mergeComps(0, 1)
    
    assert SS.K == K - 2
    assert SS._ELBOTerms.K == K - 2
    assert SS._MergeTerms.K == K - 2
    