import sys
import numpy as np
from nose.plugins.skip import Skip, SkipTest
from scipy.optimize import approx_fprime
import warnings
import unittest

import bnpy
import bnpy.allocmodel.admix.OptimizerForMAPDocTopicSticks as OptimSB

np.set_printoptions(precision=3, suppress=False, linewidth=140)

def np2flatstr(xvec, Kmax=10):
  return ' '.join( ['%9.3f' % (x) for x in xvec[:Kmax]])


########################################################### TestPrior Only
###########################################################
def CreatePriorOnlyProblem(K=2, Nd=5, seed=0):
  Xd = np.zeros(Nd)
  Ld = 1./K * np.ones((Nd,K))
  avec = 0.53 * np.ones(K)
  bvec = 0.1 * np.arange(1, K+1)
  return dict(Xd=Xd, Ld=Ld, avec=avec, bvec=bvec)

class TestPriorOnlyProblem_K2(unittest.TestCase):
  def shortDescription(self):
    return None

  def setUp(self):
    self.K = 2
    self.kwargs = CreatePriorOnlyProblem(K=self.K, Nd=5)

  def get_ideal_eta(self):
    # return transformed version of mean under prior
    avec = self.kwargs['avec']
    bvec = self.kwargs['bvec']
    return OptimSB.invsigmoid( avec / (avec + bvec))

  def test__objFunc_unconstrained__returns_without_error(self):
    print ''
    PRNG = np.random.RandomState(0)
    for s in [1, 2, 3, 4, 5, 6, 7, 8]:
      vd = PRNG.rand(self.K)
      eta = OptimSB.invsigmoid(vd)

      fc = OptimSB.objFunc_unconstrained(eta, **self.kwargs)
      #try:
      #  fc = float(f)
      #except:
      #  fc, g = f
      print fc
      assert np.asarray(fc).ndim == 0
      assert not np.any(np.isinf(fc))
      
  def test__find_optimum__returns_without_error(self):
    print ''
    PRNG = np.random.RandomState(0)
    for s in [1, 2, 3, 4, 5, 6, 7, 8]:
      vd = PRNG.rand(self.K)
      eta = OptimSB.invsigmoid(vd)

      eta, f, Info = OptimSB.find_optimum(initeta=eta, **self.kwargs)

      print '%.4e %s' % (f, np2flatstr(eta))
      assert np.asarray(f).ndim == 0
      assert not np.any(np.isinf(f))

      assert eta.size == self.K and eta.ndim == 1
      assert not np.any(np.isinf(eta))

  def test__find_optimum__findsIdealSolution(self):
    print ''
    PRNG = np.random.RandomState(0)

    etaIdeal = self.get_ideal_eta()
    fIdeal = OptimSB.objFunc_unconstrained(etaIdeal, **self.kwargs)
    print '%.4e %s  IDEAL' % (fIdeal, np2flatstr(etaIdeal))
    for s in [1, 2, 3, 4, 5, 6, 7, 8]:
      vd = PRNG.rand(self.K)
      eta = OptimSB.invsigmoid(vd)

      eta, f, Info = OptimSB.find_optimum(initeta=eta, **self.kwargs)

      print '%.4e %s' % (f, np2flatstr(eta))
      assert np.allclose(eta, etaIdeal, atol=0.0001)


########################################################### Data
###########################################################
import BarsK6V9
Data = BarsK6V9.get_data(nWordsPerDoc=1000, nDocTotal=30)

def CreateSmallBarsProblem(docID):
  start = Data.doc_range[docID,0]
  stop = Data.doc_range[docID,1]

  Xd = Data.word_count[ start:stop].copy()
  Ld = Data.TrueParams['topics'][:, Data.word_id[start:stop]].T.copy()
  _, K = Ld.shape

  avec = 0.1 * np.ones(K)
  bvec = 0.1 * np.ones(K)
  return dict(Xd=Xd, Ld=Ld, avec=avec, bvec=bvec)

class TestBarsK6_Doc0(unittest.TestCase):
  def shortDescription(self):
    return None

  def setUp(self, docID=0):
    self.docID = docID
    self.commonSetUp()

  def commonSetUp(self):
    self.K = 6
    self.kwargs = CreateSmallBarsProblem(self.docID)
    ideal_pi = Data.TrueParams['alphaPi'][self.docID, :]
    ideal_pi = np.hstack([ideal_pi, 0.0001])
    self.idealPi = ideal_pi / np.sum(ideal_pi)

  def get_ideal_eta(self):
    return OptimSB.invsigmoid(OptimSB._beta2v(self.idealPi))

  def test__find_optimum__findsIdealSolution(self):
    print ''
    PRNG = np.random.RandomState(0)

    fvalPattern = ' ' * 75 + ' %.6e  %s'

    fIdeal = np.inf
    if hasattr(self, 'piIdeal'):
      etaIdeal = self.get_ideal_eta()
      fIdeal = OptimSB.objFunc_unconstrained(etaIdeal, **self.kwargs)
      piIdeal = OptimSB._v2beta(OptimSB.sigmoid(etaIdeal))
     
      print fvalPattern % (fIdeal, 'IDEAL')
      print '   %s' % (np2flatstr(piIdeal))

    print '================================================='
    
    for s in [1, 2, 3, 4, 5, 6, 7, 8]:
      vd = PRNG.rand(self.K)
      initeta = OptimSB.invsigmoid(vd)

      eta, f, Info = OptimSB.find_optimum_multiple_tries(initeta=initeta,
                                                          **self.kwargs)

      grad = approx_fprime(eta, Info['objFunc'], 1e-10*np.ones(self.K))
      piEst = OptimSB._v2beta(OptimSB.sigmoid(eta))

      fbuffer = np.maximum(1e-4, 1e-4*np.abs(f))
      if f  <= fIdeal + fbuffer:
        warnMsg = ''
      else:
        warnMsg = '!!!!'

      if f + fbuffer < fIdeal:
        fIdeal = f

      if Info['nOverflow'] > 0:
        warnMsg += ' %d' % (Info['nOverflow'])

      print fvalPattern % (f, warnMsg)
      print '   %s' % (np2flatstr(piEst))
      #print '   %s' % (np2flatstr(grad))
      print '   %s' % (Info['msg'])
      #assert np.allclose(piIdeal, piEst, atol=0.04)

class TestBarsK6_Doc1(TestBarsK6_Doc0):

  def setUp(self, docID=1):
    self.docID = docID
    self.commonSetUp()

class TestBarsK6_Doc2(TestBarsK6_Doc0):

  def setUp(self, docID=2):
    self.docID = docID
    self.commonSetUp()

class TestBarsK6_Doc3(TestBarsK6_Doc0):

  def setUp(self, docID=3):
    self.docID = docID
    self.commonSetUp()


########################################################### K10 BarsData
###########################################################
import BarsK10V900
K10Data = BarsK10V900.get_data(topic_prior=0.8*np.ones(10), 
                               nWordsPerDoc=400, nDocTotal=30)

def CreateBarsK10Problem(docID):
  start = K10Data.doc_range[docID,0]
  stop = K10Data.doc_range[docID,1]

  Xd = K10Data.word_count[ start:stop].copy()
  Ld = K10Data.TrueParams['topics'][:, K10Data.word_id[start:stop]].T.copy()
  _, K = Ld.shape

  avec = 0.1 * np.ones(K)
  bvec = 0.1 * np.ones(K)
  return dict(Xd=Xd, Ld=Ld, avec=avec, bvec=bvec)


class TestBarsK10_Doc0(TestBarsK6_Doc0):
  def shortDescription(self):
    return None

  def setUp(self, docID=0):
    self.docID = docID
    self.commonSetUp()

  def commonSetUp(self):
    self.K = 10
    self.kwargs = CreateBarsK10Problem(self.docID)
    ideal_pi = K10Data.TrueParams['alphaPi'][self.docID, :]
    ideal_pi = np.hstack([ideal_pi, 0.0001])
    self.idealPi = ideal_pi / np.sum(ideal_pi)
    assert np.allclose(np.sum(self.idealPi), 1.0)

class TestBarsK10_Doc1(TestBarsK10_Doc0):

  def setUp(self, docID=1):
    self.docID = docID
    self.commonSetUp()

class TestBarsK10_Doc2(TestBarsK10_Doc0):

  def setUp(self, docID=2):
    self.docID = docID
    self.commonSetUp()

class TestBarsK10_Doc3(TestBarsK10_Doc0):

  def setUp(self, docID=3):
    self.docID = docID
    self.commonSetUp()


########################################################### K10 NIPS
###########################################################

import NIPSCorpus
NIPSData = NIPSCorpus.get_data()

def CreateNIPSProblem(docID, K=25):
  PRNG = np.random.RandomState(docID)
  rowIDs = PRNG.choice( NIPSData.nDoc, K, replace=False)

  DWMat = NIPSData.to_sparse_docword_matrix()[rowIDs].toarray()
  Topics = DWMat + 0.001 * PRNG.rand(K, NIPSData.vocab_size)
  Topics /= Topics.sum(axis=1)[:,np.newaxis]

  start = NIPSData.doc_range[docID,0]
  stop = NIPSData.doc_range[docID,1]
  Xd = NIPSData.word_count[start:stop].copy()
  Ld = Topics[:, NIPSData.word_id[start:stop]].T.copy()
  _, K = Ld.shape

  avec = 0.1 * np.ones(K)
  bvec = 0.1 * np.ones(K)
  return dict(Xd=Xd, Ld=Ld, avec=avec, bvec=bvec)

class TestNIPSK10_Doc0(TestBarsK6_Doc0):
  def shortDescription(self):
    return None

  def setUp(self, docID=0):
    self.docID = docID
    self.commonSetUp()

  def commonSetUp(self):
    self.K = 10
    self.kwargs = CreateNIPSProblem(self.docID, self.K)

class TestNIPSK10_Doc1(TestNIPSK10_Doc0):

  def setUp(self, docID=1):
    self.docID = docID
    self.commonSetUp()

class TestNIPSK10_Doc2(TestNIPSK10_Doc0):

  def setUp(self, docID=2):
    self.docID = docID
    self.commonSetUp()

class TestNIPSK10_Doc3(TestNIPSK10_Doc0):

  def setUp(self, docID=3):
    self.docID = docID
    self.commonSetUp()


########################################################### K50 NIPS
###########################################################
class TestNIPSK50_Doc0(TestBarsK6_Doc0):
  def shortDescription(self):
    return None

  def setUp(self, docID=0):
    self.docID = docID
    self.commonSetUp()

  def commonSetUp(self):
    self.K = 50
    self.kwargs = CreateNIPSProblem(self.docID, self.K)

class TestNIPSK50_Doc1(TestNIPSK50_Doc0):

  def setUp(self, docID=1):
    self.docID = docID
    self.commonSetUp()

class TestNIPSK50_Doc2(TestNIPSK50_Doc0):

  def setUp(self, docID=2):
    self.docID = docID
    self.commonSetUp()

class TestNIPSK50_Doc3(TestNIPSK50_Doc0):

  def setUp(self, docID=3):
    self.docID = docID
    self.commonSetUp()

########################################################### K200 science
###########################################################

import joblib
DUMP = joblib.load('/data/liv/liv-x/bnpy/local/dump/MemoLPELBODrop.dump')

def CreateScienceProblem(docID):

  Topics = DUMP['hmodel'].obsModel.getElogphiMatrix()
  Topics -= Topics.max(axis=1)[:, np.newaxis]
  Topics = np.exp(Topics)
  Topics /= Topics.sum(axis=1)[:,np.newaxis]
  assert np.allclose(np.sum(Topics, axis=1), 1.0)

  SciData = DUMP['Dchunk']
  start = SciData.doc_range[docID,0]
  stop = SciData.doc_range[docID,1]
  Xd = SciData.word_count[start:stop].copy()
  Ld = Topics[:, SciData.word_id[start:stop]].T.copy()
  _, K = Ld.shape

  avec = DUMP['hmodel'].allocModel.topicPrior1
  bvec = DUMP['hmodel'].allocModel.topicPrior0

  return dict(Xd=Xd, Ld=Ld, avec=avec, bvec=bvec)

class TestScienceK200_Doc0(TestNIPSK50_Doc0):

  def setUp(self, docID=0):
    self.docID = docID
    self.commonSetUp()

  def commonSetUp(self):
    self.K = 200
    self.kwargs = CreateScienceProblem(self.docID)





def CreateSmoothedScienceProblem(docID):

  Topics = DUMP['hmodel'].obsModel.getElogphiMatrix()
  Topics -= Topics.max(axis=1)[:, np.newaxis]
  Topics = np.exp(Topics)
  Topics /= Topics.sum(axis=1)[:,np.newaxis]

  PRNG = np.random.RandomState(docID)
  RandTopics = PRNG.rand(Topics.shape[0], Topics.shape[1])
  RandTopics /= RandTopics.sum(axis=1)[:,np.newaxis]
  
  Topics = RandTopics #0.8 * Topics + 0.2 * RandTopics
  assert np.allclose(np.sum(Topics, axis=1), 1.0)

  SciData = DUMP['Dchunk']
  start = SciData.doc_range[docID,0]
  stop = SciData.doc_range[docID,1]
  Xd = SciData.word_count[start:stop].copy()
  Ld = Topics[:, SciData.word_id[start:stop]].T.copy()
  _, K = Ld.shape

  avec = DUMP['hmodel'].allocModel.topicPrior1
  bvec = DUMP['hmodel'].allocModel.topicPrior0

  return dict(Xd=Xd, Ld=Ld, avec=avec, bvec=bvec)

class TestSmoothedK200_Doc0(TestNIPSK50_Doc0):

  def setUp(self, docID=0):
    self.docID = docID
    self.commonSetUp()

  def commonSetUp(self):
    self.K = 200
    self.kwargs = CreateSmoothedScienceProblem(self.docID)





"""
  def verify__objFunc_constrained(self, approx_grad=False):
    ''' Verify *constrained* objFunc return value has correct shape, domain
    '''
    print ''
    g = None
    K = self.v.size
    PRNG = np.random.RandomState(0)
    for s in xrange(10):
      rho = PRNG.rand(K)
      omega = 5 * (s+1) * np.ones(K)
      rhoomega = np.hstack([rho, omega])

      f = OptimSB.objFunc_constrained(rhoomega, approx_grad=approx_grad,
                                            **self.kwargs)
      try:
        f = float(f)
      except:
        f, g = f
      assert np.asarray(f).ndim == 0
      assert not np.any(np.isinf(f))
    return f, g    

  def verify__objFunc_unconstrained(self, approx_grad=False):
    ''' Verify *unconstrained* objFunc return value has correct shape, domain
    '''
    print ''
    g = None
    K = self.v.size
    PRNG = np.random.RandomState(0)
    for s in [1, 2, 3, 4, 5, 6, 7, 8]:
      rho = PRNG.rand(K)
      omega = 5.5 * s * PRNG.rand(K)
      rhoomega = np.hstack([rho, omega])
      c = OptimSB.rhoomega2c(rhoomega)
      f = OptimSB.objFunc_unconstrained(c, approx_grad=approx_grad,
                                            **self.kwargs)
      try:
        fc = float(f)
      except:
        fc, g = f
      assert np.asarray(fc).ndim == 0
      assert not np.any(np.isinf(fc))
      fro = OptimSB.objFunc_constrained(rhoomega, approx_grad=approx_grad,
                                            **self.kwargs)
      try:
        fro = float(fro)
      except:
        fro, _ = fro
      assert np.allclose(fc, fro)
  
    return fc, g

  ######################################################### known optimum
  #########################################################
  def test__known_optimum(self):
    ''' Verify that analytic optimum for rho,omega actually maximizes objFunc
    '''
    if self.kwargs['nDoc'] > 0:
      raise SkipTest
    K = self.kwargs['sumLogVd'].size
    rhoopt = 1.0 / (1.0 + self.kwargs['alpha']) * np.ones(K)
    omegaopt = (1.0 + self.kwargs['alpha']) * np.ones(K)
    roopt = np.hstack([rhoopt, omegaopt])
    fopt = OptimSB.objFunc_constrained(roopt, approx_grad=1, **self.kwargs)
    print ''
    print '%.5e ******' % (fopt)

    deltaRange = np.linspace(0.001, 0.05, 3)

    for sign in [1, -1]:
      for delta in deltaRange:
        ro = roopt.copy()
        ro[:K] += sign * delta
        ro[:K] = np.maximum(ro[:K], 1e-8)
        ro[:K] = np.minimum(ro[:K], 1-1e-8)

        f = OptimSB.objFunc_constrained(ro,  approx_grad=1, **self.kwargs)
        print '%.5e' % (f)
        print '    ', np2flatstr(ro[:K])
        print '    ', np2flatstr(ro[K:])
        assert fopt < f

  def test__gradient_at_known_optimum(self):
    ''' Verify gradient at analytic optimum is very near zero.
    '''
    if self.kwargs['nDoc'] > 0:
      raise SkipTest
    K = self.v.size
    rhoopt = 1.0 / (1.0 + self.kwargs['alpha']) * np.ones(K)
    omegaopt = (1.0 + self.kwargs['alpha']) * np.ones(K)
    roopt = np.hstack([rhoopt, omegaopt])
    fopt, gopt = OptimSB.objFunc_constrained(roopt, approx_grad=0,
                                             **self.kwargs)
    print ''
    print '%.5e' % (np.max(np.abs(gopt)))
    assert np.allclose(gopt, np.zeros(2*K))

  ######################################################### exact v approx grad
  #########################################################

  def test__gradient_equals_approx_grad(self):
    ''' Verify exact gradient computation is near the numerical estimate
    '''
    print ''
    K = self.v.size
    epsvec = np.hstack([1e-10*np.ones(K), 1e-8*np.ones(K)])

    objFunc = lambda r : OptimSB.objFunc_constrained(r, approx_grad=1,
                                             **self.kwargs)

    ro, f, Info = OptimSB.find_optimum(approx_grad=1, **self.kwargs)
    f, g = OptimSB.objFunc_constrained(ro, approx_grad=0,
                                             **self.kwargs)
    assert not np.any(np.isnan(g))
    ga = approx_fprime(ro, objFunc, epsvec)    

    absDiff = np.abs(g - ga)
    denom = np.abs(g)
    percDiff = np.abs(g - ga)/denom
    if np.sum(denom < 0.01) > 0:
      percDiff[denom < 0.01] = absDiff[denom < 0.01]
    print '-------------------------- Gradients AT OPT ****'
    print '  %s exact ' % (np2flatstr(g))
    print '  %s approx' % (np2flatstr(ga))

    roguess = np.hstack([self.v, 1+self.kwargs['nDoc']*np.ones(K)])
    fguess = OptimSB.objFunc_constrained(roguess, approx_grad=1, **self.kwargs)

    print '  %s rhoguess' % (np2flatstr(roguess[:K]))
    print '  %s rho' % (np2flatstr(ro[:K]))
    print '  %.4e fguess' % (fguess)
    print '  %.4e f' % (f)
    maxError = np.max(percDiff)
    assert maxError < 0.03

    PRNG = np.random.RandomState(0)
    for rr in np.linspace(0.01, 0.99, 10):
      rho = rr * np.ones(K) + .005 * PRNG.rand(K)
      omega = 5 * rr * np.ones(K)
      ro = np.hstack([rho, omega])
      f, g = OptimSB.objFunc_constrained(ro, approx_grad=0,
                                             **self.kwargs)
      ga = approx_fprime(ro, objFunc, epsvec)
      percDiff = np.max(np.abs(g - ga)/np.abs(g))
      print '-------------------------- perc diff %.4f' % (percDiff)
      print '  %s exact ' % (np2flatstr(g))
      print '  %s approx' % (np2flatstr(ga))
      assert percDiff < 0.03


  ######################################################### find optimum
  #########################################################
  def test__find_optimum(self):
    print ''
    roa, fa, Info = OptimSB.find_optimum(approx_grad=1, **self.kwargs)
    ro, f, Info = OptimSB.find_optimum(approx_grad=0, **self.kwargs)

    rhoa, omegaa, K = OptimSB._unpack(roa)
    rho, omega, K = OptimSB._unpack(ro)

    print Info['task']
    print '%.5e exact' % (f)
    print '%.5e approx' % (fa)

    print '  %s rho exact' % (np2flatstr(rho))
    print '  %s rho approx'  % (np2flatstr(rhoa))

    print '  %s omega exact' % (np2flatstr(omega))
    print '  %s omega approx'  % (np2flatstr(omegaa))
    
    #assert f <= fa
    assert np.allclose( rho, rhoa, atol=0.02)
    # don't even bother checking omega... 

  def test__find_optimum_multiple_tries(self):
    print ''
    rhoa, omegaa, fa, Info = OptimSB.find_optimum_multiple_tries(approx_grad=1, **self.kwargs)
    rho, omega, f, Info = OptimSB.find_optimum_multiple_tries(approx_grad=0, **self.kwargs)

    print Info['msg']
    print '%.5e exact' % (f)
    print '%.5e approx' % (fa)

    print '  %s rho exact' % (np2flatstr(rho))
    print '  %s rho approx'  % (np2flatstr(rhoa))

    print '  %s omega exact' % (np2flatstr(omega))
    print '  %s omega approx'  % (np2flatstr(omegaa))
    
    #assert f <= fa
    assert np.allclose( rho, rhoa, atol=0.02)
    # don't even bother checking omega... 

  ######################################################### Vd
  #########################################################
  def test__sampleVd(self):
    if self.kwargs['nDoc'] == 0:
      raise SkipTest
    print ''
    assert self.Vd.shape[0] == self.nDoc
    assert self.Vd.shape[1] == self.v.size
    assert self.Vd.ndim == 2
    meanVd = np.mean(self.Vd, axis=0)
    print '---------- v'
    print np2flatstr(self.v)
    print '---------- mean(Vd)'
    print np2flatstr(meanVd)
    absDiff = np.abs(meanVd - self.v)
    assert np.max(absDiff) < 0.08

    sumLogVd = self.kwargs['sumLogVd']
    assert sumLogVd.size == self.v.size

    sumLog1mVd = self.kwargs['sumLog1mVd']
    assert sumLog1mVd.size == self.v.size


class TestBasics_nDoc0_K1(TestBasics_nDoc0_K4):
  def shortDescription(self):
    return None

  def setUp(self):
    K = 1
    self.v = 0.5 * np.ones(K)
    self.kwargs = dict(sumLogVd=np.zeros(K), sumLog1mVd=np.zeros(K), nDoc=0, 
                       alpha=5.0, gamma=0.99)


class TestBasics_nDoc50_K1(TestBasics_nDoc0_K4):

  def setUp(self):
    self.v = np.asarray([0.98]) # test near the boundary!
    self.nDoc = 50
    self.Vd = sampleVd(self.v, nDoc=self.nDoc, gamma=0.99)
    sumLogVd, sumLog1mVd = summarizeVd(self.Vd)
    self.kwargs = dict(sumLogVd=sumLogVd, sumLog1mVd=sumLog1mVd, nDoc=self.nDoc, 
                       alpha=5.0, gamma=0.99)

class TestBasics_nDoc50_K4(TestBasics_nDoc0_K4):

  def setUp(self):
    self.v = np.asarray([0.1, 0.2, 0.3, 0.4])
    self.nDoc = 50
    self.Vd = sampleVd(self.v, nDoc=self.nDoc, gamma=0.99)
    sumLogVd, sumLog1mVd = summarizeVd(self.Vd)
    self.kwargs = dict(sumLogVd=sumLogVd, sumLog1mVd=sumLog1mVd, nDoc=self.nDoc, 
                       alpha=5.0, gamma=0.99)

class TestBasics_nDoc5000_K2(TestBasics_nDoc0_K4):

  def setUp(self):
    self.v = np.asarray([0.91, 0.3])
    self.nDoc = 5000
    self.Vd = sampleVd(self.v, nDoc=self.nDoc, gamma=0.99)
    sumLogVd, sumLog1mVd = summarizeVd(self.Vd)
    self.kwargs = dict(sumLogVd=sumLogVd, sumLog1mVd=sumLog1mVd, nDoc=self.nDoc, 
                       alpha=5.0, gamma=0.99)


class TestBasics_nDoc5000_K32(TestBasics_nDoc0_K4):

  def setUp(self):
    self.v = 0.42 * np.ones(32)
    self.nDoc = 5000
    self.Vd = sampleVd(self.v, nDoc=self.nDoc, gamma=0.99)
    sumLogVd, sumLog1mVd = summarizeVd(self.Vd)
    self.kwargs = dict(sumLogVd=sumLogVd, sumLog1mVd=sumLog1mVd, nDoc=self.nDoc, 
                       alpha=5.0, gamma=0.99)
"""
'''
class TestBasics_nDoc500(TestBasics_nDoc0):

  def setUp(self):
    self.v = np.asarray([0.1, 0.2, 0.3, 0.4])
    self.nDoc = 500
    self.Vd = sampleVd(self.v, nDoc=self.nDoc, gamma=0.99)
    sumLogVd, sumLog1mVd = summarizeVd(self.Vd)
    self.kwargs = dict(sumLogVd=sumLogVd, sumLog1mVd=sumLog1mVd, nDoc=self.nDoc, 
                       alpha=5.0, gamma=0.99)

class TestBasics_nDoc5000(TestBasics_nDoc0):

  def setUp(self):
    self.v = np.asarray([0.1, 0.2, 0.3, 0.4])
    self.nDoc = 5000
    self.Vd = sampleVd(self.v, nDoc=self.nDoc, gamma=0.99)
    sumLogVd, sumLog1mVd = summarizeVd(self.Vd)
    self.kwargs = dict(sumLogVd=sumLogVd, sumLog1mVd=sumLog1mVd, nDoc=self.nDoc, 
                       alpha=5.0, gamma=0.99)

'''

