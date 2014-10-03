import sys
import numpy as np
from nose.plugins.skip import Skip, SkipTest
from scipy.optimize import approx_fprime
import warnings
import unittest

import bnpy.allocmodel.admix.OptimizerForHDPStickBreak as OptimSB

np.set_printoptions(precision=3, suppress=False, linewidth=140)
def np2flatstr(xvec, Kmax=10):
  return ' '.join( ['%9.3f' % (x) for x in xvec[:Kmax]])

def sampleVd(v, nDoc=100, gamma=0.5):
  K = v.size
  PRNG = np.random.RandomState(0)
  cumprod1mv = np.ones(K)
  cumprod1mv[1:] = np.cumprod(1 - v[:-1])

  Vd = np.zeros((nDoc, K))
  for k in xrange(K):
    # Warning: beta rand generator can fail when params are very small (~1e-8)
    Vd[:,k] = PRNG.beta( gamma * cumprod1mv[k] * v[k],
                         gamma * cumprod1mv[k] * (1-v[k]),
                         size=nDoc)
    if np.any(np.isnan(Vd[:,k])):
      Vd[:,k] = PRNG.choice([0, 1], nDoc, replace=True)
  assert not np.any(np.isnan(Vd))
  return Vd

def summarizeVd(Vd):
  with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=RuntimeWarning,
                               message='divide by zero')
    logVd = np.log(Vd)
    log1mVd = np.log(1-Vd)

  assert not np.any(np.isnan(logVd))
  infmask = np.isinf(logVd)
  if np.any(infmask):
    print 'REPLACING %d values in logVd' % (np.sum(infmask))
    logVd[infmask] = -40
  infmask = np.isinf(log1mVd)
  if np.any(infmask):
    print 'REPLACING %d values in log1mVd' % (np.sum(infmask))
    log1mVd[infmask] = -40
  return np.sum(logVd, axis=0), np.sum(log1mVd, axis=0)

########################################################### TestBasics
###########################################################
class TestBasics_nDoc0_K4(unittest.TestCase):
  def shortDescription(self):
    return None

  def setUp(self):
    self.v = np.asarray( [.5, .5, .5, .5])
    self.kwargs = dict(sumLogVd=np.zeros(4), sumLog1mVd=np.zeros(4), nDoc=0, 
                       alpha=5.0, gamma=0.99)

  ######################################################### objFunc basics
  #########################################################

  def test__objFunc_constrained(self):
    f1, g1 = self.verify__objFunc_constrained(approx_grad=True)
    f2, g2 = self.verify__objFunc_constrained(approx_grad=False)
    assert np.allclose(f1, f2)
    assert g1 is None
    assert g2 is not None
    assert g2.size == 2 * self.v.size
    assert not np.any(np.isinf(g2))

  def test__objFunc_unconstrained(self):
    f1, g1 = self.verify__objFunc_unconstrained(approx_grad=True)
    f2, g2 = self.verify__objFunc_unconstrained(approx_grad=False)
    assert np.allclose(f1, f2)
    assert g1 is None
    assert g2 is not None
    assert g2.size == 2 * self.v.size
    assert not np.any(np.isinf(g2))

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