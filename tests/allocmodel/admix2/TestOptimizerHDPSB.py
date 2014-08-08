import sys
import numpy as np
from nose.plugins.skip import Skip, SkipTest
from scipy.optimize import approx_fprime
import warnings
import unittest

import bnpy.allocmodel.admix2.OptimizerHDPSB as OptimSB

np.set_printoptions(precision=3, suppress=False, linewidth=140)
def np2flatstr(xvec, fmt='%9.3f', Kmax=10):
  return ' '.join( [fmt % (x) for x in xvec[:Kmax]])

def sampleVd(u, nDoc=100, alpha=0.5, PRNG=np.random.RandomState(0)):
  K = u.size
  cumprod1mu = np.ones(K)
  cumprod1mu[1:] *= np.cumprod(1 - u[:-1])

  Vd = np.zeros((nDoc, K))
  for k in xrange(K):
    Vd[:,k] = PRNG.beta( alpha * cumprod1mu[k] * u[k],
                         alpha * cumprod1mu[k] * (1. - u[k]),
                         size=nDoc)
    ## Warning: beta rand generator can fail when both params
    ## are very small (~1e-8). This will yield NaN values.
    ## To fix, we use fact that Beta(eps, eps) will always yield a 0 or 1.
    badIDs = np.flatnonzero(np.isnan(Vd[:,k]))
    if len(badIDs) > 0:
      p = np.asarray( [1. - u[k], u[k]] )
      Vd[badIDs, k] = PRNG.choice([0, 1], len(badIDs), p=p, replace=True)
  assert not np.any(np.isnan(Vd))
  assert np.all(np.isfinite(Vd))
  return Vd

def summarizeVd(Vd):
  with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=RuntimeWarning,
                               message='divide by zero')
    logVd = np.log(Vd)
    log1mVd = np.log(1-Vd)

  assert not np.any(np.isnan(logVd))
  logVd = replaceInfVals(logVd)
  log1mVd = replaceInfVals(log1mVd)
  return np.sum(logVd, axis=0), np.sum(log1mVd, axis=0)

def replaceInfVals( logX, replaceVal=-100):
  infmask = np.isinf(logX)
  logX[infmask] = replaceVal
  return logX

########################################################### Test0Docs
###########################################################

class Test0Docs(unittest.TestCase):
  def shortDescription(self):
    return None

  def testHasSaneOutput__objFunc_constrained(self):
    ''' Verify objective value and gradient vector have correct type and size
    '''
    for K in [1, 2, 10, 101]:
      for seed in [33, 77, 888]:
        for oScale in [1, 10]:
          for approx_grad in [0, 1]:
            PRNG = np.random.RandomState(seed)
            rho = PRNG.rand(K)
            omega = oScale * PRNG.rand(K)
            rhoomega = np.hstack([rho, omega])
            kwargs = dict(alpha=1, 
                      gamma=1, 
                      nDoc=0,
                      sumLogVd=np.zeros(K),
                      sumLog1mVd=np.zeros(K))
            if approx_grad:
              f = OptimSB.objFunc_constrained(rhoomega, approx_grad=1,
                                                 **kwargs)
              g = np.ones(2*K)
            else:
              f, g = OptimSB.objFunc_constrained(rhoomega, approx_grad=0,
                                                 **kwargs)
            assert type(f) == np.float64
            assert g.ndim == 1
            assert g.size == 2 * K            
            assert np.isfinite(f)
            assert np.all(np.isfinite(g))

  def testHasSaneOutput__objFunc_unconstrained(self):
    ''' Verify objective value and gradient vector have correct type and size
    '''
    for K in [1, 2, 10, 100]:
      for seed in [33, 77, 444]:
        for oScale in [1, 45]:
          for approx_grad in [0, 1]:
            PRNG = np.random.RandomState(seed)
            rho = PRNG.rand(K)
            omega = oScale * PRNG.rand(K)
            rhoomega = np.hstack([rho, omega])
            kwargs = dict(alpha=1, 
                      gamma=1,
                      nDoc=0,
                      sumLogVd=np.zeros(K),
                      sumLog1mVd=np.zeros(K))
            c = OptimSB.rhoomega2c(rhoomega)
            if approx_grad:
              f = OptimSB.objFunc_unconstrained(c, approx_grad=1,
                                                 **kwargs)
              g = np.ones(2*K)
            else:
              f, g = OptimSB.objFunc_unconstrained(c, approx_grad=0,
                                                 **kwargs)
            assert type(f) == np.float64
            assert g.ndim == 1
            assert g.size == 2 * K            
            assert np.isfinite(f)
            assert np.all(np.isfinite(g))

  def testGradientZeroAtOptimum__objFunc_constrained(self):
    ''' Verify computed gradient at optimum is indistinguishable from zero
    '''
    for K in [1, 10, 107]:
      for alpha in [0.1, 0.95]:
        for gamma in [1., 3.14, 9.45]:
          rho = 1. / (1. + gamma) * np.ones(K)
          omega = (1 + gamma) * np.ones(K)
          rhoomega = np.hstack([rho, omega])
          kwargs = dict(alpha=alpha, 
                      gamma=gamma, 
                      nDoc=0,
                      sumLogVd=np.zeros(K),
                      sumLog1mVd=np.zeros(K))
          f, g = OptimSB.objFunc_constrained(rhoomega,
                                            approx_grad=0,
                                            **kwargs)
          print '       rho  ', np2flatstr(rho[:K])
          print '  grad rho  ', np2flatstr(g[:K])
          assert np.allclose(g, np.zeros(2*K))

  def testGradientZeroAtOptimum__objFunc_unconstrained(self):
    ''' Verify computed gradient at optimum is indistinguishable from zero
    '''
    print ''
    for K in [1, 10, 107]:
      for alpha in [0.1, 0.95]:
        for gamma in [1., 3.14, 9.45]:
          rho = 1.0 / (1. + gamma) * np.ones(K)
          omega = (1 + gamma) * np.ones(K)
          rhoomega = np.hstack([rho, omega])
          kwargs = dict(alpha=alpha, 
                      gamma=gamma, 
                      scaleVector=np.hstack([np.ones(K), gamma*np.ones(K)]),
                      nDoc=0,
                      sumLogVd=np.zeros(K),
                      sumLog1mVd=np.zeros(K))
          c = OptimSB.rhoomega2c(rhoomega, scaleVector=kwargs['scaleVector'])
          f, g = OptimSB.objFunc_unconstrained(c, approx_grad=0,
                                            **kwargs)
          print '       rho  ', np2flatstr(rho[:K])
          print '  grad rho  ', np2flatstr(g[:K])
          assert np.allclose(g, np.zeros(2*K))

  def testGradientExactAndApproxAgree__objFunc_constrained(self):
    ''' Verify computed gradient similar for exact and approx methods
    '''
    print ''
    for K in [1, 10, 107]:
      for alpha in [0.1, 0.95]:
        for gamma in [1., 3.14, 9.45]:
          for seed in [111, 222, 333]:
            PRNG = np.random.RandomState(seed)
            rho = PRNG.rand(K)
            omega = 100 * PRNG.rand(K)
            rhoomega = np.hstack([rho, omega])
            kwargs = dict(alpha=alpha, 
                      gamma=gamma,
                      nDoc=0,
                      sumLogVd=np.zeros(K),
                      sumLog1mVd=np.zeros(K))

            ## Exact gradient
            _, g = OptimSB.objFunc_constrained(rhoomega,
                                               approx_grad=0,
                                               **kwargs)

            ## Numerical gradient
            objFunc = lambda x: OptimSB.objFunc_constrained(x,
                                                            approx_grad=1,
                                                            **kwargs)
            epsvec = np.hstack([1e-8*np.ones(K), 1e-8*np.ones(K)])
            gapprox = approx_fprime(rhoomega, objFunc, epsvec)    

            print '      rho 1:10 ', np2flatstr(rho)
            print '     grad 1:10 ', np2flatstr(g[:K], fmt='% .6e')
            print '     grad 1:10 ', np2flatstr(gapprox[:K], fmt='% .6e')
            if K > 10:
              print '    rho K-10:K ', np2flatstr(rho[-10:])
              print '   grad K-10:K ', np2flatstr(g[K-10:K], fmt='% .6e')
              print 'gapprox K-10:K ', np2flatstr(gapprox[K-10:K], fmt='% .6e')
            assert np.allclose(g[:K], gapprox[:K], atol=1e-6, rtol=0.01)

            print np2flatstr(g[K:])
            print np2flatstr(gapprox[K:])
            assert np.allclose(g[K:], gapprox[K:], atol=1e-4, rtol=0.05)

  def testRecoverAnalyticOptimum__find_optimum(self):
    ''' Verify that find_optimum's result is indistiguishable from analytic opt
    '''
    for K in [1, 10, 23, 61, 68, 100]:
      for alpha in [0.1, 0.95]:
        for gamma in [1.1, 3.141, 9.45, 21.1337]:
          print '================== K %d | gamma %.2f' % (K, gamma)

          for seed in [111, 222, 333]:
            PRNG = np.random.RandomState(seed)
            initrho = PRNG.rand(K)
            initomega = 100 * PRNG.rand(K)
            scaleVec = np.hstack([np.ones(K), gamma*np.ones(K)])
            kwargs = dict(alpha=alpha, 
                      gamma=gamma, 
                      scaleVector=scaleVec,
                      nDoc=0,
                      sumLogVd=np.zeros(K),
                      sumLog1mVd=np.zeros(K))
            ro, f, Info = OptimSB.find_optimum(initrho=initrho,
                                               initomega=initomega,
                                               **kwargs)
            rho_est, omega_est, KK = OptimSB._unpack(ro)
            assert np.all(np.isfinite(rho_est))
            assert np.all(np.isfinite(omega_est))
            assert np.isfinite(f)
            print Info['task']

            rho_opt = 1.0 / (1. + gamma) * np.ones(K)
            omega_opt = (1. + gamma) * np.ones(K)

            print '  rho_est', np2flatstr(rho_est, fmt='%9.6f')
            print '  rho_opt', np2flatstr(rho_opt, fmt='%9.6f')

            print '  omega_est', np2flatstr(omega_est, fmt='%9.6f')
            print '  omega_opt', np2flatstr(omega_opt, fmt='%9.6f')

            assert np.allclose(rho_est, rho_opt, atol=1e-5, rtol=1e-5)

            assert np.allclose(omega_est, omega_opt, atol=1e-5, rtol=0.01)


########################################################### Test with Many Docs
###########################################################
class TestManyDocs(unittest.TestCase):
  def shortDescription(self):
    return None

  def testHasSaneOutput__objFunc_constrained(self):
    ''' Verify objective value and gradient vector have correct type and size
    '''
    for K in [1, 2, 10, 101]:
      for seed in [33, 77, 888]:
        for alpha in [0.1, 0.9]:
          for nDoc in [1, 50, 5000]:
            PRNG = np.random.RandomState(seed)
            u = np.linspace(0.1, 0.9, K)
            Vd = sampleVd(u, nDoc, alpha, PRNG=PRNG)
            sumLogVd, sumLog1mVd = summarizeVd(Vd)

            for approx_grad in [0, 1]:
              rho = PRNG.rand(K)
              omega = nDoc * PRNG.rand(K)
              rhoomega = np.hstack([rho, omega])
              kwargs = dict(alpha=0.5, 
                      gamma=1, 
                      nDoc=nDoc,
                      sumLogVd=sumLogVd,
                      sumLog1mVd=sumLog1mVd)
              if approx_grad:
                f = OptimSB.objFunc_constrained(rhoomega, approx_grad=1,
                                                 **kwargs)
                g = np.ones(2*K)
              else:
                f, g = OptimSB.objFunc_constrained(rhoomega, approx_grad=0,
                                                 **kwargs)
              assert type(f) == np.float64
              assert g.ndim == 1
              assert g.size == 2 * K            
              assert np.isfinite(f)
              assert np.all(np.isfinite(g))


  def testGradientExactAndApproxAgree__objFunc_constrained(self):
    ''' Verify computed gradient similar for exact and approx methods
    '''
    for K in [1, 10, 107]:
      for alpha in [0.1, 0.95]:
        for gamma in [1., 3.14, 9.45]:
          for nDoc in [1, 100, 1000]:

            print '================== K %d | nDoc %d | alpha %.2f' \
                  % (K, nDoc, alpha)

            for seed in [111, 222, 333]:
              PRNG = np.random.RandomState(seed)
              u = np.linspace(0.01, 0.99, K)
              Vd = sampleVd(u, nDoc, alpha, PRNG=PRNG)
              sumLogVd, sumLog1mVd = summarizeVd(Vd)

              rho = PRNG.rand(K)
              omega = 100 * PRNG.rand(K)
              rhoomega = np.hstack([rho, omega])
              kwargs = dict(alpha=alpha, 
                      gamma=gamma, 
                      nDoc=nDoc,
                      sumLogVd=sumLogVd,
                      sumLog1mVd=sumLog1mVd)
              
              ## Exact gradient
              f, g = OptimSB.objFunc_constrained(rhoomega, approx_grad=0,
                                                 **kwargs)

              ## Approx gradient
              objFunc = lambda x: OptimSB.objFunc_constrained(x, approx_grad=1,
                                                               **kwargs)
              epsvec = np.hstack([1e-8*np.ones(K), 1e-8*np.ones(K)])
              gapprox = approx_fprime(rhoomega, objFunc, epsvec)    

              print '      rho 1:10 ', np2flatstr(rho)
              print '     grad 1:10 ', np2flatstr(g[:K], fmt='% .6e')
              print '     grad 1:10 ', np2flatstr(gapprox[:K], fmt='% .6e')
              if K > 10:
                print '    rho K-10:K ', np2flatstr(rho[-10:])
                print '   grad K-10:K ', np2flatstr(g[K-10:K], fmt='% .6e')
                print 'gapprox K-10:K ', np2flatstr(gapprox[K-10:K], fmt='% .6e')
              rtol_rho = 0.01
              atol_rho = 1e-6
              rtol_omega = 0.05
              atol_omega = 0.01
              ## Note: small omega derivatives tend to cause lots of problems
              ## so we should use high atol to avoid these issues
              # -0.000321  0.004443  0.009009  0.004381 -0.001312  exact
              #  0.001455  0.002910  0.007276  0.002910  0.000000  approx
              assert np.allclose(g[:K], gapprox[:K], atol=atol_rho,
                                                        rtol=rtol_rho)
              oGradOK = np.allclose(g[K:], gapprox[K:], atol=atol_omega,
                                                        rtol=rtol_omega)
              if not oGradOK:
                print 'VIOLATION DETECTED!'
                print 'grad_approx DOES NOT EQUAL grad_exact (within tolerance)'
                
                absDiff = np.abs(g[K:] - gapprox[K:])
                tolDiff = (atol_omega + rtol_omega * np.abs(gapprox[K:])) \
                            - absDiff
                worstIDs = np.argsort(tolDiff)
                print 'Top 5 worst mismatches'
                print np2flatstr( g[K + worstIDs[:5]], fmt='% .6f')
                print np2flatstr( gapprox[K + worstIDs[:5]], fmt='% .6f')
              assert oGradOK

  def testRecoverGlobalSticksFromGeneratedData(self):
    ''' Verify that mean of V_d matrix is equal to original vector u
    '''
    print ''
    gamma = 1.0
    for K in [1, 10, 107]:
      for alpha in [0.95, 0.5]:
        for nDoc in [10000]:
          print '================== K %d | alpha %.2f | nDoc %d' \
                % (K, alpha, nDoc)
          
          for seed in [111, 222, 333]:

              PRNG = np.random.RandomState(seed)
              u_true = np.linspace(0.01, 0.99, K)
              Vd = sampleVd(u_true, nDoc, alpha, PRNG=PRNG)

              assert Vd.shape[0] == nDoc
              assert Vd.shape[1] == K
              assert Vd.ndim == 2
              meanVd = np.mean(Vd, axis=0)
              print '    u   1:10 ', np2flatstr(u_true)
              print ' E[v_d] 1:10 ', np2flatstr(meanVd)
              if K > 10:
                print '    u   -10: ', np2flatstr(u_true[-10:])
                print ' E[v_d] -10: ', np2flatstr(meanVd[-10:])
              assert np.allclose(u_true, meanVd, atol=0.02)

  def testRecoverRhoThatGeneratedData__find_optimum(self):
    ''' Verify that find_optimum's result is indistiguishable from analytic opt
    '''
    print ''
    gamma = 1.0
    for K in [93, 107, 85]: #, 10, 107]:
      for alpha in [0.9999]:
        for nDoc in [10000]:
          print '================== K %d | alpha %.2f | nDoc %d' \
                % (K, alpha, nDoc)

          for seed in [111, 222, 333]:

              PRNG = np.random.RandomState(seed)
              u_true = np.linspace(0.01, 0.99, K)
              Vd = sampleVd(u_true, nDoc, alpha, PRNG=PRNG)
              sumLogVd, sumLog1mVd = summarizeVd(Vd)

              initrho = PRNG.rand(K)
              initomega = 100 * PRNG.rand(K)
              scale = 1.0 #float(1+nDoc)/K
              kwargs = dict(alpha=alpha, 
                      gamma=gamma, 
                      nDoc=nDoc,                      
                      scaleVector=np.hstack([np.ones(K),
                                             float(scale) * np.ones(K)]),
                      sumLogVd=sumLogVd,
                      sumLog1mVd=sumLog1mVd,
                      )
              rho_est, omega_est, f_est, Info = \
                       OptimSB.find_optimum_multiple_tries(
                                                 initrho=initrho,
                                                 initomega=initomega,
                                                 **kwargs)
              assert np.all(np.isfinite(rho_est))
              assert np.all(np.isfinite(omega_est))
              assert np.isfinite(f_est)
              print Info['msg']

              rho_orig = u_true
              omega_orig = (1 + gamma) * np.ones(K)
              ro_orig = np.hstack([rho_orig, omega_orig])
              rho_hot, omega_hot, f_hot, Ihot = \
                       OptimSB.find_optimum_multiple_tries(
                                                 initrho=rho_orig,
                                                 initomega=omega_orig,
                                                 **kwargs)

              f_orig, _ = OptimSB.objFunc_constrained(ro_orig, **kwargs)
              print '  f_orig %.7f' % (f_orig)
              print '  f_hot  %.7f' % (f_hot)
              print '  f_est  %.7f' % (f_est)


              print '  rho_orig', np2flatstr(rho_orig, fmt='%9.6f')
              print '  rho_hot ', np2flatstr(rho_hot, fmt='%9.6f')
              print '  rho_est ', np2flatstr(rho_est, fmt='%9.6f')

              assert f_hot <= f_orig
              assert np.allclose(f_est, f_hot, rtol=0.01)
              assert np.allclose(rho_est, rho_hot, atol=0.02, rtol=1e-5)

            
"""
########################################################### TestBasics
###########################################################
class TestBasics_nDoc0_K4(unittest.TestCase):
  def shortDescription(self):
    return None

  def setUp(self):
    self.u = np.asarray( [.5, .5, .5, .5])
    self.kwargs = dict(sumLogVd=np.zeros(4), sumLog1mVd=np.zeros(4), nDoc=0, 
                       gamma=5.0, alpha=0.99)

  ######################################################### objFunc basics
  #########################################################

  def test__objFunc_constrained(self):
    f1, g1 = self.verify__objFunc_constrained(approx_grad=True)
    f2, g2 = self.verify__objFunc_constrained(approx_grad=False)
    assert np.allclose(f1, f2)
    assert g1 is None
    assert g2 is not None
    assert g2.size == 2 * self.u.size
    assert not np.any(np.isinf(g2))

  def test__objFunc_unconstrained(self):
    f1, g1 = self.verify__objFunc_unconstrained(approx_grad=True)
    f2, g2 = self.verify__objFunc_unconstrained(approx_grad=False)
    assert np.allclose(f1, f2)
    assert g1 is None
    assert g2 is not None
    assert g2.size == 2 * self.u.size
    assert not np.any(np.isinf(g2))

  def verify__objFunc_constrained(self, approx_grad=False):
    ''' Verify *constrained* objFunc return value has correct shape, domain
    '''
    print ''
    g = None
    K = self.u.size
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
    K = self.u.size
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
  def test__known_optimum__nDoc0(self):
    ''' Verify that analytic optimum for rho,omega actually maximizes objFunc
    '''
    if self.kwargs['nDoc'] > 0:
      raise SkipTest
    K = self.kwargs['sumLogVd'].size
    rhoopt = 1.0 / (1.0 + self.kwargs['gamma']) * np.ones(K)
    omegaopt = (1.0 + self.kwargs['gamma']) * np.ones(K)
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
        print '    rho   ', np2flatstr(ro[:K])
        print '    omega ', np2flatstr(ro[K:])
        assert fopt < f

  def test__gradient_at_known_optimum(self):
    ''' Verify gradient at analytic optimum is very near zero.
    '''
    if self.kwargs['nDoc'] > 0:
      raise SkipTest
    K = self.u.size
    rhoopt = 1.0 / (1.0 + self.kwargs['gamma']) * np.ones(K)
    omegaopt = (1.0 + self.kwargs['gamma']) * np.ones(K)
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
    K = self.u.size

    objFunc = lambda r : OptimSB.objFunc_constrained(r, approx_grad=1,
                                             **self.kwargs)

    ro, f, Info = OptimSB.find_optimum(approx_grad=1, **self.kwargs)
    f, g = OptimSB.objFunc_constrained(ro, approx_grad=0,
                                             **self.kwargs)
    assert not np.any(np.isnan(g))

    epsvec = np.hstack([1e-10*np.ones(K), 1e-8*np.ones(K)])
    ga = approx_fprime(ro, objFunc, epsvec)    

    absDiff = np.abs(g - ga)
    denom = np.abs(g)
    percDiff = np.abs(g - ga)/denom
    if np.sum(denom < 0.01) > 0:
      percDiff[denom < 0.01] = absDiff[denom < 0.01]
    print '-------------------------- Gradients AT OPT ****'
    print '  %s exact ' % (np2flatstr(g))
    print '  %s approx' % (np2flatstr(ga))

    roguess = np.hstack([self.u, 1+self.kwargs['nDoc']*np.ones(K)])
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
    print '%.5e exact ' % (f)
    print '%.5e approx' % (fa)

    print '  %s rho exact ' % (np2flatstr(rho))
    print '  %s rho approx'  % (np2flatstr(rhoa))

    print '  %s omega exact ' % (np2flatstr(omega))
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
    assert self.Vd.shape[1] == self.u.size
    assert self.Vd.ndim == 2
    meanVd = np.mean(self.Vd, axis=0)
    print '---------- v'
    print np2flatstr(self.u)
    print '---------- mean(Vd)'
    print np2flatstr(meanVd)
    absDiff = np.abs(meanVd - self.u)
    assert np.max(absDiff) < 0.08

    sumLogVd = self.kwargs['sumLogVd']
    assert sumLogVd.size == self.u.size

    sumLog1mVd = self.kwargs['sumLog1mVd']
    assert sumLog1mVd.size == self.u.size


class TestBasics_nDoc0_K1(TestBasics_nDoc0_K4):
  def shortDescription(self):
    return None

  def setUp(self):
    K = 1
    self.u = 0.5 * np.ones(K)
    self.kwargs = dict(sumLogVd=np.zeros(K), sumLog1mVd=np.zeros(K), nDoc=0, 
                       gamma=5.0, alpha=0.99)


class TestBasics_nDoc50_K1(TestBasics_nDoc0_K4):

  def setUp(self):
    self.u = np.asarray([0.98]) # test near the boundary!
    self.nDoc = 50
    self.Vd = sampleVd(self.u, nDoc=self.nDoc, alpha=0.99)
    sumLogVd, sumLog1mVd = summarizeVd(self.Vd)
    self.kwargs = dict(sumLogVd=sumLogVd, sumLog1mVd=sumLog1mVd, nDoc=self.nDoc, 
                       gamma=5.0, alpha=0.99)

class TestBasics_nDoc50_K4(TestBasics_nDoc0_K4):

  def setUp(self):
    self.u = np.asarray([0.1, 0.2, 0.3, 0.4])
    self.nDoc = 50
    self.Vd = sampleVd(self.u, nDoc=self.nDoc, alpha=0.99)
    sumLogVd, sumLog1mVd = summarizeVd(self.Vd)
    self.kwargs = dict(sumLogVd=sumLogVd, sumLog1mVd=sumLog1mVd, nDoc=self.nDoc, 
                       gamma=5.0, alpha=0.99)

class TestBasics_nDoc5000_K2(TestBasics_nDoc0_K4):

  def setUp(self):
    self.u = np.asarray([0.91, 0.3])
    self.nDoc = 5000
    self.Vd = sampleVd(self.u, nDoc=self.nDoc, alpha=0.99)
    sumLogVd, sumLog1mVd = summarizeVd(self.Vd)
    self.kwargs = dict(sumLogVd=sumLogVd, sumLog1mVd=sumLog1mVd, nDoc=self.nDoc, 
                       gamma=5.0, alpha=0.99)


class TestBasics_nDoc5000_K32(TestBasics_nDoc0_K4):

  def setUp(self):
    self.u = 0.42 * np.ones(32)
    self.nDoc = 5000
    self.Vd = sampleVd(self.u, nDoc=self.nDoc, alpha=0.99)
    sumLogVd, sumLog1mVd = summarizeVd(self.Vd)
    self.kwargs = dict(sumLogVd=sumLogVd, sumLog1mVd=sumLog1mVd, nDoc=self.nDoc, 
                       gamma=5.0, alpha=0.99)
'''
class TestBasics_nDoc500(TestBasics_nDoc0):

  def setUp(self):
    self.u = np.asarray([0.1, 0.2, 0.3, 0.4])
    self.nDoc = 500
    self.Vd = sampleVd(self.u, nDoc=self.nDoc, alpha=0.99)
    sumLogVd, sumLog1mVd = summarizeVd(self.Vd)
    self.kwargs = dict(sumLogVd=sumLogVd, sumLog1mVd=sumLog1mVd, nDoc=self.nDoc, 
                       gamma=5.0, alpha=0.99)

class TestBasics_nDoc5000(TestBasics_nDoc0):

  def setUp(self):
    self.u = np.asarray([0.1, 0.2, 0.3, 0.4])
    self.nDoc = 5000
    self.Vd = sampleVd(self.u, nDoc=self.nDoc, alpha=0.99)
    sumLogVd, sumLog1mVd = summarizeVd(self.Vd)
    self.kwargs = dict(sumLogVd=sumLogVd, sumLog1mVd=sumLog1mVd, nDoc=self.nDoc, 
                       gamma=5.0, alpha=0.99)

'''
"""