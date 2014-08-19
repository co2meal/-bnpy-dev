import sys
import numpy as np
from scipy.optimize import approx_fprime
import warnings
import unittest

import bnpy.allocmodel.admix2.OptimizerHDPFast as OptimFast

from SampleVUtil import sampleVd, summarizeVdToDocTopicCount
from DebugUtil import np2flatstr, assert_allclose, printVectors

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
                      DocTopicCount=np.zeros(K))
            if approx_grad:
              f = OptimFast.objFunc_constrained(rhoomega, approx_grad=1,
                                                 **kwargs)
              g = np.ones(2*K)
            else:
              f, g = OptimFast.objFunc_constrained(rhoomega, approx_grad=0,
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
                      DocTopicCount=np.zeros(K))
            c = OptimFast.rhoomega2c(rhoomega)
            if approx_grad:
              f = OptimFast.objFunc_unconstrained(c, approx_grad=1,
                                                 **kwargs)
              g = np.ones(2*K)
            else:
              f, g = OptimFast.objFunc_unconstrained(c, approx_grad=0,
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
                      DocTopicCount=np.zeros(K))
          f, g = OptimFast.objFunc_constrained(rhoomega,
                                            approx_grad=0,
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
                      DocTopicCount=np.zeros(K))

            ## Exact gradient
            _, g = OptimFast.objFunc_constrained(rhoomega,
                                               approx_grad=0,
                                               **kwargs)

            ## Numerical gradient
            objFunc = lambda x: OptimFast.objFunc_constrained(x,
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
                      DocTopicCount=np.zeros(K))

            ro, f, Info = OptimFast.find_optimum(initrho=initrho,
                                               initomega=initomega,
                                               **kwargs)
            rho_est, omega_est, KK = OptimFast._unpack(ro)
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

            beta_est = OptimFast.rho2beta_active(rho_est)
            beta_opt = OptimFast.rho2beta_active(rho_opt)
            print '  beta_est', np2flatstr(beta_est, fmt='%9.6f')
            print '  beta_opt', np2flatstr(beta_opt, fmt='%9.6f')

            assert np.allclose(beta_est, beta_opt, atol=1e-4)

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
            DocTopicCount = summarizeVdToDocTopicCount(Vd)
            rho = PRNG.rand(K)
            omega = nDoc * PRNG.rand(K)
            for approx_grad in [0, 1]:
              rhoomega = np.hstack([rho, omega])
              kwargs = dict(alpha=0.5, 
                      gamma=1, 
                      nDoc=nDoc,
                      DocTopicCount=DocTopicCount)
              if approx_grad:
                f = OptimFast.objFunc_constrained(rhoomega, approx_grad=1,
                                                 **kwargs)
                g = np.ones(2*K)
                fapprox = f

              else:
                f, g = OptimFast.objFunc_constrained(rhoomega, approx_grad=0,
                                                 **kwargs)
                fexact = f
              assert type(f) == np.float64
              assert g.ndim == 1
              assert g.size == 2 * K            
              assert np.isfinite(f)
              assert np.all(np.isfinite(g))
            print fexact
            print fapprox
            print ''


  def testGradientExactAndApproxAgree__objFunc_constrained(self):
    ''' Verify computed gradient similar for exact and approx methods
    '''
    print ''
    for K in [1, 10, 55]:
      for alpha in [0.1, 0.95]:
        for gamma in [1., 9.45]:
          for nDoc in [1, 100, 1000]:

            print '================== K %d | nDoc %d | alpha %.2f' \
                  % (K, nDoc, alpha)

            for seed in [111, 222, 333]:
              PRNG = np.random.RandomState(seed)
              u = np.linspace(0.01, 0.99, K)
              Vd = sampleVd(u, nDoc, alpha, PRNG=PRNG)
              DocTopicCount = summarizeVdToDocTopicCount(Vd)

              rho = PRNG.rand(K)
              omega = 100 * PRNG.rand(K)
              rhoomega = np.hstack([rho, omega])
              kwargs = dict(alpha=alpha, 
                      gamma=gamma, 
                      nDoc=nDoc,
                      DocTopicCount=DocTopicCount)
              
              ## Calculate Exact gradient
              f, g = OptimFast.objFunc_constrained(rhoomega, approx_grad=0,
                                                 **kwargs)

              ## Calculate Approx gradient
              objFunc = lambda x: OptimFast.objFunc_constrained(x, approx_grad=1,
                                                               **kwargs)
              epsvec = np.hstack([1e-8*np.ones(K), 1e-8*np.ones(K)])
              gapprox = approx_fprime(rhoomega, objFunc, epsvec)    

              ## Verify rho gradient
              rtol_rho = 0.00001
              atol_rho = 0.001
              assert_allclose(g[:K], gapprox[:K], 'rho exact', 'rho approx', 
                              atol=atol_rho,
                              rtol=rtol_rho)

              ## Verify omega gradient
              # Note: small omega derivatives tend to cause lots of problems
              # so we should use high atol to avoid these issues
              rtol_omega = 0.05
              atol_omega = 0.01
              assert_allclose(g[K:], gapprox[K:], 'omega exact', 'omega approx', 
                              atol=atol_omega,
                              rtol=rtol_omega)



  def testRecoverRhoThatGeneratedData(self):
    ''' Verify that find_optimum's result is indistiguishable from analytic opt
    '''
    print ''
    gamma = 1.0
    for K in [10]: #, 10, 107]:
      for alpha in [0.9999]:
        for nDoc in [10000]:
          print '================== K %d | alpha %.2f | nDoc %d' \
                % (K, alpha, nDoc)

          for seed in [111, 222, 333]:

              PRNG = np.random.RandomState(seed)
              u_true = np.linspace(0.01, 0.99, K)
              Vd = sampleVd(u_true, nDoc, alpha, PRNG=PRNG)
              DocTopicCount = summarizeVdToDocTopicCount(Vd)

              initrho = PRNG.rand(K)
              initomega = 100 * PRNG.rand(K)
              scale = 1.0 #float(1+nDoc)/K
              kwargs = dict(alpha=alpha, 
                      gamma=gamma, 
                      nDoc=nDoc,                      
                      scaleVector=np.hstack([np.ones(K),
                                             float(scale) * np.ones(K)]),
                      DocTopicCount=DocTopicCount
                      )
              rho_est, omega_est, f_est, Info = \
                       OptimFast.find_optimum_multiple_tries(
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
                       OptimFast.find_optimum_multiple_tries(
                                                 initrho=rho_orig,
                                                 initomega=omega_orig,
                                                 **kwargs)

              f_orig, _ = OptimFast.objFunc_constrained(ro_orig, **kwargs)
              print '  f_orig %.7f' % (f_orig)
              print '  f_hot  %.7f' % (f_hot)
              print '  f_est  %.7f' % (f_est)

              print '  rho_orig', np2flatstr(rho_orig, fmt='%9.6f')
              print '  rho_hot ', np2flatstr(rho_hot, fmt='%9.6f')
              print '  rho_est ', np2flatstr(rho_est, fmt='%9.6f')

              assert f_hot <= f_orig
              assert np.allclose(f_est, f_hot, rtol=0.01)
              assert np.allclose(rho_est, rho_hot, atol=0.02, rtol=1e-5)
