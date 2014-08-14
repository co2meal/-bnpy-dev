import sys
import numpy as np
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

class TestBounds(unittest.TestCase):
  def shortDescription(self):
    return None

  def test_rho2beta_simple(self):
    ''' Verify that conversion back and forth succeeds
    ''' 
    print ''
    EPS = 0.00001
    ## Make a "bad" initrho (where beta for last active topic will be zero!)
    initrho = np.asarray([0.5, 0.5, 1.0, EPS/100])
    initbeta = OptimSB.rho2beta_active(initrho)

    ## Verify this does bad things
    initrho2 = OptimSB.beta2rho(initbeta, initrho.size)
    assert not np.allclose( initrho, initrho2, atol=EPS)

    ## Now force in bounds, and verify it does good things
    rho = OptimSB.forceRhoInBounds(initrho, EPS)
    rho2 = OptimSB.beta2rho(OptimSB.rho2beta_active(rho), rho.size)
    assert np.allclose(rho, rho2, atol=EPS)

    beta = OptimSB.rho2beta_active(rho)
    betaRem = 1 - np.sum(beta)
    print ' final beta', np2flatstr(beta, fmt='%.6f')
    print '  init beta', np2flatstr(initbeta, fmt='%.6f')
    assert np.allclose(initbeta, beta, atol=EPS)


  def test_rho2beta_Kmany(self, EPS=1e-8):
    ''' Verify that conversion back and forth succeeds
    ''' 
    print ''
    for K in [50, 100, 500]:
      print '=================== K %d' % (K)
      for seed in [11, 22, 33, 44, 55]:
        PRNG = np.random.RandomState(seed)
        initrho = PRNG.rand(K)
        assert initrho.min() >= EPS

        initbeta = OptimSB.rho2beta_active(initrho)
        assert np.sum(initbeta) <= 1.0

        rho = OptimSB.forceRhoInBounds(initrho, EPS)
        assert rho.min() >= EPS
        assert rho.max() <= 1-EPS

        beta = OptimSB.rho2beta_active(rho)        
        print beta.min()
        print beta.max()
        assert beta.min() >= EPS
        assert beta.max() <= 1-EPS

        rho2 = OptimSB.beta2rho(beta, rho.size)
        assert rho2.min() >= EPS
        assert rho2.max() <= 1-EPS

        beta2 = OptimSB.rho2beta_active(rho2)
        assert np.sum(beta2) <= 1.0
        assert np.min(beta2) >= EPS

        print ' final beta', np2flatstr(beta2, fmt='%.6f')
        print '  init beta', np2flatstr(initbeta, fmt='%.6f')


        print ' final beta', np2flatstr(beta2[-10:], fmt='%.6f')
        print '  init beta', np2flatstr(initbeta[-10:], fmt='%.6f')

        assert np.allclose(initbeta, beta2, atol=1.05*K*EPS)


  def testForceRhoInBounds(self):
    ''' Verify forceRhoInBounds works as expected
    '''
    print ''
    EPS = 0.00001
    initrho = np.asarray([0.5, 0.5, 1.0, EPS/100])
    rho = OptimSB.forceRhoInBounds(initrho, EPS)
    assert np.all(rho < 1-EPS)
    assert np.all(rho > EPS)
    print rho

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

            beta_est = OptimSB.rho2beta_active(rho_est)
            beta_opt = OptimSB.rho2beta_active(rho_opt)
            print '  beta_est', np2flatstr(beta_est, fmt='%9.6f')
            print '  beta_opt', np2flatstr(beta_opt, fmt='%9.6f')
            assert np.allclose(beta_est, beta_opt, atol=1e-4)
            assert np.allclose(omega_est, omega_opt, atol=1e-5, rtol=0.01)

## Note: we don't care *so* so much about whether the rho values are spot on
## Rather, we care about how well the beta vectors match up,
##  since that is what really matters


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
