import sys
import numpy as np
from scipy.optimize import approx_fprime
import warnings
import unittest

from bnpy.allocmodel.topics import OptimizerRhoOmega
from SampleVUtil import sampleVd, summarizeVdToPi

np.set_printoptions(precision=3, suppress=False, linewidth=140)
def np2flatstr(xvec, fmt='%9.3f', Kmax=10):
  return ' '.join( [fmt % (x) for x in xvec[:Kmax]])

########################################################### Test0Docs
###########################################################

class Test0Docs(unittest.TestCase):
  def shortDescription(self):
    return None

  def testHasSaneOutput__objFunc_constrained(self):
    ''' Verify objective value and gradient vector have correct type and size

        f should be a finite scalar
        g should be a vector of size 2K (first half is drho, second is domega)
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
                      sumLogPi=np.zeros(K+1))
            if approx_grad:
              f = OptimizerRhoOmega.objFunc_constrained(
                                    rhoomega, approx_grad=1, **kwargs)
              g = np.ones(2*K) # placeholder, this would be done automatically
            else:
              f, g = OptimizerRhoOmega.objFunc_constrained(
                                    rhoomega, approx_grad=0, **kwargs)
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
                      sumLogPi=np.zeros(K+1))
            c = OptimizerRhoOmega.rhoomega2c(rhoomega)
            if approx_grad:
              f = OptimizerRhoOmega.objFunc_unconstrained(
                                    c, approx_grad=1, **kwargs)
              g = np.ones(2*K)
            else:
              f, g = OptimizerRhoOmega.objFunc_unconstrained(
                                    c, approx_grad=0, **kwargs)
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
          for kappa in [0, 100]:
            rho = 1. / (1. + gamma) * np.ones(K)
            omega = (1 + gamma) * np.ones(K)
            rhoomega = np.hstack([rho, omega])
            
            kwargs = dict(alpha=alpha, 
                          gamma=gamma,
                          kappa=kappa,
                          nDoc=0,
                          sumLogPi=np.zeros(K+1))
            f, g = OptimizerRhoOmega.objFunc_constrained(rhoomega,
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
                      sumLogPi=np.zeros(K+1))
          c = OptimizerRhoOmega.rhoomega2c(rhoomega, scaleVector=kwargs['scaleVector'])
          f, g = OptimizerRhoOmega.objFunc_unconstrained(c, approx_grad=0,
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
                      sumLogPi=np.zeros(K+1))

            ## Exact gradient
            _, g = OptimizerRhoOmega.objFunc_constrained(
                                    rhoomega, approx_grad=0, **kwargs)
            ## Numerical gradient
            objFunc = lambda x: OptimizerRhoOmega.objFunc_constrained(
                                    x, approx_grad=1, **kwargs)
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
          for kappa in [0, 100]:
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
                            sumLogPi=np.zeros(K+1))
              ro, f, Info = OptimizerRhoOmega.find_optimum(
                initrho=initrho,
                initomega=initomega,
                **kwargs)
              rho_est, omega_est, KK = OptimizerRhoOmega._unpack(ro)
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

              beta_est = OptimizerRhoOmega.rho2beta_active(rho_est)
              beta_opt = OptimizerRhoOmega.rho2beta_active(rho_opt)
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
            sumLogPi = summarizeVdToPi(Vd)
            rho = PRNG.rand(K)
            omega = nDoc * PRNG.rand(K)
            for approx_grad in [0, 1]:
              rhoomega = np.hstack([rho, omega])
              kwargs = dict(alpha=0.5, 
                      gamma=1, 
                      nDoc=nDoc,
                      sumLogPi=sumLogPi)
              if approx_grad:
                f = OptimizerRhoOmega.objFunc_constrained(
                                    rhoomega, approx_grad=1, **kwargs)
                g = np.ones(2*K)
                fapprox = f

              else:
                f, g = OptimizerRhoOmega.objFunc_constrained(
                                    rhoomega, approx_grad=0, **kwargs)
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
    for K in [1, 10, 107]:
      for alpha in [0.1, 0.95]:
        for gamma in [1., 9.45]:
          for nDoc in [1, 100, 1000]:

            print '================== K %d | nDoc %d | alpha %.2f' \
                  % (K, nDoc, alpha)

            for seed in [111, 222, 333]:
              PRNG = np.random.RandomState(seed)
              u = np.linspace(0.01, 0.99, K)
              Vd = sampleVd(u, nDoc, alpha, PRNG=PRNG)
              sumLogPi = summarizeVdToPi(Vd)

              rho = PRNG.rand(K)
              omega = 100 * PRNG.rand(K)
              rhoomega = np.hstack([rho, omega])
              kwargs = dict(alpha=alpha, 
                      gamma=gamma, 
                      nDoc=nDoc,
                      sumLogPi=sumLogPi)
              
              ## Exact gradient
              f, g = OptimizerRhoOmega.objFunc_constrained(
                                    rhoomega, approx_grad=0, **kwargs)

              ## Approx gradient
              objFunc = lambda x: OptimizerRhoOmega.objFunc_constrained(
                                    x, approx_grad=1, **kwargs)
              epsvec = np.hstack([1e-8*np.ones(K), 1e-8*np.ones(K)])
              gapprox = approx_fprime(rhoomega, objFunc, epsvec)    

              print '      rho 1:10 ', np2flatstr(rho)
              print '     grad 1:10 ', np2flatstr(g[:K], fmt='% .6e')
              print '     grad 1:10 ', np2flatstr(gapprox[:K], fmt='% .6e')
              if K > 10:
                print '    rho K-10:K ', np2flatstr(rho[-10:])
                print '   grad K-10:K ', np2flatstr(g[K-10:K], fmt='% .6e')
                print 'gapprox K-10:K ', np2flatstr(gapprox[K-10:K],fmt='% .6e')
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
              sumLogPi = summarizeVdToPi(Vd)

              initrho = PRNG.rand(K)
              initomega = 100 * PRNG.rand(K)
              scale = 1.0 #float(1+nDoc)/K
              kwargs = dict(alpha=alpha, 
                      gamma=gamma, 
                      nDoc=nDoc,                      
                      scaleVector=np.hstack([np.ones(K),
                                             float(scale) * np.ones(K)]),
                      sumLogPi=sumLogPi
                      )
              rho_est, omega_est, f_est, Info = \
                       OptimizerRhoOmega.find_optimum_multiple_tries(
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
                       OptimizerRhoOmega.find_optimum_multiple_tries(
                                                 initrho=rho_orig,
                                                 initomega=omega_orig,
                                                 **kwargs)

              f_orig, _ = OptimizerRhoOmega.objFunc_constrained(
                                                 ro_orig, **kwargs)
              print '  f_orig %.7f' % (f_orig)
              print '  f_hot  %.7f' % (f_hot)
              print '  f_est  %.7f' % (f_est)


              print '  rho_orig', np2flatstr(rho_orig, fmt='%9.6f')
              print '  rho_hot ', np2flatstr(rho_hot, fmt='%9.6f')
              print '  rho_est ', np2flatstr(rho_est, fmt='%9.6f')

              assert f_hot <= f_orig
              assert np.allclose(f_est, f_hot, rtol=0.01)
              assert np.allclose(rho_est, rho_hot, atol=0.02, rtol=1e-5)
