'''
Test for HDP with tight-local, PE-global variational approximation.

Gamma = 1.0 corresponds to uniform prior over interval rho [0, 1]
'''


import sys
import numpy as np
from scipy.optimize import approx_fprime
import warnings
import unittest

import bnpy.allocmodel.admix2.OptimizerHDPTightPE as Optim

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
      for gamma in [1.0, 5.0, 15.553]:
        for seed in [33, 77, 888]:
          print '==================  K %5d | gamma %.2f' % (K, gamma)
          for approx_grad in [0, 1]:
            PRNG = np.random.RandomState(seed)
            rho = PRNG.rand(K)
            kwargs = dict(alpha=1, 
                      gamma=gamma, 
                      nDoc=0,
                      DocTopicCount=np.zeros((1,K)),
                      )
            if approx_grad:
              f = Optim.objFunc_constrained(rho, approx_grad=1,
                                                 **kwargs)
              g = np.ones(K)
            else:
              f, g = Optim.objFunc_constrained(rho, approx_grad=0,
                                                 **kwargs)

            print ' f= ', f
            assert type(f) == np.float64
            assert g.ndim == 1
            assert g.size == K            
            assert np.isfinite(f)
            assert np.all(np.isfinite(g))


  def testHasSaneOutput__objFunc_unconstrained(self):
    ''' Verify objective value and gradient vector have correct type and size
    '''
    for K in [1, 2, 10, 100]:
      for gamma in [1.0, 5.0, 15.553]:
        for seed in [33, 77, 444]:
          print '==================  K %5d | gamma %.2f' % (K, gamma)

          for approx_grad in [0, 1]:
            PRNG = np.random.RandomState(seed)
            rho = PRNG.rand(K)
            kwargs = dict(alpha=1, 
                      gamma=gamma,
                      nDoc=0,
                      DocTopicCount=np.zeros((1,K)),
                      )

            c = Optim.rho2c(rho)
            if approx_grad:
              f = Optim.objFunc_unconstrained(rho, approx_grad=1,
                                                 **kwargs)
              g = np.ones(K)
            else:
              f, g = Optim.objFunc_unconstrained(rho, approx_grad=0,
                                                 **kwargs)
            print ' f= ', f
            assert type(f) == np.float64
            assert g.ndim == 1
            assert g.size == K            
            assert np.isfinite(f)
            assert np.all(np.isfinite(g))

  def testGradientZeroAtOptimum__objFunc_constrained(self):
    ''' Verify computed gradient at optimum is indistinguishable from zero

        Note: when gamma > 1, actually the optimum occurs at boundary rho=0,
              and the gradient will NOT be zero.
    '''
    print ''
    for K in [1, 10, 107]:
      for alpha in [0.1, 0.95]:
        for gamma in [1., 3.14, 9.45]:
          print '==================  K %5d | gamma %.2f' % (K, gamma)

          rho = (1 - 1.0) / (1 + gamma - 2.0 + 1e-14) * np.ones(K)
          kwargs = dict(alpha=alpha, 
                      gamma=gamma, 
                      nDoc=0,
                      DocTopicCount=np.zeros((1,K)),
                      )
          _, g = Optim.objFunc_constrained(rho,
                                             approx_grad=0,
                                             **kwargs)

          ## Numerical gradient
          objFunc = lambda x: Optim.objFunc_constrained(x,
                                                          approx_grad=1,
                                                          **kwargs)
          epsvec = 1e-8*np.ones(K)
          gapprox = approx_fprime(rho, objFunc, epsvec)    

          print '       rho  ', np2flatstr(rho)
          print '  grad rho  ', np2flatstr(g)
          print '  grad approx', np2flatstr(gapprox)

          if gamma == 1.0:
            assert np.allclose(g, np.zeros(K))
          else:
            ## optimum occurs at boundary, so gradient will not be zero!
            assert np.all( g > 0)


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
            kwargs = dict(alpha=alpha, 
                      gamma=gamma,
                      nDoc=0,
                      DocTopicCount=np.zeros((1,K)),
                      )

            ## Exact gradient
            _, g = Optim.objFunc_constrained(rho,
                                               approx_grad=0,
                                               **kwargs)

            ## Numerical gradient
            objFunc = lambda x: Optim.objFunc_constrained(x,
                                                          approx_grad=1,
                                                          **kwargs)
            epsvec = 1e-8 * np.ones(K)
            gapprox = approx_fprime(rho, objFunc, epsvec)    

            rtol_rho = 0.002
            atol_rho = 0.001
            assert_allclose(g, gapprox,
                            'rho grad exact', 'rho grad approx',
                            rtol=rtol_rho, atol=atol_rho)



########################################################### Test with Many Docs
###########################################################
class TestManyDocs(unittest.TestCase):
  def shortDescription(self):
    return None


  def testHasSaneOutput__objFunc_constrained(self):
    ''' Verify objective value and gradient vector have correct type and size
    '''
    print ''
    for K in [101, 315]:
      for alpha in [0.1, 0.9]:
        for nDoc in [1, 5000]:
          print '==================  K %5d | alpha %.2f' % (K, alpha)
          for seed in [33, 77, 88, 99, 111, 222]:

            PRNG = np.random.RandomState(seed)
            u = np.linspace(0.1, 0.9, K)
            Vd = sampleVd(u, nDoc, alpha, PRNG=PRNG)
            DocTopicCount = summarizeVdToDocTopicCount(Vd)
            rho = Optim.forceRhoInBounds( PRNG.rand(K))
            for approx_grad in [0, 1]:
              kwargs = dict(alpha=alpha, 
                      gamma=1, 
                      nDoc=nDoc,
                      DocTopicCount=DocTopicCount)

              if approx_grad:
                f = Optim.objFunc_constrained(rho, approx_grad=1,
                                                 **kwargs)
                g = np.ones(K)
              else:
                f, g = Optim.objFunc_constrained(rho, approx_grad=0,
                                                 **kwargs)
              print f
              assert type(f) == np.float64
              assert g.ndim == 1
              assert g.size == K            
              assert np.isfinite(f)
              assert np.all(np.isfinite(g))


  def testGradientExactAndApproxAgree__objFunc_constrained(self):
    ''' Verify computed gradient similar for exact and approx methods
    '''
    print ''
    for K in [3, 10, 55, 111]:
      for alpha in [0.1, 0.95]:
        for gamma in [1., 3.14, 9.45]:
          for nDoc in [1, 100, 1000]:

            print '================== K %d | nDoc %d | alpha %.2f' \
                  % (K, nDoc, alpha)

            for seed in [111, 222, 333]:
              PRNG = np.random.RandomState(seed)
              u = np.linspace(0.01, 0.95, K)
              Vd = sampleVd(u, nDoc, alpha, PRNG=PRNG)
              DocTopicCount = summarizeVdToDocTopicCount(Vd)

              rho = Optim.forceRhoInBounds(0.5 * u)
              kwargs = dict(alpha=alpha, 
                      gamma=gamma, 
                      nDoc=nDoc,
                      DocTopicCount=DocTopicCount)
              
              ## Exact gradient
              f, g = Optim.objFunc_constrained(rho, approx_grad=0,
                                                 **kwargs)

              ## Approx gradient
              objFunc = lambda x: Optim.objFunc_constrained(x, approx_grad=1,
                                                               **kwargs)
              epsvec = 1e-8 * np.ones(K)
              gapprox = approx_fprime(rho, objFunc, epsvec)    

              rtol_rho = 0.002
              atol_rho = 0.001
              assert_allclose(g, gapprox,
                              'rho grad exact', 'rho grad approx',
                              rtol=rtol_rho, atol=atol_rho)


  def testRecoverRhoThatGeneratedData__find_optimum(self):
    ''' Verify that find_optimum's result is indistiguishable from analytic opt
    '''
    print ''
    gamma = 1.0
    for K in [10, 27]:
      for alpha in [0.9995]:
        for nDoc in [2000]:
          print '================== K %d | alpha %.2f | nDoc %d' \
                % (K, alpha, nDoc)

          for seed in [111, 222, 333]:

              PRNG = np.random.RandomState(seed)
              u_true = np.linspace(0.01, 0.9, K)
              u_true = Optim.forceRhoInBounds(u_true)
              Vd = sampleVd(u_true, nDoc, alpha, PRNG=PRNG)
              DocTopicCount = summarizeVdToDocTopicCount(Vd)

              initrho = Optim.forceRhoInBounds(PRNG.rand(K) * u_true)
              kwargs = dict(alpha=alpha, 
                      gamma=gamma, 
                      nDoc=nDoc,                      
                      DocTopicCount=DocTopicCount,
                      )

              ## Find optimum from "naive" init
              rho_est, f_est, Info = \
                       Optim.find_optimum_multiple_tries(
                                                 initrho=initrho,
                                                 **kwargs)
              assert np.all(np.isfinite(rho_est))
              assert np.isfinite(f_est)
              print Info['msg']

              ## Find optimum from "hot-start" init
              rho_orig = u_true
              rho_hot, f_hot, Ihot = \
                       Optim.find_optimum_multiple_tries(
                                                 initrho=rho_orig,
                                                 **kwargs)

              f_orig, _ = Optim.objFunc_constrained(rho_orig, **kwargs)
              print '  f_orig % 12.6f' % (f_orig)
              print '   f_hot % 12.6f' % (f_hot)
              print '   f_est % 12.6f' % (f_est)

              assert f_hot <= f_orig
              assert_allclose(rho_est, rho_hot, 'rho_est', 'rho_hot',
                              atol=0.02, rtol=1e-5)
              assert_allclose(f_est, f_hot, rtol=0.01)

              ## Verify that top-ranked comps are near "true" values
              beta_true = Optim.rho2beta_active(u_true)
              activeIDs = np.flatnonzero(np.cumsum(beta_true) < 0.995)
              nActive = len(activeIDs)

              beta_est = Optim.rho2beta_active(rho_est)
              assert_allclose(beta_est[:nActive], beta_true[:nActive],
                              'beta_est', 'beta_true',
                              atol=0.02, rtol=0.05)

