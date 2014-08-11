import sys
import numpy as np
from nose.plugins.skip import Skip, SkipTest
from scipy.optimize import approx_fprime
import warnings
import unittest

from bnpy.allocmodel.admix2.HDPSB import gtsum
import bnpy.allocmodel.admix2.LocalOptimizer as Optim

import BarsK10V900

def MakeSingleDocProblem(K=0, alpha=0, seed=0, nWordsPerDoc=100):
  Data = BarsK10V900.get_data(seed=seed, nDocTotal=1, nWordsPerDoc=100)
  Lik = np.log(Data.TrueParams['topics']).T[Data.word_id].copy()
  Lik = Lik[:, :K]
  Lik = np.exp(Lik - np.max(Lik, axis=1)[:,np.newaxis])

  Ebeta = (np.ones(K) - 0.01) / K
  Ebeta_gt = gtsum(Ebeta) + (1 - Ebeta.sum())
  return dict(wc_d=Data.word_count, Lik_d=Lik,
              alphaEbeta=alpha * Ebeta, alphaEbeta_gt=alpha * Ebeta_gt)

########################################################### TestK1
###########################################################
""" when K=1, we have a simple analytic solution
etaON = Nd + alphaEbeta
etaOFF = 0 + alphaEbeta_gt
"""

class TestK1(unittest.TestCase):
  def shortDescription(self):
    return None

  def testHasSaneOutput__objFunc_constrained(self):
    ''' Verify objective value and gradient vector have correct type and size
    '''
    print ''
    for K in [1]:
      for alpha in [0.1, 0.99]:
        for seed in [333, 777, 888]:
          print '==================  K %5d | alpha %.2f' % (K, alpha)
          for approx_grad in [1]:
            PRNG = np.random.RandomState(seed)
            ProbDict = MakeSingleDocProblem(K=K, seed=seed)
            etahat = 5 * PRNG.rand(2*K)
            if approx_grad:
              f = Optim.objFunc_constrained(etahat, approx_grad=1,
                                                 **ProbDict)
              g = np.ones(K)
            else:
              f, g = Optim.objFunc_constrained(etahat, approx_grad=0,
                                                 **ProbDict)

            print ' f= ', f
            assert type(f) == np.float64
            assert g.ndim == 1
            assert g.size == K            
            assert np.isfinite(f)
            assert np.all(np.isfinite(g))

  def testGradientZeroAtOptimum__objFunc_constrained(self):
    ''' Verify computed gradient at optimum is indistinguishable from zero
    '''
    print ''
    for K in [1]:
      for alpha in [0.1, 0.95]:
        print '==================  K %5d | alpha %.2f' % (K, alpha)

        ProbDict = MakeSingleDocProblem(K=K, seed=0, alpha=alpha)
        etaON = ProbDict['wc_d'].sum() + ProbDict['alphaEbeta']
        etaOFF = ProbDict['alphaEbeta_gt']
        etahat = np.hstack([etaON, etaOFF])
        #_, g = Optim.objFunc_constrained(uhat,
        #                                     approx_grad=0,
        #                                     **kwargs)

        ## Numerical gradient
        objFunc = lambda x: Optim.objFunc_constrained(x,
                                                          approx_grad=1,
                                                          **ProbDict)
        epsvec = 1e-8*np.ones(K)
        gapprox = approx_fprime(etahat, objFunc, epsvec)    

        printVectors('gapprox', gapprox)
        assert np.allclose(gapprox, np.zeros(K), atol=0.01)
        

  def testGradientExactAndApproxAgree__objFunc_constrained(self):
    ''' Verify computed gradient similar for exact and approx methods
    '''
    print ''
    for K in [1, 10, 107]:
      for alpha in [0.1, 0.95]:
        for gamma in [1., 3.14, 9.45]:
          for seed in [111, 222, 333]:
            PRNG = np.random.RandomState(seed)
            uhat = PRNG.rand(K)
            kwargs = dict(alpha=alpha, 
                      gamma=gamma,
                      nDoc=0,
                      sumLogVd=np.zeros(K),
                      sumLog1mVd=np.zeros(K))

            ## Exact gradient
            _, g = Optim.objFunc_constrained(uhat,
                                               approx_grad=0,
                                               **kwargs)

            ## Numerical gradient
            objFunc = lambda x: Optim.objFunc_constrained(x,
                                                            approx_grad=1,
                                                            **kwargs)
            epsvec = 1e-8 * np.ones(K)
            gapprox = approx_fprime(uhat, objFunc, epsvec)    

            print '      rho 1:10 ', np2flatstr(uhat)
            print '     grad 1:10 ', np2flatstr(g, fmt='% .6e')
            print '     grad 1:10 ', np2flatstr(gapprox, fmt='% .6e')
            if K > 10:
              print '    rho K-10:K ', np2flatstr(uhat[-10:])
              print '   grad K-10:K ', np2flatstr(g[-10:], fmt='% .6e')
              print 'gapprox K-10:K ', np2flatstr(gapprox[-10:], fmt='% .6e')

            assert np.allclose(g, gapprox, atol=1e-4, rtol=0.05)


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
              uhat = PRNG.rand(K)
              kwargs = dict(alpha=alpha, 
                      gamma=1, 
                      nDoc=nDoc,
                      sumLogVd=sumLogVd,
                      sumLog1mVd=sumLog1mVd)
              if approx_grad:
                f = Optim.objFunc_constrained(uhat, approx_grad=1,
                                                 **kwargs)
                g = np.ones(K)
              else:
                f, g = Optim.objFunc_constrained(uhat, approx_grad=0,
                                                 **kwargs)
              assert type(f) == np.float64
              assert g.ndim == 1
              assert g.size == K            
              assert np.isfinite(f)
              assert np.all(np.isfinite(g))


  def testGradientExactAndApproxAgree__objFunc_constrained(self):
    ''' Verify computed gradient similar for exact and approx methods
    '''
    print ''
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
              kwargs = dict(alpha=alpha, 
                      gamma=gamma, 
                      nDoc=nDoc,
                      sumLogVd=sumLogVd,
                      sumLog1mVd=sumLog1mVd)
              
              ## Exact gradient
              f, g = Optim.objFunc_constrained(rho, approx_grad=0,
                                                 **kwargs)

              ## Approx gradient
              objFunc = lambda x: Optim.objFunc_constrained(x, approx_grad=1,
                                                               **kwargs)
              epsvec = 1e-8 * np.ones(K)
              gapprox = approx_fprime(rho, objFunc, epsvec)    

              print '      rho 1:10 ', np2flatstr(rho)
              print '     grad 1:10 ', np2flatstr(g, fmt='% .6e')
              print '  gapprox 1:10 ', np2flatstr(gapprox, fmt='% .6e')
              if K > 10:
                print '    rho K-10:K ', np2flatstr(rho[-10:])
                print '   grad K-10:K ', np2flatstr(g[-10:], fmt='% .6e')
                print 'gapprox K-10:K ', np2flatstr(gapprox[-10:], fmt='% .6e')
              rtol_rho = 0.01
              atol_rho = 1e-4
              rhoGradOK = np.allclose(g, gapprox, atol=atol_rho,
                                                  rtol=rtol_rho)
              if not rhoGradOK:
                print 'VIOLATION DETECTED!'
                print 'grad_approx DOES NOT EQUAL grad_exact (within tolerance)'
                
                absDiff = np.abs(g - gapprox)
                tolDiff = (atol_rho + rtol_rho * np.abs(gapprox)) \
                            - absDiff
                worstIDs = np.argsort(tolDiff)
                print 'Top 5 worst mismatches'
                print np2flatstr( g[worstIDs[:5]], fmt='% .6f')
                print np2flatstr( gapprox[worstIDs[:5]], fmt='% .6f')
              assert rhoGradOK

  def testRecoverMeanGlobalSticksFromGeneratedData(self):
    ''' Verify that mean of V_d matrix is equal to original vector u
    '''
    print ''
    gamma = 1.0
    for K in [1, 10, 107]:
      for alpha in [0.9999, 0.9, 0.5]:
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

  def testRecoverUhatThatGeneratedData__find_optimum(self):
    ''' Verify that find_optimum's result is indistiguishable from analytic opt
    '''
    print ''
    gamma = 1.0
    for K in [93, 107, 85]: #, 10, 107]:
      for alpha in [0.9995]:
        for nDoc in [10000]:
          print '================== K %d | alpha %.2f | nDoc %d' \
                % (K, alpha, nDoc)

          for seed in [111, 222, 333]:

              PRNG = np.random.RandomState(seed)
              u_true = np.linspace(0.01, 0.99, K)
              Vd = sampleVd(u_true, nDoc, alpha, PRNG=PRNG)
              sumLogVd, sumLog1mVd = summarizeVd(Vd)

              inituhat = PRNG.rand(K)
              kwargs = dict(alpha=alpha, 
                      gamma=gamma, 
                      nDoc=nDoc,                      
                      sumLogVd=sumLogVd,
                      sumLog1mVd=sumLog1mVd,
                      )

              ## Find optimum from "naive" init
              uhat_est, f_est, Info = \
                       Optim.find_optimum_multiple_tries(
                                                 inituhat=inituhat,
                                                 **kwargs)
              assert np.all(np.isfinite(uhat_est))
              assert np.isfinite(f_est)
              print Info['msg']

              ## Find optimum from "hot-start" init
              uhat_orig = u_true
              uhat_hot, f_hot, Ihot = \
                       Optim.find_optimum_multiple_tries(
                                                 inituhat=uhat_orig,
                                                 **kwargs)

              f_orig, _ = Optim.objFunc_constrained(uhat_orig, **kwargs)
              print '  f_orig % 12.6f' % (f_orig)
              print '   f_hot % 12.6f' % (f_hot)
              print '   f_est % 12.6f' % (f_est)

              assert f_hot <= f_orig
              assert_allclose(uhat_est, uhat_hot, 'uhat_est', 'uhat_hot',
                              atol=0.01, rtol=1e-5)
              assert_allclose(f_est, f_hot, rtol=0.01)

              ## Verify that top-ranked comps are near "true" values
              beta_true = u_true.copy()
              beta_true[1:] *= np.cumprod( 1-u_true[:-1])
              activeIDs = np.flatnonzero(np.cumsum(beta_true) < 0.995)
              nActive = len(activeIDs)
              beta_est = uhat_est.copy()
              beta_est[1:] *= np.cumprod( 1-uhat_est[:-1])
              
              assert_allclose(beta_est[:nActive], beta_true[:nActive],
                              'beta_est', 'beta_true',
                              atol=0.015, rtol=1e-5)


def assert_allclose(a, b, aname='', bname='', 
                          atol=1e-8, rtol=1e-8, Ktop=5, fmt='% .6f'):
  if len(aname) > 0:
    print '------------------------- %s' % (aname.split('_')[0])
    printVectors(aname, a, bname, b)
  isOK = np.allclose(a, b, atol=atol, rtol=rtol)
  if not isOK:
    print 'VIOLATION DETECTED!'
    print 'args are not equal (within tolerance)'
                
    absDiff = np.abs(a - b)
    tolDiff = (atol + rtol * np.abs(b)) - absDiff
    worstIDs = np.argsort(tolDiff)
    print 'Top %d worst mismatches' % (Ktop)
    print np2flatstr( a[worstIDs[:Ktop]], fmt=fmt)
    print np2flatstr( b[worstIDs[:Ktop]], fmt=fmt)
  assert isOK

def printVectors(aname, a, bname=None, b=np.zeros(2), fmt='%9.6f', Kmax=10):
  if len(a) > Kmax:
    print 'FIRST %d' % (Kmax)
    printVectors(aname, a[:Kmax], bname, b[:Kmax], fmt, Kmax)
    print 'LAST %d' % (Kmax)
    printVectors(aname, a[-Kmax:], bname, b[-Kmax:], fmt, Kmax)

  else:
    print ' %10s %s' % (aname, np2flatstr(a, fmt, Kmax))
    if bname is not None:
      print ' %10s %s' % (bname, np2flatstr(b, fmt, Kmax))

def np2flatstr(xvec, fmt='%9.3f', Kmax=10):
  return ' '.join( [fmt % (x) for x in xvec[:Kmax]])
