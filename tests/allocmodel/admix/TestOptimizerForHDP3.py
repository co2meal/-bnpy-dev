
import sys
import numpy as np
import warnings
import unittest
from matplotlib import pylab
import bnpy.allocmodel.admix.OptimizerForHDP3 as HO
import bnpy.allocmodel.admix.OptimizerForHDP2 as HO2
import bnpy.allocmodel.admix.OptimizerForHDPFullVarModel as HVO

np.set_printoptions(precision=3, suppress=False, linewidth=140)
def np2flatstr(xvec, Kmax=10):
  return ' '.join( ['%9.3f' % (x) for x in xvec[:Kmax]])

def makePiMatrix(beta, nDoc=1000, gamma=0.5):
  PRNG = np.random.RandomState(0)
  Pi = PRNG.dirichlet(gamma * beta, size=nDoc)
  assert np.allclose(np.sum(Pi, axis=1), 1.0)
  return Pi

def summarizePi(Pi):
  with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=RuntimeWarning,
                               message='divide by zero')
    logPi = np.log(Pi)
  infmask = np.isinf(logPi)
  if np.any(infmask):
    minVal = np.min(logPi[np.logical_not(infmask)])
    logPi[infmask] = np.minimum(-10000, minVal-5000)
  print 'MIN VAL:', logPi.min()
  return np.sum( logPi, axis=0)

class TestBasics(unittest.TestCase):
  def shortDescription(self):
    return None

  def test__c2rhoomega(self):
    ''' Verify we can transform back and forth between
          constrained and unconstrained variables
    '''
    for s in [2, 3, 4, 5, 6, 7, 8]:
      rho = 1.0/s * np.ones(4)
      omega = 10*s * np.ones(4)
      rhoomega = np.hstack([rho, omega])
      c = HO.rhoomega2c(rhoomega)
      assert c.size == 8
      assert np.allclose( c[4:], np.log(omega))
      ro = HO.c2rhoomega(c)
      assert np.allclose(rhoomega, ro)
    
  def test__sigmoid(self):
    ''' Verify that sigmoid and invsigmoid functions
          deliver correct results
    '''
    assert np.allclose( HO.sigmoid(0), 0.5)
    assert np.allclose( HO.sigmoid(-100), 0)
    assert np.allclose( HO.sigmoid(100), 1)

    assert np.allclose( HO.sigmoid(HO.invsigmoid(0.01)), 0.01)
    assert np.allclose( HO.sigmoid(HO.invsigmoid(0.4)), 0.4)
    assert np.allclose( HO.sigmoid(HO.invsigmoid(0.99)), 0.99)

    assert np.allclose( HO.invsigmoid(HO.sigmoid(9.23)), 9.23)
    assert np.allclose( HO.invsigmoid(HO.sigmoid(-9.13)), -9.13)

  def test__initrho_beta2v(self):
    initrho = HO.create_initrho(5)
    assert initrho.min() > 0
    assert initrho.max() < 1.0
    beta = HO._v2beta(initrho)
    print initrho
    print beta
    assert beta.min() > HO.EPS
    assert beta[-2] > beta[-1]
    assert np.allclose( beta[0], beta[1])

  def test__v2beta(self):
    PRNG = np.random.RandomState(0)
    for trial in range(10):
      v = PRNG.rand(7)
      beta = HO._v2beta(v)
      v2 = HO._beta2v(beta)
      assert np.allclose( v, v2)
      assert np.allclose(1.0, beta.sum())


  def test__objFunc_constrained(self):
    ''' Verify *constrained* objective function
          * delivers scalar func value, grad vector of correct size
    '''
    kwargs = dict(sumLogPi=0, nDoc=0, alpha=0.5, gamma=1.0)
    for s in [2, 3, 4, 5, 6, 7, 8]:
      rho = 1.0 / s * np.ones(4)
      omega = 5.5 * s * np.ones(4)
      rhoomega = np.hstack([rho, omega])
      f, grad = HO.objFunc_constrained(rhoomega, **kwargs)
      print f
      print grad
      assert np.asarray(f).ndim == 0

      assert grad.ndim == 1
      assert grad.size == 8
      assert not np.any(np.isinf(f))
      assert not np.any(np.isinf(grad))

  def test__objFunc_unconstrained(self):
    ''' Verify *unconstrained* objective function 
          * delivers scalar func value, grad vector of correct size
          * delivers same function value as constrained objective
    '''
    PRNG = np.random.RandomState(0)
    kwargs = dict(sumLogPi=0, nDoc=0, alpha=0.5, gamma=1.0)
    for s in [1, 2, 3, 4, 5, 6, 7, 8]:
      rho = PRNG.rand(4)
      omega = 5.5 * s * PRNG.rand(4)
      rhoomega = np.hstack([rho, omega])
      c = HO.rhoomega2c(rhoomega)

      f, grad = HO.objFunc_unconstrained(c, **kwargs)
      print f
      print grad
      assert np.asarray(f).ndim == 0

      assert grad.ndim == 1
      assert grad.size == 8
      assert not np.any(np.isinf(f))
      assert not np.any(np.isinf(grad))

      f2, grad2 = HO.objFunc_constrained(rhoomega, **kwargs)
      assert np.allclose(f, f2)
      assert not np.allclose(grad, grad2)


class TestOptimizationK5(unittest.TestCase):
  ''' Unit tests for accuracy and consistency of gradient-descent
  '''
  def shortDescription(self):
    return None

  def setUp(self):
    ''' Create a model and some test data for quick experiments.
    '''
    self.alpha0 = 1.0
    self.gamma = 0.99
    self.K = 5
    self.beta = np.asarray( [0.350, 0.150, 0.300, 0.100, 0.060, 0.040] )
    assert np.allclose(1.0, self.beta.sum())

    ## Make data
    self.nDoc = 2000
    Pi5 = makePiMatrix(self.beta, nDoc=self.nDoc, gamma=self.gamma)

    self.sumLogPi = summarizePi(Pi5)

  def verify__estimated_beta_near_truth(self, truebeta, 
                                              nDoc, sumLogPi,
                                              **kwargs):
    ''' Verify that we recover variational parameters
          whose E[beta] is very close to the true beta
    '''
    assert 'alpha' in kwargs
    assert 'gamma' in kwargs
    print ''
    rhoomega, f, Info = HO.find_optimum(
                          sumLogPi=sumLogPi,
                          nDoc=nDoc,
                          thetaVersion=0,
                          **kwargs)

    rho, omega, K = HO._unpack(rhoomega)
    assert K == sumLogPi.size - 1

    Ebeta = HO._v2beta(rho)
    print 'E[beta]'
    print '    ', np2flatstr(truebeta), '  truth'
    print '    ', np2flatstr(Ebeta), '  estimated'
    assert self.verify_beta(Ebeta, truebeta)
    return rhoomega, f, Info

  def verify_beta(self, Ebeta, truebeta=None):
    ''' Verify that given vector Ebeta is "close enough" to desired beta
    '''
    if truebeta is None:
      truebeta = self.beta
    absDiff = np.abs(Ebeta - truebeta)
    percDiff = absDiff / truebeta
    absDiffPasses = np.all(absDiff[:-1] < 0.04)
    if self.K < 10:
      percDiffPasses = np.all(percDiff[:-1] < 0.16)
    else:
      percDiffPasses = np.all(percDiff[:-1] < 0.20)

    assert absDiffPasses
    assert percDiffPasses
    return absDiffPasses and percDiffPasses

  def verify__beta_near_truth__multiple_tries(self, truebeta, 
                                              nDoc, sumLogPi,
                                              alpha=None, 
                                              **kwargs):
    ''' Verify that we recover variational parameters
          whose E[beta] is very close to the true beta
    '''
    if alpha is None:
      alpha = self.alpha0

    print ''
    rho, omega, f, Info = HO.find_optimum_multiple_tries(
                          sumLogPi=sumLogPi[:-1],
                          nDoc=nDoc,
                          gamma=self.gamma,
                          alpha=alpha,
                          **kwargs)


    Ebeta = HO._v2beta(rho)
    print 'E[beta]'
    print '    ', np2flatstr(truebeta), '  truth'
    print '    ', np2flatstr(Ebeta), '  estimated'
    assert self.verify_beta(Ebeta, truebeta)
    return rho, omega, f, Info
  
  ######################################################### tests
  def test__find_optimum__nDoc0_approxgrad(self):
    ''' Verify for K=5 data that we recover variational parameters
          whose E[beta] is very close to the true beta
    '''
    for alpha in [0.5, 1.0, 1.5, 2.0, 5.0]:
      beta0 = HO._v2beta( 1.0/(1+alpha) * np.ones(self.K) )
      sumLogPi = np.zeros(self.K+1)
      self.verify__estimated_beta_near_truth(beta0, 
                                            0, sumLogPi,
                                            alpha=alpha,
                                            approx_grad=True)

  def test__find_optimum__nDoc0(self):
    ''' Verify for K=5 data that we recover variational parameters
          whose E[beta] is very close to the true beta
    '''
    gamma = 0.5
    for alpha in [0.5, 1.0, 1.5, 2.0, 5.0]:
      beta0 = HO._v2beta( 1.0/(1+alpha) * np.ones(self.K))
      sumLogPi = np.zeros(self.K + 1)
      rhoomega, f, Info = self.verify__estimated_beta_near_truth(
                                            beta0, 0, sumLogPi,
                                            alpha=alpha, gamma=gamma,
                                            approx_grad=False, factr=1e6)
      f, grad = Info['objFunc'](rhoomega)
      if self.K < 10:
        assert np.max(np.abs(grad)) < 1e-4
      else:
        assert np.max(np.abs(grad)) < 5*1e-4


  def test__find_optimum_multiple_tries__nDoc0(self):
    ''' Verify for K=5 data that we recover variational parameters
          whose E[beta] is very close to the true beta
    '''
    for alpha in [0.5, 1.0, 1.5, 2.0, 5.0]:
      beta0 = HO._v2beta( 1.0/(1+alpha) * np.ones(self.K) )
      sumLogPi = np.zeros(self.K+1)
      rho, omega, f, Info = self.verify__beta_near_truth__multiple_tries(
                                            beta0, 0, sumLogPi,
                                            alpha=alpha,
                                            approx_grad=False, factr=1e6)

  def test__find_optimum__nDocSensitivity(self):
    nDocVals = [100, 200, 400, 800, 1600, 3200, 6400]
    omegaVals = np.zeros( (len(nDocVals), self.K))
    rhoVals = np.zeros( (len(nDocVals), self.K))
    betaVals = np.zeros( (len(nDocVals), self.K+1))
    for ii, nDoc in enumerate(nDocVals): 
      Pi5 = makePiMatrix(self.beta, nDoc=nDoc, gamma=self.gamma)
      sumLogPi = summarizePi(Pi5)

      rho, omega, f, Info = HO.find_optimum_multiple_tries(
                          sumLogPi=sumLogPi,
                          nDoc=nDoc,
                          gamma=self.gamma,
                          alpha=self.alpha0,
                          thetaVersion=0,
                          approx_grad=True)
      rho2, omega2, f2, Info2 = HO.find_optimum_multiple_tries(
                          sumLogPi=sumLogPi,
                          nDoc=nDoc,
                          gamma=self.gamma,
                          alpha=self.alpha0,
                          thetaVersion=0,
                          approx_grad=False)
      assert np.allclose(rho, rho2, atol=0.001)
      omegaVals[ii] = omega2
      rhoVals[ii] = rho2
      betaVals[ii] = HO._v2beta(rho2)
      
    from IPython import embed; embed()
    from matplotlib import pylab
    pylab.subplot(3, 1, 1)
    pylab.plot( nDocVals, omegaVals[:, -1], 'm--')
    pylab.ylabel('omega(K)')

    pylab.subplot(3, 1, 2)
    pylab.plot( nDocVals, rhoVals[:, -1], 'm--')
    pylab.ylabel('rho(K)')

    pylab.subplot(3, 1, 3)
    pylab.plot( nDocVals, betaVals[:, -1], 'm--')
    pylab.ylabel('beta (leftover)')
    pylab.xlabel('nDoc')


  def test__find_optimum__nDoc2000__compareApproxAndExact(self):
    rhoomega, f, Info = self.verify__estimated_beta_near_truth(self.beta, 
                                            self.nDoc, self.sumLogPi,
                                            alpha=self.alpha0,
                                            approx_grad=True, factr=1e5)
    rhoomega2, f2, Info2 = self.verify__estimated_beta_near_truth(self.beta, 
                                            self.nDoc, self.sumLogPi,
                                            alpha=self.alpha0,
                                            approx_grad=False, factr=1e5)

    f, grad = Info['objFunc'](rhoomega)
    f2, grad2 = Info2['objFunc'](rhoomega2)

    print 'Objective function'
    print '  %.9e  %s approx' % (f, Info['task'])
    print '  %.9e  %s exact' % (f2, Info2['task'])

    if self.K < 10:
      print 'rhoomega optimal value'
      print '  ', np2flatstr(rhoomega), '  approx'
      print '  ', np2flatstr(rhoomega2), '  exact'

      print 'gradient at optimal value'
      print '  ', np2flatstr(grad), '  approx'
      print '  ', np2flatstr(grad2), '  exact'
    else:
      print 'rho optimal value'
      print '  ', np2flatstr(rhoomega[:10]), '  approx'
      print '  ', np2flatstr(rhoomega2[:10]), '  exact'

      print 'omega optimal value'
      print '  ', np2flatstr(rhoomega[self.K:self.K+10]), '...  approx'
      print '  ', np2flatstr(rhoomega2[self.K:self.K+10]), '...  exact'

      print 'max(abs(gradient)) at optimal value'
      print '  %.3f approx' % (np.max(np.abs(grad)))
      print '  %.3f exact' % (np.max(np.abs(grad2)))

    K = rhoomega.size/2
    assert np.allclose(rhoomega[:K], rhoomega2[:K], atol=0.003)
    if self.K < 10:
      assert np.allclose(rhoomega[K:], rhoomega2[K:], rtol=0.5)
      assert np.max(np.abs(grad)) < 1e-2
      assert np.max(np.abs(grad2)) < 1e-3
    else:
      assert np.max(np.abs(grad2)) < 1e-2


  def test__find_optimum_multiple_tries__nDoc2000(self):
    ''' Verify for K=5 data that we recover variational parameters
          whose E[beta] is very close to the true beta
    '''
    rho, omega, f, Info = self.verify__beta_near_truth__multiple_tries(
                                            self.beta, self.nDoc, 
                                            sumLogPi=self.sumLogPi,
                                            alpha=self.alpha0,
                                            approx_grad=False)

  def test__optimum_does_not_move_with_successive_runs(self):
    ''' Verify for K=5 data that we recover variational parameters
          whose E[beta] is very close to the true beta
    '''
    rho, omega, f, Info = self.verify__beta_near_truth__multiple_tries(
                                            self.beta, self.nDoc, 
                                            sumLogPi=self.sumLogPi,
                                            alpha=self.alpha0,
                                            approx_grad=False)
    print '% .8e' % (f) 
    print '             ', np2flatstr(rho)
    Orig = dict(rho=rho, omega=omega, f=f)
    for trial in range(10):
      rho, omega, f, Info = self.verify__beta_near_truth__multiple_tries(
                                            self.beta, self.nDoc, 
                                            sumLogPi=self.sumLogPi,
                                            alpha=self.alpha0,
                                            gamma=self.gamma,
                                            initrho=rho, initomega=omega,
                                            approx_grad=False)
      print '% .8e' % (f) 
      print '             ', np2flatstr( rho - Orig['rho'])
      assert np.allclose(f, Orig['f'])
      if self.K < 10:
        assert np.allclose(rho, Orig['rho'], atol=1e-6)
      else:
        print rho[-5:]
        print Orig['rho'][-5:]
        assert np.allclose(rho[:self.K-3], Orig['rho'][:self.K-3], atol=1e-3)

  def test__find_optimum__sensitivityToInit(self):
    factr = 1e5
    seeds = [0, 1, 2, 3, 4, 5, 6]
    fs = np.zeros_like(seeds)
    for ii, seed in enumerate(seeds):
      PRNG = np.random.RandomState(seed)
      initrho = PRNG.rand(self.K)
      initomega = self.nDoc * PRNG.rand(self.K)
      rhoomega, f, Info = self.verify__estimated_beta_near_truth(self.beta, 
                                            self.nDoc, self.sumLogPi,
                                            alpha=self.alpha0,
                                            approx_grad=False, factr=factr,
                                            initrho=initrho,
                                            initomega=initomega)
      f, grad = Info['objFunc'](rhoomega)
      fs[ii] = f
      print 'Objective function'
      print '  %.9e  %s' % (f, Info['task'])
   
      print 'rho optimal value'
      print '  ', np2flatstr(rhoomega[:self.K])

      print 'omega optimal value'
      print '  ', np2flatstr(rhoomega[self.K:])

    fs = np.abs(fs)
    percDiff = (np.max(fs) - np.min(fs)) / np.min(fs)
    print 'percDiff(fvals for solutions)=', percDiff
    assert percDiff < 0.0002


  def test__find_optimum__sensitivityToFactr(self):
    factrs = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]  #1e10 is bad!
    fs = np.zeros_like(factrs)
    for ii, factr in enumerate(factrs):
      rhoomega, f, Info = self.verify__estimated_beta_near_truth(self.beta, 
                                            self.nDoc, self.sumLogPi,
                                            alpha=self.alpha0,
                                            approx_grad=False, factr=factr)
      f, grad = Info['objFunc'](rhoomega)
      fs[ii] = f
      print 'Objective function'
      print '  %.9e  %s' % (f, Info['task'])
   
      print 'rhoomega optimal value'
      print '  ', np2flatstr(rhoomega)
    
    fs = np.abs(fs)
    percDiff = (np.max(fs) - np.min(fs)) / np.min(fs)
    print 'percDiff(fvals for solutions)=', percDiff
    assert percDiff < 0.0002
  

class TestOptimizationK50(TestOptimizationK5):
  ''' Unit tests for accuracy and consistency of gradient-descent
  '''

  def setUp(self):
    ''' Create a model and some test data for quick experiments.
    '''
    self.alpha0 = 1.0
    self.gamma = 0.99
    self.K = 50
    self.beta = np.hstack([ 0.3/(self.K/2)*np.ones(self.K/2),
                            0.7/(self.K/2+1)*np.ones(self.K/2 +1),
                           ])
    assert self.beta.size == self.K + 1
    assert np.allclose(1.0, self.beta.sum())

    ## Make data
    self.nDoc = 2000
    Pi = makePiMatrix(self.beta, nDoc=self.nDoc, gamma=self.gamma)
    self.sumLogPi = summarizePi(Pi)
    assert self.sumLogPi.size == self.K + 1
