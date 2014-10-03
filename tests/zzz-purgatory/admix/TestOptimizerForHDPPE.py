
import sys
import numpy as np
import warnings
import unittest
from matplotlib import pylab
import bnpy.allocmodel.admix.OptimizerForHDPPE as OptPE

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
    print minVal, '!!!!!!'
  return np.sum( logPi, axis=0)

########################################################### TestBasics
###########################################################
class TestBasics(unittest.TestCase):
  def shortDescription(self):
    return None

  def test__c2v(self):
    ''' Verify we can transform back and forth between
          constrained and unconstrained variables
    '''
    K = 5
    for s in [2, 3, 4, 5, 6, 7, 8]:
      v = 1.0/s * np.ones(K)
      c = OptPE.v2c(v)
      assert c.size == K
      v2 = OptPE.c2v(c)
      assert np.allclose(v, v2)

  def test__initv(self):
    initv = OptPE.create_initv(5)
    assert initv.min() > 0
    assert initv.max() < 1.0
    beta = OptPE._v2beta(initv)
    assert beta.min() > OptPE.EPS
    assert beta[-2] > beta[-1]
    assert np.allclose( beta[0], beta[1])

  def test__objFunc_constrained(self):
    ''' Verify *constrained* objective function
          * delivers scalar func value, grad vector of correct size
    '''
    K = 5
    kwargs = dict(sumLogPi=0, nDoc=0, alpha=0.5, gamma=1.0)
    for s in [2, 3, 4, 5, 6, 7, 8]:
      v = 1.0 / s * np.ones(K)
      f, grad = OptPE.objFunc_constrained(v, **kwargs)
      print f
      print grad
      assert np.asarray(f).ndim == 0

      assert grad.ndim == 1
      assert grad.size == K
      assert not np.any(np.isinf(f))
      assert not np.any(np.isinf(grad))

  def test__objFunc_unconstrained(self):
    ''' Verify *unconstrained* objective function 
          * delivers scalar func value, grad vector of correct size
          * delivers same function value as constrained objective
    '''
    K = 5
    PRNG = np.random.RandomState(0)
    kwargs = dict(sumLogPi=0, nDoc=0, alpha=0.5, gamma=1.0)
    for s in [1, 2, 3, 4, 5, 6, 7, 8]:
      v = PRNG.rand(K)
      c = OptPE.v2c(v)

      f, grad = OptPE.objFunc_unconstrained(c, **kwargs)
      print f
      print grad
      assert np.asarray(f).ndim == 0

      assert grad.ndim == 1
      assert grad.size == K
      assert not np.any(np.isinf(f))
      assert not np.any(np.isinf(grad))

      f2, grad2 = OptPE.objFunc_constrained(v, **kwargs)
      assert np.allclose(f, f2)
      v2, dvdc = OptPE.c2v(c, doGrad=True)
      assert np.allclose(grad, dvdc * grad2)

  def test__knownoptimum__nDoc0(self):
    ''' Verify that MAP estimate does *not* exist when nDoc=0
        Mode for random variable v ~ Beta(a, b) = (a-1) / (a+b-2) 
    '''
    K = 2
    gamma = 1.0
    kwargs = dict(sumLogPi=0, nDoc=0, approx_grad=True, alpha=0.5, gamma=gamma)
    vopt = 1.0 / (1.0 + gamma) * np.ones(K)
    fopt = OptPE.objFunc_constrained(vopt, **kwargs)
    print '%.3e ******' % (fopt)
    print '    ', np2flatstr(vopt)

    for delta in np.linspace(0.001, 0.05, 3):
      v = vopt + delta
      f = OptPE.objFunc_constrained(v, **kwargs)
      print '%.3e' % (f)
      print '    ', np2flatstr(v)
      assert fopt > f

    for delta in np.linspace(0.001, 0.05, 3):
      v = vopt - delta
      f = OptPE.objFunc_constrained(v, **kwargs)
      print '%.3e' % (f)
      print '    ', np2flatstr(v)
      assert f > fopt


########################################################### Test K=5
###########################################################
class TestOptimizationK5(unittest.TestCase):
  ''' Unit tests for accuracy and consistency of gradient-descent
  '''
  def shortDescription(self):
    return None

  def setUp(self):
    ''' Create a model and some test data for quick experiments.
    '''
    self.alpha0 = 2.0
    self.gamma = 0.99
    self.K = 5
    self.beta = np.asarray( [0.350, 0.150, 0.300, 0.100, 0.060, 0.040] )
    assert np.allclose(1.0, self.beta.sum())

  def test__find_optimum__approxgrad(self):
    ''' Verify for K=5 data that we recover variational parameters
          whose E[beta] is very close to the true beta
    '''
    print ''
    nDocRange = [100, 1000, 10000]
    atol = [0.04, 0.01, 0.005]
    error = np.zeros(len(nDocRange))
    for ii, nDoc in enumerate(nDocRange):
      Pi = makePiMatrix(self.beta, nDoc, self.gamma)
      sumLogPi = summarizePi(Pi)

      v, f, Info = OptPE.find_optimum(nDoc=nDoc, sumLogPi=sumLogPi,
                                      gamma=self.gamma, alpha=self.alpha0,
                                      approx_grad=True)
      betaest = OptPE._v2beta(v)

      print '----------- nDoc %d' % (nDoc)
      print '%s  true' % (np2flatstr(self.beta))
      print '%s  est' % (np2flatstr(betaest))
      error[ii] = np.abs(betaest - self.beta).sum()
      assert np.allclose(self.beta, betaest, atol=atol[ii])
    print error
    # Verify error with more data is smaller than with little data
    assert error[0] > error[-1]


  def test__find_optimum__truegrad(self):
    ''' Verify for K=5 data that we recover variational parameters
          whose E[beta] is very close to the true beta
    '''
    print ''
    nDocRange = [100, 1000, 10000]
    atol = [0.04, 0.01, 0.005]
    error = np.zeros(len(nDocRange))
    for ii, nDoc in enumerate(nDocRange):
      Pi = makePiMatrix(self.beta, nDoc, self.gamma)
      sumLogPi = summarizePi(Pi)

      v, f, Info = OptPE.find_optimum(nDoc=nDoc, sumLogPi=sumLogPi,
                                      gamma=self.gamma, alpha=self.alpha0,
                                      approx_grad=False)
      betaest = OptPE._v2beta(v)

      print '----------- nDoc %d' % (nDoc)
      print '%s  true' % (np2flatstr(self.beta))
      print '%s  est' % (np2flatstr(betaest))
      error[ii] = np.abs(betaest - self.beta).sum()
      assert np.allclose(self.beta, betaest, atol=atol[ii])
    print error
    # Verify error with more data is smaller than with little data
    assert error[0] > error[-1]

  def test__find_optimum_multiple_tries__truegrad(self):
    ''' Verify for K=5 data that we recover variational parameters
          whose E[beta] is very close to the true beta
    '''
    print ''
    nDocRange = [100, 1000, 10000]
    atol = [0.04, 0.01, 0.005]
    error = np.zeros(len(nDocRange))
    for ii, nDoc in enumerate(nDocRange):
      Pi = makePiMatrix(self.beta, nDoc, self.gamma)
      sumLogPi = summarizePi(Pi)

      v, f, Info = OptPE.find_optimum_multiple_tries(
                                      nDoc=nDoc, sumLogPi=sumLogPi,
                                      gamma=self.gamma, alpha=self.alpha0,
                                      approx_grad=False)
      betaest = OptPE._v2beta(v)

      print '----------- nDoc %d' % (nDoc)
      print '%s  true' % (np2flatstr(self.beta))
      print '%s  est' % (np2flatstr(betaest))
      error[ii] = np.abs(betaest - self.beta).sum()
      assert np.allclose(self.beta, betaest, atol=atol[ii])
    print error
    # Verify error with more data is smaller than with little data
    assert error[0] > error[-1]




class TestOptimizationK50(unittest.TestCase):
  ''' Unit tests for accuracy and consistency of gradient-descent
  '''
  def shortDescription(self):
    return None

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


  def test__find_optimum__truegrad(self):
    ''' Verify find_optimum gives good value for v
    '''
    print ''
    nDocRange = [100, 1000, 10000]
    atol = [0.02, 0.01, 0.005]
    error = np.zeros(len(nDocRange))
    for ii, nDoc in enumerate(nDocRange):
      Pi = makePiMatrix(self.beta, nDoc, self.gamma)
      sumLogPi = summarizePi(Pi)

      v, f, Info = OptPE.find_optimum(nDoc=nDoc, sumLogPi=sumLogPi,
                                      gamma=self.gamma, alpha=self.alpha0,
                                      approx_grad=False)
      betaest = OptPE._v2beta(v)

      print '----------- nDoc %d' % (nDoc)
      print '%s  true' % (np2flatstr(self.beta))
      print '%s  est' % (np2flatstr(betaest))
      error[ii] = np.abs(betaest - self.beta).sum()
      assert np.allclose(self.beta, betaest, atol=atol[ii])
    print ' ERRORS:', error
    # Verify error with more data is smaller than with little data
    assert error[0] > error[-1]


class TestOptimizationK100(unittest.TestCase):
  ''' Unit tests for accuracy and consistency of gradient-descent
  '''
  def shortDescription(self):
    return None

  def setUp(self):
    ''' Create a model and some test data for quick experiments.
    '''
    self.K = 100
    self.alpha0 = 1.0
    self.gamma = 0.99

    PRNG = np.random.RandomState(0)
    beta = 0.01 + PRNG.rand(self.K+1)
    beta[-1] = 0.0001
    self.beta = beta / np.sum(beta)
    assert self.beta.size == self.K + 1
    assert np.allclose(1.0, self.beta.sum())


  def test__find_optimum__truegrad(self):
    ''' Verify find_optimum gives good value for v
    '''
    print ''
    nDocRange = [100, 1000, 10000, 100000]
    atol = [0.02, 0.01, 0.005, 0.003]
    error = np.zeros(len(nDocRange))
    for ii, nDoc in enumerate(nDocRange):
      Pi = makePiMatrix(self.beta, nDoc, self.gamma)
      sumLogPi = summarizePi(Pi)

      v, f, Info = OptPE.find_optimum(nDoc=nDoc, sumLogPi=sumLogPi,
                                      gamma=self.gamma, alpha=self.alpha0,
                                      approx_grad=False)
      betaest = OptPE._v2beta(v)

      print '----------- nDoc %d' % (nDoc)
      print '%s  true' % (np2flatstr(self.beta))
      print '%s  est' % (np2flatstr(betaest))
      error[ii] = np.abs(betaest - self.beta).sum()
      assert np.allclose(self.beta, betaest, atol=atol[ii])
    print ' ERRORS:', error
    # Verify error with more data is smaller than with little data
    assert error[0] > error[-1]
    print betaest[-1]
    print self.beta[-1]
