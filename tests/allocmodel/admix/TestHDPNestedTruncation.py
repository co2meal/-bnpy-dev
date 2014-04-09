'''
Unit tests for HDPModel.py that examine nested truncation

Test Strategy:

We start with a known v / beta at fixed truncation K,
 and generate the "observed data": group-level probs Pi


We then examine the estimated posterior for v
  using Pi with all K components, 
  and an "condensed" Pi which combines the last J comps in stick-break order

The hope is that both versions have essentially the same posterior over the first few components of v. Tests verify whether this is true.

'''

import sys
import numpy as np
import unittest
from matplotlib import pylab

import bnpy.allocmodel.admix.OptimizerForHDPFullVarModel as HVO

np.set_printoptions(precision=3, suppress=False, linewidth=140)
def np2flatstr(xvec):
  return ' '.join( ['%9.3f' % (x) for x in xvec])

def makePiMatrix(beta, nDoc=1000, gamma=0.5):
  PRNG = np.random.RandomState(0)
  Pi = PRNG.dirichlet(gamma * beta, size=nDoc)
  assert np.allclose(np.sum(Pi, axis=1), 1.0)
  return Pi

def summarizePi(Pi):
  logPi = np.log(Pi)
  infmask = np.isinf(logPi)
  if np.any(infmask):
    minVal = np.min(logPi[np.logical_not(infmask)])
    logPi[infmask] = np.minimum(-10000, minVal)
  return np.sum( logPi, axis=0)

class TestNestedTrunc(unittest.TestCase):
  def shortDescription(self):
    return None

  def setUp(self):
    ''' Create a model and some test data for quick experiments.
    '''
    self.alpha0 = 1.0
    self.gamma = 0.5
    self.K = 5
    self.beta = np.asarray( [0.350, 0.150, 0.300, 0.100, 0.060, 0.040] )
    assert np.allclose(1.0, self.beta.sum())

    ## Make data
    self.nDoc = 2000
    Pi5 = makePiMatrix(self.beta, nDoc=self.nDoc, gamma=self.gamma)
    Pi4 = np.hstack([Pi5[:,:4], Pi5[:, 4:].sum(axis=1)[:,np.newaxis]])
    assert np.allclose(1.0, np.sum(Pi4, axis=1))
    assert Pi4.shape[1] == 4 + 1

    self.sumLogPi_K5 = summarizePi(Pi5)
    self.sumLogPi_K4 = summarizePi(Pi4)

  def test_K5__estimated_beta_near_truth(self, nTrial=1):
    ''' Verify for K=5 data that we recover variational parameters
          whose E[beta] is very close to the true beta
    '''
    self.verify__estimated_beta_near_truth(self.beta, self.sumLogPi_K5)


  def test_K4__estimated_beta_near_truth(self, nTrial=1):
    ''' Verify for K=4 data that we recover variational parameters
          whose E[beta] is very close to the true beta
    '''
    truebeta = np.hstack([self.beta[:4], self.beta[4:].sum()])
    self.verify__estimated_beta_near_truth(truebeta, self.sumLogPi_K4)

  def verify__estimated_beta_near_truth(self, truebeta, sumLogPi, nTrial=1):
    ''' Verify for K=5 data that we recover variational parameters
          whose E[beta] is very close to the true beta
    '''
    print ''
    u, fofu, Info = HVO.estimate_u_multiple_tries(
                          sumLogPi=sumLogPi,
                          nDoc=self.nDoc,
                          gamma=self.gamma, alpha0=self.alpha0,
                          initu=None, approx_grad=False)
    Ebeta = HVO.u2beta(u)
    print 'f(u):  %.5e' % (fofu)
    print 'E[beta]'
    print '    ', np2flatstr(truebeta), '  truth'
    print '    ', np2flatstr(Ebeta), '  estimated'
    u1, u0 = HVO._unpack(u)
    print ''
    print 'u1  ', np2flatstr(u1)
    print 'u0  ', np2flatstr(u0)
    assert self.verify_beta(Ebeta, truebeta)

  def verify_beta(self, Ebeta, truebeta=None):
    ''' Verify that given vector Ebeta is "close enough" to desired beta
    '''
    if truebeta is None:
      truebeta = self.beta
    absDiff = np.abs(Ebeta - truebeta)
    percDiff = absDiff / truebeta
    absDiffPasses = np.all(absDiff < 0.02)
    percDiffPasses = np.all(percDiff < 0.10)
    assert absDiffPasses
    assert percDiffPasses
    return absDiffPasses and percDiffPasses


  def test_plot(self):
    sumLogPi = self.sumLogPi_K5
    u, fofu, Info = HVO.estimate_u_multiple_tries(
                          sumLogPi=sumLogPi,
                          nDoc=self.nDoc,
                          gamma=self.gamma, alpha0=self.alpha0,
                          initu=None, approx_grad=False)
    K = u.size/2
    bestomega = np.zeros(K)
    for k in xrange(K):
      pylab.subplot(K, 1, k+1)
      bestomega[k] = self.plot_objective_vary_omega(u, k, sumLogPi)
    pylab.show(block=True)
    # Last component should have more variance (smaller best omega)
    assert bestomega[0] > bestomega[-1]

  def plot_objective_vary_omega(self, u, k, sumLogPi):
    u = u.copy()
    K = u.size/2
    Ev = HVO.u2v(u)
    omega = np.linspace(0.5*self.nDoc, 10*self.nDoc, 1000)
    f = np.zeros_like(omega)
    for ii in xrange(len(omega)):
      u[k] = Ev[k] * omega[ii]
      u[K + k] = (1 - Ev[k]) * omega[ii]
      f[ii] = HVO.objFunc_u(u, sumLogPi, 
                                self.nDoc, self.gamma, self.alpha0)
    bestomega = omega[f.argmin()]
    pylab.plot( omega, f, 'k.-')
    # Plot vertical line to indicate minimum value
    pylab.plot( bestomega*np.ones(2), [f.min(), f.max()], 'r--')
    return bestomega
