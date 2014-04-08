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

import bnpy.allocmodel.admix.OptimizerForHDPFullValModel as HVO

EPS = 1e-12
nDoc=4000

def makePiMatrix(beta, nDoc=1000, gamma=0.5):
  PRNG = np.random.RandomState(0)
  Pi = PRNG.dirichlet(gamma * beta, size=nDoc)
  assert np.allclose(np.sum(self.Pi, axis=1), 1.0)
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
    self.gamma = 0.99
    self.K = 5
    self.beta = np.asarray( [0.3, 0.2, 0.3, 0.1, 0.06, 0.04] )
    assert np.allclose(1.0, self.beta.sum())

    ## Make data
    self.nDoc = 1000
    Pi5 = makePiMatrix(beta, nDoc=self.nDoc, gamma=self.gamma)
    Pi4 = np.hstack([Pi5[:,:4], Pi5[:, 4:].sum(axis=1)])
    assert np.allclose(1.0, np.sum(Pi4, axis=1))
    assert Pi4.shape[1] == 4 + 1

    self.sumLogPi_K5 = summarizePi(Pi5)
    self.sumLogPi_K4 = summarizePi(Pi4)

  def test_estimate_is_near_truth(self, nTrial=1):
    ''' Verify that we recover variational parameters
          whose E[beta] is very close to the true beta
    '''
    success = 0
    np.set_printoptions(precision=4, suppress=True)
    print self.beta, '*optimal*'
    for trial in range(nTrial):
      u, fofu, Info = HVO.estimate_u_multiple_tries(
                          sumLogPi=self.sumLogPi_K5,
                          nDoc=self.nDoc,
                          gamma=self.gamma, alpha0=self.alpha0,
                          initu=None)
      Ebeta = HVO.u2beta(u)
      if self.verify_beta(Ebeta):
        success += 1
      else:
        print Ebeta
    print "%d/%d succeeded." % (success, nTrial)
    assert success == nTrial

  def verify_beta(self, Ebeta):
    ''' Verify that given vector Ebeta is "close enough" to desired beta
    '''
    absDiff = np.abs(Ebeta - self.beta)
    percDiff = absDiff / self.beta
    
