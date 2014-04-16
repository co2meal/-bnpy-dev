'''
TestELBOInsertComps.py

Verify that when we insert new comps into an HDP topic model,
  the resulting "fast" construction via suff stats
  is equivalent to a simple direct manipulation of local parameters.

We **cannot** just use SS.insertComps() here. The sumLogPi term resulting from this procedure is off, and needs significant correction!
'''

from scipy.special import digamma, gammaln
import numpy as np
import unittest

import bnpy
import UtilForBirthTest as U

alpha = 5.0
gamma = 0.5

def createExpandedModel__viaSS(model, SS, freshSS):
  xSS = SS.copy()
  xmodel = model.copy()
  xSS.insertComps(freshSS)
  xmodel.update_global_params(xSS)
  return xmodel, xSS

def createExpandedModel__viaSScorrected2(model, SS, freshSS):
  xSS = SS.copy()
  xmodel = model.copy()
  xSS, AInfo, RInfo = xmodel.allocModel.insertCompsIntoSuffStatBag(
                                                  xSS, freshSS)

  xmodel.update_global_params(xSS)
  return xmodel, xSS

def createExpandedModel__viaLP(model, LP, freshSS, Data, freshData):
  ''' Directly construct LP and model with expanded set of K + Kfresh comps
  '''
  freshK = freshSS.K

  # First, prepare an expanded-set of local params,
  # exactly representing the whole dataset Data
  # with no influence from freshData
  xmodel = model.copy()
  xLP = dict()
  xLP['word_variational'] = np.hstack([LP['word_variational'],
                                       np.zeros((Data.nObs,freshK))])
  xLP['DocTopicCount'] = np.hstack([LP['DocTopicCount'],
                                      np.zeros((Data.nDoc,freshK))])
  xLP['digammasumTheta'] = LP['digammasumTheta'].copy()

  # New expected beta values via stick-breaking construction
  Ebeta = xmodel.allocModel.Ebeta
  remEbeta = Ebeta[-1]
  newEbeta = np.zeros(freshK)
  for k in range(freshK):
    newEbeta[k] = 1./(1.0 + alpha) * remEbeta
    remEbeta = remEbeta - newEbeta[k]
  xEbeta = np.hstack( [Ebeta[:-1], newEbeta, remEbeta])
  assert Ebeta.size == xEbeta.size - freshK
  assert np.allclose(xEbeta[:-(freshK+1)], Ebeta[:-1])
  assert np.allclose(xEbeta.sum(), 1.0)

  # Standard updates for theta, theta_u
  xLP['theta'] = xLP['DocTopicCount'] + gamma * xEbeta[:-1]
  assert xLP['theta'].shape[1] == LP['theta'].shape[1] + freshK
  assert np.allclose( xLP['theta'][:, :-freshK], LP['theta'])
  xLP['theta_u'] = gamma * xEbeta[-1]

  digammaSumTheta = digamma(xLP['theta_u'] + xLP['theta'].sum(axis=1))
  digammaSumTheta_orig = digamma(LP['theta_u'] + LP['theta'].sum(axis=1))
  assert np.allclose(digammaSumTheta_orig, digammaSumTheta)
  xLP['E_logPi'] = digamma(xLP['theta']) - digammaSumTheta[:,np.newaxis]
  xLP['E_logPi_u'] = digamma(xLP['theta_u']) - digammaSumTheta

  # Second, standard summary step creates expanded suff stats xSS from xLP.
  xSS = xmodel.get_global_suff_stats(Data, xLP)

  # Finally, add in freshSS
  xSS.sumLogPiActive[-freshK:] += freshSS.sumLogPiActive
  xSS.sumLogPiUnused += freshSS.sumLogPiUnused
  xSS.WordCounts[-freshK:] += freshSS.WordCounts
  xSS.N[-freshK:] += freshSS.N
  xSS.nDoc += freshSS.nDoc

  ## Update xmodel given xSS (which now include freshData)
  xmodel.update_global_params(xSS)
  return xmodel, xSS

def np2flatstr(xvec):
  return ' '.join( ['%9.3f' % (x) for x in xvec])



class Test_BarsK6V9(unittest.TestCase):

  def shortDescription(self):
    return None

  def setUp(self):
    Data = U.getBarsData('BarsK6V9')
    model, SS, LP = U.MakeModelWithTrueTopics(Data, alpha0=alpha, gamma=gamma)
    LP = model.calc_local_params(Data)
    SS = model.get_global_suff_stats(Data, LP)

    freshData = Data.get_random_sample(50, randstate=np.random.RandomState(0))
    freshModel, _, _ = U.MakeModelWithFiveTopics(freshData)
    freshLP = freshModel.calc_local_params(freshData)
    freshSS = freshModel.get_global_suff_stats(freshData, freshLP)

    #xmodelWRONG, xSSWRONG = createExpandedModel__viaSS(model, SS, freshSS)
    self.xmodel, self.xSS = createExpandedModel__viaLP(model, LP, freshSS, 
                                               Data, freshData)
    self.xmodelC, self.xSSC = createExpandedModel__viaSScorrected2(
                                               model, SS, freshSS)
    model.update_global_params(SS)

    self.model = model
    self.SS  = SS

  def test_equivalent_sumLogPiUnused(self):
    print ''
    print "orig        ", self.SS.sumLogPiUnused
    print "expanded_LP ", self.xSS.sumLogPiUnused
    print "expanded_SS ", self.xSSC.sumLogPiUnused
    assert np.allclose( self.xSS.sumLogPiUnused, 
                        self.xSSC.sumLogPiUnused)

  
  def test_equivalent_sumLogPiActive(self):
    print ''
    print "orig        ", np2flatstr(self.SS.sumLogPiActive)
    print "expanded_LP ", np2flatstr(self.xSS.sumLogPiActive)
    print "expanded_SS ", np2flatstr(self.xSSC.sumLogPiActive)
    assert np.allclose(self.xSS.sumLogPiActive, self.xSSC.sumLogPiActive)

  def test_equivalent_Ebeta(self):
    print ''
    print "orig        ", np2flatstr(self.model.allocModel.Ebeta)
    print "expanded_LP ", np2flatstr(self.xmodel.allocModel.Ebeta)
    print "expanded_SS ", np2flatstr(self.xmodelC.allocModel.Ebeta)
    assert np.allclose(self.xmodel.allocModel.Ebeta,
                       self.xmodelC.allocModel.Ebeta)
    # Test that all original active comps have not changed too much
    assert np.allclose(self.xmodel.allocModel.Ebeta[:-(5+1)], 
                       self.model.allocModel.Ebeta[:-1], atol=0.01)

  

