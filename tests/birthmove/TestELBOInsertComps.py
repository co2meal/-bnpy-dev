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

np.set_printoptions(precision=3, suppress=True, linewidth=140)

gamma = 5.0
alpha = 0.5
priorFrac = 1./(1.0 + gamma)

def createExpandedModel__viaSS(model, SS, freshSS):
  xSS = SS.copy()
  xmodel = model.copy()
  xSS.insertComps(freshSS)
  xmodel.update_global_params(xSS)
  return xmodel, xSS

def createExpandedModel__viaSScorrected2(model, SS, freshSS, correctFresh=False):
  xSS = SS.copy()
  xmodel = model.copy()
  xSS = xmodel.allocModel.insertCompsIntoSuffStatBag(xSS, freshSS, correctFresh)

  xmodel.update_global_params(xSS)
  return xmodel, xSS

def createExpandedModel__viaLP(model, LP, freshLP, Data, freshData):
  freshK = freshLP['DocTopicCount'].shape[1]

  ## First, prepare an expanded-set of local params,
  # with no influence from the freshDataset (yet)
  xmodel = model.copy()
  xLP = dict()
  xLP['word_variational'] = np.hstack([LP['word_variational'],
                                       np.zeros((Data.nObs,freshK))])
  xLP['DocTopicCount'] = np.hstack([LP['DocTopicCount'],
                                      np.zeros((Data.nDoc,freshK))])
  xLP['digammasumTheta'] = LP['digammasumTheta'].copy()

  Ebeta = xmodel.allocModel.Ebeta
  remEbeta = Ebeta[-1]
  newEbeta = np.zeros(freshK)
  for k in range(freshK):
    newEbeta[k] = priorFrac * remEbeta
    remEbeta = remEbeta - newEbeta[k]
  xEbeta = np.hstack( [Ebeta[:-1], newEbeta, remEbeta])
  assert Ebeta.size == xEbeta.size - freshK
  assert np.allclose(xEbeta[:-(freshK+1)], Ebeta[:-1])
  assert np.allclose(xEbeta.sum(), 1.0)

  xLP['theta'] = xLP['DocTopicCount'] + alpha * xEbeta[:-1]
  assert xLP['theta'].shape[1] == LP['theta'].shape[1] + freshK
  assert np.allclose( xLP['theta'][:, :-freshK], LP['theta'])

  xLP['theta_u'] = alpha * xEbeta[-1]
  digammaSumTheta_orig = digamma(LP['theta_u'] + LP['theta'].sum(axis=1))
  digammaSumTheta = digamma(xLP['theta_u'] + xLP['theta'].sum(axis=1))
  assert np.allclose( digammaSumTheta_orig, digammaSumTheta)

  xLP['E_logPi'] = digamma(xLP['theta']) - digammaSumTheta[:,np.newaxis]
  xLP['E_logPi_u'] =   digamma(xLP['theta_u']) - digammaSumTheta

  xSS = xmodel.get_global_suff_stats(Data, xLP)

  ## Now, add in the fresh suff stats
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


Data = U.getBarsData('BarsK6V9')
model, SS, LP = U.MakeModelWithTrueTopics(Data)
LP = model.calc_local_params(Data)
SS = model.get_global_suff_stats(Data, LP)

freshData = Data.get_random_sample(50, randstate=np.random.RandomState(0))
freshModel, freshSS, freshLP = U.MakeModelWithFiveTopics(freshData)
freshLP = freshModel.calc_local_params(freshData)
freshSS = freshModel.get_global_suff_stats(freshData, freshLP)
#freshSS._Fields.setAllFieldsToZero()

xmodel, xSS = createExpandedModel__viaSS(model, SS, freshSS)
xmodel2, xSS2 = createExpandedModel__viaLP(model, LP, freshLP, Data, freshData)
xmodelC2, xSSC2 = createExpandedModel__viaSScorrected2(model, SS, freshSS)
xmodelC3, xSSC3 = createExpandedModel__viaSScorrected2(model, SS, freshSS, True)
model.update_global_params(SS)

nDoc = SS.nDoc + freshSS.nDoc
wc = Data.word_count.sum() + freshData.word_count.sum()
#assert U.verify_suffstats_at_desired_scale( xSS2,
#                                               nDoc = SS.nDoc + freshSS.nDoc,
#                                               word_count = wc)
#print xSSC2.nDoc, nDoc
#print xSSC2.WordCounts.sum(), wc
#U.verify_suffstats_at_desired_scale( xSS2, nDoc=nDoc, word_count=wc)
#U.verify_suffstats_at_desired_scale( xSSC2, nDoc=nDoc, word_count=wc)

class Test_BarsK6V9(unittest.TestCase):

  def shortDescription(self):
    return None

  def setUp(self):
    pass

  def test_equivalent_sumLogPiUnused(self):
    print ''
    print "orig        ", SS.sumLogPiUnused
    print "expanded_LP ", xSS2.sumLogPiUnused
    print "expanded_SS ", xSSC2.sumLogPiUnused
    print "expanded_SS3", xSSC3.sumLogPiUnused
    print "wrong       ", xSS.sumLogPiUnused
    #assert np.allclose( xSS2.sumLogPiActive, xSSC.sumLogPiActive)
    #assert np.allclose( xSS2.sumLogPiActive, xSSC2.sumLogPiActive)

  def test_equivalent_sumLogPiActive(self):
    print ''
    print "orig        ", np2flatstr(SS.sumLogPiActive)
    print "expanded_LP ", np2flatstr(xSS2.sumLogPiActive)
    print "expanded_SS ", np2flatstr(xSSC2.sumLogPiActive)
    print "expanded_SS3", np2flatstr(xSSC3.sumLogPiActive)
    print "wrong       ", np2flatstr(xSS.sumLogPiActive)
    #assert np.allclose( xSS2.sumLogPiActive, xSSC.sumLogPiActive)
    #assert np.allclose( xSS2.sumLogPiActive, xSSC2.sumLogPiActive)

  def test_equivalent_Ebeta(self):
    print ''
    print "orig        ", np2flatstr(model.allocModel.Ebeta)
    print "expanded_LP ", np2flatstr(xmodel2.allocModel.Ebeta)
    print "expanded_SS ", np2flatstr(xmodelC2.allocModel.Ebeta)
    print "expanded_SS3", np2flatstr(xmodelC3.allocModel.Ebeta)
    print "wrong       ", np2flatstr(xmodel.allocModel.Ebeta)
    #assert np.allclose( xmodel2.allocModel.Ebeta, xmodelC.allocModel.Ebeta)
    #assert np.allclose( xmodel2.allocModel.Ebeta, xmodelC2.allocModel.Ebeta)
    #assert np.allclose( xmodel2.allocModel.Ebeta[:-2], 
    #                    model.allocModel.Ebeta[:-1], atol=0.001)

