from scipy.special import digamma, gammaln
import numpy as np
import unittest

import bnpy
import UtilForBirthTest as U

np.set_printoptions(precision=3, suppress=True, linewidth=140)

gamma = 5.0
alpha = 0.5
priorFrac = 1./(1.0 + gamma)

def createModelWithEmptyLastComp__viaSS(model, SS):
  ''' WRONG Direct construction of empty comps.
  '''
  xSS = SS.copy()
  xmodel = model.copy()
  xSS.insertEmptyComps(1)
  xmodel.update_global_params(xSS)
  return xmodel, xSS

def createModelWithEmptyLastComp__viaSScorrected2(model, SS):
  xSS = SS.copy()
  xmodel = model.copy()
  assert xSS.hasELBOTerms()
  xSS, AI, RI = xmodel.allocModel.insertEmptyCompsIntoSuffStatBag( xSS, 1)
  xmodel.update_global_params(xSS)
  return xmodel, xSS

def createModelWithEmptyLastComp__viaSScorrected(model, SS, Data):
  xSS = SS.copy()
  xmodel = model.copy()
  xSS.insertEmptyComps(1)

  # Correction of sumLogPi term!
  nWordPerDoc = np.asarray(Data.to_sparse_docword_matrix().sum(axis=1))
  remEbeta = model.allocModel.Ebeta[-1]
  EbetaKp1 = priorFrac * remEbeta
  EbetaKp2 = (1-priorFrac) * remEbeta

  thetaKp1 = alpha * EbetaKp1
  thetaKp2 = alpha * EbetaKp2

  sumDigamma_Nalpha = np.sum(digamma(alpha + nWordPerDoc))
  xSS.sumLogPiActive[-1] = Data.nDoc * digamma(thetaKp1) - sumDigamma_Nalpha
  xSS.sumLogPiUnused = Data.nDoc * digamma(thetaKp2) - sumDigamma_Nalpha

  # Correction of ELBOterm ElogqPiUnused
  ElogqPiActive = xSS.getELBOTerm('ElogqPiActive')
  ElogqPiActive[-1] = (thetaKp1-1) * xSS.sumLogPiActive[-1] \
                    - Data.nDoc * gammaln(thetaKp1)
  ElogqPiUnused = (thetaKp2-1) * xSS.sumLogPiUnused \
                    - Data.nDoc * gammaln(thetaKp2)
  xSS.setELBOTerm('ElogqPiActive', ElogqPiActive, dims=('K')) 
  xSS.setELBOTerm('ElogqPiUnused', ElogqPiUnused, dims=None)

  xmodel.update_global_params(xSS)
  return xmodel, xSS

def createModelWithEmptyLastComp__viaLP(model, LP, Data):
  xmodel = model.copy()
  xLP = dict()
  xLP['word_variational'] = np.hstack([LP['word_variational'],
                                       np.zeros((Data.nObs,1))])
  xLP['DocTopicCount'] = np.hstack([LP['DocTopicCount'],
                                      np.zeros((Data.nDoc,1))])
  Ebeta = xmodel.allocModel.Ebeta

  xEbeta = np.hstack([Ebeta[:-1], 
                      priorFrac*Ebeta[-1],
                      (1-priorFrac)*Ebeta[-1]])
  assert Ebeta.size == xEbeta.size - 1
  assert np.allclose(xEbeta.sum(), 1.0)

  xLP['theta'] = xLP['DocTopicCount'] + alpha * xEbeta[:-1]
  assert xLP['theta'].shape[1] == LP['theta'].shape[1] + 1

  assert np.allclose( xLP['theta'][:, :-1], LP['theta'])

  xLP['theta_u'] = alpha * xEbeta[-1]

  digammaSumTheta_orig = digamma(LP['theta_u'] + LP['theta'].sum(axis=1))
  digammaSumTheta = digamma(xLP['theta_u'] + xLP['theta'].sum(axis=1))
  assert np.allclose( digammaSumTheta_orig, digammaSumTheta)

  xLP['E_logPi'] = digamma(xLP['theta']) - digammaSumTheta[:,np.newaxis]
  xLP['E_logPi_u'] =   digamma(xLP['theta_u']) - digammaSumTheta

  xSS = xmodel.get_global_suff_stats(Data, xLP, doPrecompEntropy=True)
  xmodel.update_global_params(xSS)
  return xmodel, xSS

def np2flatstr(xvec):
  return ' '.join( ['%9.5f' % (x) for x in xvec])


class TestBarsK6V9(unittest.TestCase):

  def shortDescription(self):
    return None

  def setUp(self):
    Data = U.getBarsData('BarsK6V9')
    model, SS, LP = U.MakeModelWithTrueTopics(Data, aModel='HDPModel',
                                              alpha0=gamma, gamma=alpha)
    LP = model.calc_local_params(Data)
    SS = model.get_global_suff_stats(Data, LP, doPrecompEntropy=True)

    self.xmodel, self.xSS = createModelWithEmptyLastComp__viaLP(
                                                  model, LP, Data)
    self.xmodelC, self.xSSC = createModelWithEmptyLastComp__viaSScorrected(
                                                  model, SS, Data)
    self.xmodelC2, self.xSSC2 = createModelWithEmptyLastComp__viaSScorrected2(
                                                  model, SS)
    model.update_global_params(SS)
    self.model = model

  def test_eq_sumLogPiActive(self):
    print ''
    print "LP direct    ", np2flatstr(self.xSS.sumLogPiActive)
    print "SS direct    ", np2flatstr(self.xSSC.sumLogPiActive)
    print "SS insertComp", np2flatstr(self.xSSC2.sumLogPiActive)
    assert np.allclose( self.xSS.sumLogPiActive, self.xSSC.sumLogPiActive)
    assert np.allclose( self.xSS.sumLogPiActive, self.xSSC2.sumLogPiActive)

  def test_eq_Ebeta(self):
    print ''
    print "LP direct    ", np2flatstr(self.xmodel.allocModel.Ebeta)
    print "SS direct    ", np2flatstr(self.xmodelC.allocModel.Ebeta)
    print "SS insertComp", np2flatstr(self.xmodelC2.allocModel.Ebeta)
    assert np.allclose( self.xmodel.allocModel.Ebeta, 
                        self.xmodelC.allocModel.Ebeta)
    assert np.allclose( self.xmodel.allocModel.Ebeta, 
                        self.xmodelC2.allocModel.Ebeta)
    # Verify beta weight for original topics has not changed much
    assert np.allclose( self.xmodel.allocModel.Ebeta[:-2], 
                        self.model.allocModel.Ebeta[:-1], atol=0.001)

  def test_eq_allocELBO__xFromLP_and_xFromSS(self):
    print ''
    def allocELBO(m, SS):
      return getattr(m,'allocModel').calc_evidence(None, SS, None)
    print "LP direct    ", allocELBO(self.xmodel, self.xSS)
    print "SS direct    ", allocELBO(self.xmodelC, self.xSSC)
    print "SS insertComp", allocELBO(self.xmodelC2, self.xSSC2)
    assert np.allclose( allocELBO(self.xmodel, self.xSS), 
                        allocELBO(self.xmodelC, self.xSSC))
    assert np.allclose( allocELBO(self.xmodel, self.xSS), 
                        allocELBO(self.xmodelC2, self.xSSC2))

  def test_eq_obsmodelELBO__orig_and_expanded(self):
    print ''
    def allocELBO(m, SS):
      return getattr(m,'obsModel').calc_evidence(None, SS, None)
    print "LP direct    ", allocELBO(self.xmodel, self.xSS)
    print "SS direct    ", allocELBO(self.xmodelC, self.xSSC)
    print "SS insertComp", allocELBO(self.xmodelC2, self.xSSC2)
    assert np.allclose( allocELBO(self.xmodel, self.xSS), 
                        allocELBO(self.xmodelC, self.xSSC))
    assert np.allclose( allocELBO(self.xmodel, self.xSS), 
                        allocELBO(self.xmodelC2, self.xSSC2))

class TestHDPModel2_BarsK6V9(TestBarsK6V9):

  def setUp(self):
    Data = U.getBarsData('BarsK6V9')
    model, SS, LP = U.MakeModelWithTrueTopics(Data, aModel='HDPModel2',
                                              alpha0=gamma, gamma=alpha)
    LP = model.calc_local_params(Data)
    SS = model.get_global_suff_stats(Data, LP, doPrecompEntropy=True)

    self.xmodel, self.xSS = createModelWithEmptyLastComp__viaLP(
                                                  model, LP, Data)
    self.xmodelC, self.xSSC = createModelWithEmptyLastComp__viaSScorrected(
                                                  model, SS, Data)
    self.xmodelC2, self.xSSC2 = createModelWithEmptyLastComp__viaSScorrected2(
                                                  model, SS)
    model.update_global_params(SS)
    self.model = model