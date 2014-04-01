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
  xSS = SS.copy()
  xmodel = model.copy()
  xSS.insertEmptyComps(1)
  xmodel.update_global_params(xSS)
  return xmodel, xSS


def createModelWithEmptyLastComp__viaSScorrected(model, SS, Data):
  xSS = SS.copy()
  xmodel = model.copy()
  xSS.insertEmptyComps(1)

  # Correction of sumLogPi term!
  nWordPerDoc = np.asarray(Data.to_sparse_docword_matrix().sum(axis=1))
  remEbeta = model.allocModel.Ebeta[-1]
  EbetaKp1 = alpha * priorFrac * remEbeta
  EbetaKp2 = alpha * (1-priorFrac) * remEbeta

  sumDigamma_Nalpha = np.sum(digamma(alpha + nWordPerDoc))
  xSS.sumLogPiActive[-1] = Data.nDoc * digamma(EbetaKp1) - sumDigamma_Nalpha
  xSS.sumLogPiUnused = Data.nDoc * digamma(EbetaKp2) - sumDigamma_Nalpha

  # Correction of ELBOterm ElogqPiUnused
  ElogqPiActive = xSS.getELBOTerm('ElogqPiActive')
  ElogqPiActive[-1] = (EbetaKp1-1) * xSS.sumLogPiActive[-1] \
                    - Data.nDoc * gammaln(EbetaKp1)
  ElogqPiUnused = (EbetaKp2-1) * xSS.sumLogPiUnused \
                    - Data.nDoc * gammaln(EbetaKp2)
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

  xEbeta = np.hstack( [Ebeta[:-1], priorFrac*Ebeta[-1], (1-priorFrac)*Ebeta[-1]])
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
  return ' '.join( ['%9.3f' % (x) for x in xvec])

Data = U.getBarsData('BarsK6V9')
model, SS, LP = U.MakeModelWithTrueTopics(Data)
LP = model.calc_local_params(Data)
SS = model.get_global_suff_stats(Data, LP, doPrecompEntropy=True)

xmodel, xSS = createModelWithEmptyLastComp__viaSS(model, SS)
xmodelC, xSSC = createModelWithEmptyLastComp__viaSScorrected(model, SS, Data)
xmodel2, xSS2 = createModelWithEmptyLastComp__viaLP(model, LP, Data)
model.update_global_params(SS)

print '................... alloc model E[beta]'
print "orig        ", np2flatstr(model.allocModel.Ebeta)
print "expanded_SS ", np2flatstr(xmodel.allocModel.Ebeta)
print "expanded_SSc", np2flatstr(xmodelC.allocModel.Ebeta)
print "expanded_LP ", np2flatstr(xmodel2.allocModel.Ebeta)


print '................... alloc model U1'
print "orig        ", np2flatstr(model.allocModel.U1)
print "expanded_SS ", np2flatstr(xmodel.allocModel.U1)
print "expanded_SSC", np2flatstr(xmodelC.allocModel.U1)
print "expanded_LP ", np2flatstr(xmodel2.allocModel.U1)
print '................... alloc model U0'
print "orig        ", np2flatstr(model.allocModel.U0)
print "expanded_SS ", np2flatstr(xmodel.allocModel.U0)
print "expanded_SSC", np2flatstr(xmodelC.allocModel.U0)
print "expanded_LP ", np2flatstr(xmodel2.allocModel.U0)

print '................... suff stats sumLogPiActive'
print "orig        ", np2flatstr(SS.sumLogPiActive)
print "expanded_SS ", np2flatstr(xSS.sumLogPiActive)
print "expanded_SSC", np2flatstr(xSSC.sumLogPiActive)
print "expanded_LP ", np2flatstr(xSS2.sumLogPiActive)

xELBO_SS = xmodel.obsModel.calc_evidence(None, xSS, None)
xELBO_SSC = xmodelC.obsModel.calc_evidence(None, xSSC, None)
xELBO_LP = xmodel2.obsModel.calc_evidence(None, xSS2, None)
curELBO = model.obsModel.calc_evidence(None, SS, None)

print '................... obs model ELBO'
print "%.9e" % (curELBO)
print "%.9e" % (xELBO_SS)
print "%.9e" % (xELBO_SSC)
print "%.9e" % (xELBO_LP)

xELBO_SS = xmodel.allocModel.calc_evidence(None, xSS, None)
xELBO_SSC = xmodelC.allocModel.calc_evidence(None, xSSC, None)
xELBO_LP = xmodel2.allocModel.calc_evidence(None, xSS2, None)
curELBO = model.allocModel.calc_evidence(None, SS, None)
print '................... alloc model ELBO'
print "%.9e" % (curELBO)
print "%.9e" % (xELBO_SS)
print "%.9e" % (xELBO_SSC)
print "%.9e" % (xELBO_LP)

Cdict = xmodelC.allocModel.calc_evidence(None, xSSC, None, todict=True)
for key in Cdict:
  print key, Cdict[key]

Rdict = xmodel2.allocModel.calc_evidence(None, xSS2, None, todict=True)
for key in Rdict:
  print key, Rdict[key]

"""

print '................... alloc model E[logpZ]'
print xSS2.getELBOTerm('ElogpZ')
print SS.getELBOTerm('ElogpZ')

print '................... alloc model E[logqZ]'
print xSS2.getELBOTerm('ElogqZ')
print SS.getELBOTerm('ElogqZ')
print "%.9e" % (xSS2.getELBOTerm('ElogpZ')[:-1].sum() - \
                xSS2.getELBOTerm('ElogqZ')[:-1].sum())
print "%.9e" % (SS.getELBOTerm('ElogpZ').sum() - \
                SS.getELBOTerm('ElogqZ').sum())

print '................... alloc model E[log q Pi] const'
print "expanded %.9e"  % (xSS2.getELBOTerm('ElogqPiConst'))
print "orig     %.9e" % (SS.getELBOTerm('ElogqPiConst'))

print '................... alloc model E[log q Pi] unused'
print "expanded %.9e" % (xSS2.getELBOTerm('ElogqPiUnused'))
print "orig     %.9e" % (SS.getELBOTerm('ElogqPiUnused'))

print '................... alloc model E[log q Pi] active'
print ' '.join(['% 9.3f' % (x) for x in xSS2.getELBOTerm('ElogqPiActive')])
print ' '.join(['% 9.3f' % (x) for x in SS.getELBOTerm('ElogqPiActive')])


print '................... alloc model E[log p(V) - log q(V)]'
print model.allocModel.E_logpV() - model.allocModel.E_logqV()
print xmodel2.allocModel.E_logpV() - xmodel2.allocModel.E_logqV()

print model.allocModel.U1
print xmodel2.allocModel.U1
"""
