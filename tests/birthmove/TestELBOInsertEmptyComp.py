from scipy.special import digamma
import numpy as np
import unittest

import bnpy
import UtilForBirthTest as U

np.set_printoptions(precision=3, suppress=True, linewidth=140)

def createModelWithEmptyLastComp__viaSS(model, SS):
  xSS = SS.copy()
  xmodel = model.copy()
  xSS.insertEmptyComps(1)
  xmodel.update_global_params(xSS)
  return xmodel, xSS

def createModelWithEmptyLastComp__viaLP(model, LP, Data):
  xLP = dict()
  xLP['word_variational'] = np.hstack([LP['word_variational'],
                                       np.zeros((Data.nObs,1))])
  xLP['DocTopicCount'] = np.hstack([LP['DocTopicCount'],
                                      np.zeros((Data.nDoc,1))])
  Ebeta = model.allocModel.Ebeta
  xEbeta = np.hstack( [Ebeta[:-1], 0.5*Ebeta[-1], 0.5*Ebeta[-1]])
  assert Ebeta.size == xEbeta.size - 1
  assert np.allclose(xEbeta.sum(), 1.0)

  xLP['theta'] = xLP['DocTopicCount'] + 0.5 * xEbeta[:-1]
  assert xLP['theta'].shape[1] == LP['theta'].shape[1] + 1

  assert np.allclose( xLP['theta'][:, :-1], LP['theta'])

  xLP['theta_u'] = 0.5 * xEbeta[-1]

  digammaSumTheta_orig = digamma(LP['theta_u'] + LP['theta'].sum(axis=1))
  digammaSumTheta = digamma(xLP['theta_u'] + xLP['theta'].sum(axis=1))
  assert np.allclose( digammaSumTheta_orig, digammaSumTheta)

  xLP['E_logPi'] = digamma(xLP['theta']) - digammaSumTheta[:,np.newaxis]
  xLP['E_logPi_u'] =   digamma(xLP['theta_u']) - digammaSumTheta

  xSS = model.get_global_suff_stats(Data, xLP, doPrecompEntropy=True)
  xmodel.update_global_params(xSS)
  return xmodel, xSS

Data = U.getBarsData('BarsK6V9')
model, SS, LP = U.MakeModelWithTrueTopics(Data)
LP = model.calc_local_params(Data)
SS = model.get_global_suff_stats(Data, LP, doPrecompEntropy=True)

xmodel, xSS = createModelWithEmptyLastComp__viaSS(model, SS)
xmodel2, xSS2 = createModelWithEmptyLastComp__viaLP(model, LP, Data)
for x in range(10):
  model.update_global_params(SS)

xELBO_SS = xmodel.obsModel.calc_evidence(None, xSS, None)
xELBO_LP = xmodel2.obsModel.calc_evidence(None, xSS2, None)
curELBO = model.obsModel.calc_evidence(None, SS, None)
print '................... obs model ELBO'
print "%.9e" % (curELBO)
print "%.9e" % (xELBO_SS)
print "%.9e" % (xELBO_LP)

xELBO_SS = xmodel.allocModel.calc_evidence(None, xSS, None)
xELBO_LP = xmodel2.allocModel.calc_evidence(None, xSS2, None)
curELBO = model.allocModel.calc_evidence(None, SS, None)
print '................... alloc model ELBO'
print "%.9e" % (curELBO)
print "%.9e" % (xELBO_SS)
print "%.9e" % (xELBO_LP)


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