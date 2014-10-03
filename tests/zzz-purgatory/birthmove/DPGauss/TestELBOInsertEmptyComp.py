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
  xLP['resp'] = np.hstack([LP['resp'],
                                       np.zeros((Data.nObs,1))])
  xSS = model.get_global_suff_stats(Data, xLP, doPrecompEntropy=True)
  xmodel.update_global_params(xSS)
  return xmodel, xSS

import AsteriskK8
Data = AsteriskK8.get_data()
model, SS, LP = U.MakeModelWithTrueComps(Data)
LP = model.calc_local_params(Data)
SS = model.get_global_suff_stats(Data, LP, doPrecompEntropy=True)

xmodel, xSS = createModelWithEmptyLastComp__viaSS(model, SS)
xmodel2, xSS2 = createModelWithEmptyLastComp__viaLP(model, LP, Data)
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