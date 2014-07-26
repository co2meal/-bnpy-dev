import numpy as np

def viz_birth_proposal(curModel, propModel, Plan, **kwargs):
  if str(type(curModel.obsModel)).count('Gauss') > 0:
    _viz_Gauss(curModel, propModel, Plan['ktarget'], **kwargs)
  else:
    _viz_Mult(curModel, propModel, Plan, **kwargs)


def _viz_Gauss(curModel, propModel, ktarget,
              curELBO=None, propELBO=None, block=False, **kwargs):
  from ..viz import GaussViz
  from matplotlib import pylab
  pylab.figure()
  h=pylab.subplot(1,2,1)
  GaussViz.plotGauss2DFromHModel(curModel, compsToHighlight=ktarget)
  h=pylab.subplot(1,2,2)
  newCompIDs = np.arange(curModel.obsModel.K, propModel.obsModel.K)
  GaussViz.plotGauss2DFromHModel(propModel, compsToHighlight=newCompIDs)
  pylab.show(block=block)

def _viz_Mult(curModel, propModel, ktarget,
              curELBO=None, propELBO=None, block=False, **kwargs):
  from ..viz import BarsViz
  from matplotlib import pylab
  pylab.figure()
  h=pylab.subplot(1,2,1)
  BarsViz.plotBarsFromHModel(curModel, compsToHighlight=ktarget, figH=h)
  h=pylab.subplot(1,2,2)
  newCompIDs = np.arange(curModel.obsModel.K, propModel.obsModel.K)
  BarsViz.plotBarsFromHModel(propModel, compsToHighlight=newCompIDs, figH=h)
  if curELBO is not None:
    pylab.xlabel("%.3e" % (propELBO - curELBO))
  pylab.show(block=block)
