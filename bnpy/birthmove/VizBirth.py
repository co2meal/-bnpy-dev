
def viz_birth_proposal(curModel, propModel, birthCompIDs, **kwargs):
  if str(type(curModel.obsModel)).count('Gauss') > 0:
    _viz_Gauss(curModel, propModel, birthCompIDs, **kwargs)
  else:
    _viz_Mult(curModel, propModel, birthCompIDs, **kwargs)


def _viz_Gauss(curModel, propModel, birthCompIDs,
              curELBO=None, propELBO=None, block=False, **kwargs):
  from ..viz import GaussViz
  from matplotlib import pylab
  pylab.figure()
  h=pylab.subplot(1,2,1)
  GaussViz.plotGauss2DFromHModel(curModel)
  h=pylab.subplot(1,2,2)
  GaussViz.plotGauss2DFromHModel(propModel, compsToHighlight=birthCompIDs)
  pylab.show(block=block)

def _viz_Mult(curModel, propModel, birthCompIDs,
              curELBO=None, propELBO=None, block=False, **kwargs):
  from ..viz import BarsViz
  from matplotlib import pylab
  pylab.figure()
  h=pylab.subplot(1,2,1)
  BarsViz.plotBarsFromHModel(curModel, figH=h)
  h=pylab.subplot(1,2,2)
  BarsViz.plotBarsFromHModel(propModel, compsToHighlight=birthCompIDs, figH=h)
  if curELBO is not None:
    pylab.xlabel("%.3e" % (propELBO - curELBO))
  pylab.show(block=block)
