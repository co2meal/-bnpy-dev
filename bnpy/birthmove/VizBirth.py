
def viz_birth_proposal(curModel, propModel, birthCompIDs,
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
