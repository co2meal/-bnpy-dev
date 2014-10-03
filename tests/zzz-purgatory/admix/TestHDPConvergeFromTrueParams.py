import numpy as np
import unittest

import bnpy
import UtilHDP as U
from matplotlib import pylab

Colors = ['r',  'k', 'b', 'g', 'c', 'm']
def plotdiffs(xs, cID, style='.-'):
  color = Colors[cID]
  pylab.plot( np.arange(len(xs)), xs, style, markeredgecolor=color, color=color)
  

class TestConvergeFromTrue(unittest.TestCase):
  def shortDescription(self):
    return None

  def setUp(self):
    Data = U.getBarsData('BarsK10V900')
    model, SS = U.MakeModelWithTrueTopics(Data, aModel='HDPModel2')
    self.Data = Data
    self.model = model

  def test_convergence(self):
    bnpy.viz.BarsViz.plotBarsFromHModel(self.model)
    pylab.show(block=True)
    self.trace_convergence(remMass=0.05, nIters=30)
    self.trace_convergence(remMass=0.01, nIters=30, style='+-')
    self.trace_convergence(remMass=0.001, nIters=30, style='--')

    pylab.show(block=False)
    from IPython import embed; embed()

  def trace_convergence(self, remMass=0.01, nIters=10, style='o-'):
    model = self.model.copy()
    Data = self.Data
    K = model.allocModel.K
    model.allocModel.Ebeta = (1-remMass)/float(K) * np.ones(K+1)
    model.allocModel.Ebeta[-1] = remMass

    traceN1 = np.zeros(nIters)
    traceN2 = np.zeros(nIters)
    traceN3 = np.zeros(nIters)
    traceTrem = np.zeros(nIters)
    traceBrem = np.zeros((nIters, K+1))
    traceELBO = np.zeros(nIters)
    traceA = np.zeros(nIters)
    traceZ = np.zeros(nIters)
    traceData = np.zeros(nIters)
    for ii in range(nIters):
      LP = model.calc_local_params(Data)
      traceBrem[ii] = model.allocModel.Ebeta.copy()
      if ii == 0:
        k1 = np.argmax(LP['DocTopicCount'][0,:])
        k2 = np.argmax(LP['DocTopicCount'][1,:])
        k3 = np.argmax(LP['DocTopicCount'][2,:])

      traceN1[ii] = LP['DocTopicCount'][0,k1]
      traceN2[ii] = LP['DocTopicCount'][1,k2]
      traceN3[ii] = LP['DocTopicCount'][2,k3]
      traceTrem[ii] = LP['theta_u']

      SS = model.get_global_suff_stats(Data, LP, doPrecompEntropy=1)
      model.update_global_params(SS)

      traceELBO[ii] = model.calc_evidence(SS=SS)
      elbo = model.calc_evidence(SS=SS, todict=1)
      traceA[ii] = elbo['v_Elogp'] - elbo['v_Elogq'] \
                 + elbo['pi_Elogp'] - elbo['pi_Elogq']
      traceZ[ii] = elbo['z_Elogp'] - elbo['z_Elogq'] 
      traceData[ii] = elbo['data_Elogp']

    pylab.figure(101)
    pylab.subplot(3, 1, 1)
    plotdiffs(traceN1, 0, style)
    plotdiffs(traceN2, 1, style)
    plotdiffs(traceN3, 2, style)
    pylab.xticks([])
    pylab.ylabel('Ndk doc-topic count', fontsize=13)

    pylab.subplot(3, 1, 2)
    plotdiffs(traceTrem, 0, style)
    pylab.xticks([])
    pylab.ylabel('theta rem', fontsize=13)

    pylab.subplot(3, 1, 3)
    plotdiffs( traceBrem[:,0], 0, style)
    plotdiffs( traceBrem[:,1], 1, style)
    plotdiffs( traceBrem[:,2], 2, style)
    plotdiffs( traceBrem[:,3], 3, style)
    pylab.ylabel('E[beta]', fontsize=13)

    pylab.figure(102)
    pylab.subplot(2, 1, 1)
    plotdiffs( traceELBO, 5, style)
    pylab.ylabel('ELBO', fontsize=13)

    pylab.subplot(2, 1, 2)
    plotdiffs( traceData, 5, style)
    pylab.ylabel('ELBO (data only)', fontsize=13)
