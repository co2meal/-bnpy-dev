import argparse
import numpy as np
import bnpy
from matplotlib import pylab

from bnpy.viz import GaussViz

DefaultPlan = dict(
    targetCompID=0,
    bcreationProposalName='randomSplit',
    targetMaxSize=200)

def showBirthBeforeAfter(
        curModel=None, propModel=None,
        curSS=None, propSS=None,
        Plan=None,
        Data_b=None, Data_t=None, **kwargs):
    ''' Show before/after images of learned comps for a birth proposal.

    Post Condition
    --------------
    Figure is displayed, but not blocking execution.
    '''
    if str(type(curModel.obsModel)).count('Gauss') > 0:
        _viz_Gauss_before_after(**locals())
    else:
        _viz_Mult(**locals())
    pylab.show(block=0)


def _viz_Gauss_before_after(
        curModel=None, propModel=None,
        curSS=None, propSS=None,
        Plan=None,
        Data_b=None, Data_t=None, 
        **kwargs):
    pylab.subplots(nrows=1, ncols=2, figsize=(8, 4))
    h = pylab.subplot(1, 2, 1)
    GaussViz.plotGauss2DFromHModel(
        curModel, compsToHighlight=Plan['targetCompID'], figH=h)
    h = pylab.subplot(1, 2, 2)
    newCompIDs = np.arange(curModel.obsModel.K, propModel.obsModel.K)
    GaussViz.plotGauss2DFromHModel(
        propModel, compsToHighlight=newCompIDs, figH=h)

def showBirthFromSavedPath():
    '''
    '''
    pass


def showBirthFromScratch(hmodel=None, Data=None, **Plan):
    '''
    '''
    from BMain import runBirthMove

    LP = hmodel.calc_local_params(Data)
    propLP = runBirthMove(
        Data, hmodel, None, LP, **Plan)

if __name__ == '__main__':
    hmodel, Info = bnpy.run() # will auto-scrape stdin for all args

    Plan = dict(**DefaultPlan)
    Plan.update(Info['UnkArgs'])

    showBirthFromScratch(hmodel, Data=Info['Data'], **Plan)
