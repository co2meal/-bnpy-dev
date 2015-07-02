import argparse
import numpy as np
import bnpy
from matplotlib import pylab

from bnpy.viz import GaussViz

DefaultPlan = dict(
    targetCompID=0,
    bcreationProposalName='randomSplit',
    targetMaxSize=200)

def showBirthProposal(
        curModel=None, propModel=None,
        Plan=None,
        origK=0,
        propK=0,
        **kwargs):
    ''' Show before/after images of learned comps for a birth proposal.

    Post Condition
    --------------
    Figure is displayed, but not blocking execution.
    '''
    compsToHighlight = np.arange(origK, propK)
    bnpy.viz.PlotComps.plotCompsFromHModel(
        hmodel=propModel, compsToHighlight=compsToHighlight)
    pylab.show(block=0)


def showBirthBeforeAfter(**kwargs):
    ''' Show before/after images of learned comps for a birth proposal.

    Post Condition
    --------------
    Figure is displayed, but not blocking execution.
    '''
    if str(type(kwargs['curModel'].obsModel)).count('Gauss') > 0:
        _viz_Gauss_before_after(**kwargs)
    else:
        _viz_Mult(**kwargs)
    pylab.show(block=0)


def _viz_Gauss_before_after(
        curModel=None, propModel=None,
        curSS=None, propSS=None,
        Plan=None,
        propLscore=None, curLscore=None,
        Data_b=None, Data_t=None, 
        **kwargs):
    pylab.subplots(
        nrows=1, ncols=2, figsize=(8, 4))
    h1 = pylab.subplot(1, 2, 1)
    GaussViz.plotGauss2DFromHModel(
        curModel, compsToHighlight=Plan['targetCompID'], figH=h1)
    pylab.title('%.4f' % (curLscore))

    h2 = pylab.subplot(1, 2, 2, sharex=h1, sharey=h1)
    newCompIDs = np.arange(curModel.obsModel.K, propModel.obsModel.K)
    GaussViz.plotGauss2DFromHModel(
        propModel, compsToHighlight=newCompIDs, figH=h2)
    pylab.title('%.4f' % (propLscore))
    
    Lgain = propLscore - curLscore
    if Lgain > 0:
        pylab.xlabel('ACCEPT +%.2f' % (Lgain))
    else:
        pylab.xlabel('REJECT %.2f' % (Lgain))
    pylab.tight_layout()

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
