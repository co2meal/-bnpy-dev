import numpy as np
import unittest
import sys
import copy

import bnpy
import StandardNormalK1

from bnpy.mergemove import OptimizerMultiwayMerge as OMM

class TestOptimizerMultiwayMerge(unittest.TestCase):
  
  def setUp(self):
    ''' Create allocmodel and obsmodel that need a merge
    '''
    Data = StandardNormalK1.get_data()
    model, _, _ = bnpy.run(Data, 'DPMixModel', 'Gauss', 'VB',
                      K=5, initname='randexamples',
                      nLap=25, printEvery=0, saveEvery=0)
    LP = model.calc_local_params(Data)
    SS = model.get_global_suff_stats(Data, LP)
    model.update_global_params(SS)

    # Now make the "new version" obsmodel
    sys.path.append('/data/liv/liv-x/bnpy/bnpy/obsmodel2/')
    import GaussObsModel
    obsM = GaussObsModel.GaussObsModel('VB', D=Data.dim, 
                                        ECovMat='eye')
    obsM.Prior.nu = model.obsModel.obsPrior.dF
    obsM.Prior.B = model.obsModel.obsPrior.invW
    obsM.Prior.m = model.obsModel.obsPrior.m
    obsM.Prior.kappa = model.obsModel.obsPrior.kappa

    obsM.updatePost(SS)
    aFunc = model.allocModel.calcMergeELBO_alph
    oFunc = obsM.calcMergeELBO_alph

    self.model = model
    self.obsM = obsM
    self.SS = SS
    self.aFunc = aFunc
    self.oFunc = oFunc

  def test__find_optimum(self):
    ''' Find optimal blend vector alph
    '''
    SS = self.SS
    kdel = 0
    alph, f, Info = OMM.find_optimum(SS, kdel, self.aFunc, self.oFunc)

    print alph
    from IPython import embed; embed()

  def test_calcMergeELBO__allocmodel(self):
    ''' For the allocmodel, calc merge ELBO two ways, make sure they both agree
        method A : make orig and merge models, call calcELBO() on both, and calc diff
        method B : call the calcMergeELBO() method directly on orig model
    '''
    SS = self.SS
    aM = self.model.allocModel
    SS.setELBOTerm('ElogqZ', np.zeros(SS.K), dims=('K'))

    kdel = 0
    alph = np.asarray([.5, .1, .2,.2])

    alphx = np.zeros(SS.K)
    alphx[:kdel] = alph[:kdel]
    alphx[kdel+1:] = alph[kdel:]

    propSS = SS.copy()
    propSS.multiMergeComps(kdel, alphx)
    propM = copy.deepcopy(aM)
    propM.update_global_params(propSS)

    elboBEFORE = aM.calc_evidence(None, SS, None)
    elboAFTER = propM.calc_evidence(None, propSS, None)
    aELBODelta = elboAFTER - elboBEFORE

    aELBODelta2 = aM.calcMergeELBO(SS, kdel, alphx)
    print aELBODelta
    print aELBODelta2
    #assert np.allclose(oELBODelta, oELBODelta2)

  def test_calcMergeELBO__obsmodel(self):
    ''' For the obsmodel, calculate merge ELBO two ways, make sure they both agree
        method A : make orig and merge models, call calcELBO() on both, and calc diff
        method B : call the calcMergeELBO() method directly on orig model
    '''
    SS = self.SS
    obsM = self.obsM

    kdel = 0
    alph = np.asarray([.5, .5, 0, 0])

    alphx = np.zeros(SS.K)
    alphx[:kdel] = alph[:kdel]
    alphx[kdel+1:] = alph[kdel:]

    # now actually try the merge
    propSS = SS.copy()
    propSS.multiMergeComps(kdel, alphx)
    propM = copy.deepcopy(obsM)
    propM.updatePost(propSS)

    elboBEFORE = obsM.calcELBO_Memoized(SS, doFast=0)
    elboAFTER = propM.calcELBO_Memoized(propSS, doFast=0)
    oELBODelta = elboAFTER - elboBEFORE
    oELBODelta2 = obsM.calcMergeELBO(SS, kdel, alphx)
    print oELBODelta
    print oELBODelta2
    assert np.allclose(oELBODelta, oELBODelta2)

    # Try with version1 obsmodel
    propModel = copy.deepcopy(self.model.obsModel)
    propModel.update_global_params(propSS)
    elboBEFORE1 = self.model.obsModel.calc_evidence(None, SS, None)
    elboAFTER1 = propModel.calc_evidence(None, propSS, None)
    print elboAFTER1 - elboBEFORE1

    print elboBEFORE1
    print elboBEFORE

    print elboAFTER1
    print elboAFTER
