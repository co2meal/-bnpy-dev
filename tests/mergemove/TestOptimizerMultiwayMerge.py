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
    sys.path.append('/Users/mhughes/git/bnpy2/bnpy/obsmodel2/')
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

  def test_dummy(self):
    SS = self.SS
    obsM = self.obsM

    kdel = 0
    alph, f, Info = OMM.find_optimum(SS, kdel, self.aFunc, self.oFunc)

    alphx = np.zeros(SS.K)
    alphx[:kdel] = alph[:kdel]
    alphx[kdel+1:] = alph[kdel:]

    # now actually try the merge
    propSS = SS.copy()
    propSS.multiMergeComps(kdel, alphx)
    propM = copy.deepcopy(obsM)
    propM.updatePost(propSS)

    elboBEFORE = obsM.calcELBO_Memoized(SS, doFast=1)
    elboAFTER = propM.calcELBO_Memoized(propSS, doFast=1)
    elboDelta = elboAFTER - elboBEFORE
    elboDelta2 = obsM.calcMergeELBO(SS, kdel, alphx)
    print elboDelta
    print elboDelta2
    print self.oFunc(SS, kdel, alphx)
