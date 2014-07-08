'''
'''
import numpy as np
import unittest
import sys
import os

import bnpy
import CustomHook_VerifyELBOTracking as CustomHook

testsdir = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])
sys.path.append(os.path.join(testsdir,'base'))

print testsdir
import DPGaussBase

OutArgs = dict(traceEvery=1, 
               saveEvery=0,
               printEvery=1,
               customFuncPath=None,
               customFuncArgs=None)

AlgArgs = dict(doFullPassBeforeMstep=0,
            convergeSigFig=6,
            startLap=0,
            doMemoizeLocalParams=1,
            nCoordAscentItersLP=10,
            convThrLP=0.01,
            doShowSeriousWarningsOnly=1)

###########################################################
###########################################################
class TestMOVBELBOTracking_DPGauss(unittest.TestCase):

  def test_K1D1(self):
    self.verify_single_run( 'K1D1', 'true')

  def test_K1D2(self):
    self.verify_single_run( 'K1D2', 'true')


  def verify_single_run(self, dName, mName, **kwargs):
    print ''
    Data = DPGaussBase.MakeData(dName)
    model, SS, LP = DPGaussBase.MakeModel(mName, Data, **kwargs)

    outP = dict(**OutArgs)
    outP['customFuncPath'] = CustomHook

    algP = dict(**AlgArgs)
    algP['nLap'] = 5
    
    DataIterator = Data.to_minibatch_iterator(nBatch=4, nLap=algP['nLap'],
                                              dataorderseed=25)
    learnAlg = bnpy.learnalg.MOVBAlg(savedir=None, seed=0, 
                              algParams=algP, outputParams=outP)

    # CustomHook is called within this learning algorithm,
    #  and conducts necessary tests internally to verify suff stats
    learnAlg.fit(model, DataIterator)


