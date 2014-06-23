'''
'''
import numpy as np
import unittest

import bnpy
import UtilForBirthTest as U
import CustomMOVBInspector

###########################################################
###########################################################
class TestHDPSuffStats(unittest.TestCase):

  def setUp(self):
    nLap = 2
    Data = U.getBarsData('BarsK6V9')
    self.model, _, _ = U.MakeModelWithFiveTopics(Data)
    DataIterator = bnpy.data.AdmixMinibatchIterator(Data, nBatch=4, nLap=nLap,
                                                    dataorderseed=42)
    self.DataIterator = DataIterator

    outP = dict(**U.outArgs)
    outP['customFuncPath'] = CustomMOVBInspector

    algP = dict(**U.algArgs)
    algP['nLap'] = nLap

    self.learnAlg = bnpy.learnalg.MemoizedOnlineVBLearnAlg(savedir=None, seed=0, 
                              algParams=algP, outputParams=outP)

  def test__global_summary_equals_sum_of_batch_summaries(self):
    print ''
    # CustomMOVBInspector is called within this learning algorithm,
    #  and conducts necessary tests internally to verify suff stats
    self.learnAlg.fit(self.model, self.DataIterator)


class TestHDPSuffStats_BirthMove(TestHDPSuffStats):

  def setUp(self):
    nLap = 5
    Data = U.getBarsData('BarsK6V9')
    self.model, _, _ = U.MakeModelWithFiveTopics(Data)
    DataIterator = bnpy.data.AdmixMinibatchIterator(Data, nBatch=4, nLap=nLap,
                                                    dataorderseed=42)
    self.DataIterator = DataIterator

    outP = dict(**U.outArgs)
    outP['customFuncPath'] = CustomMOVBInspector

    algP = dict(**U.algArgs)
    algP['nLap'] = nLap
    algP['birth'] = dict(**U.birthArgs)
    algP['birth']['birthPerLap'] = 2
    algP['birth']['birthRetainExtraMass'] = 1
    algP['birth']['expandAdjustSuffStats'] = 0
    self.learnAlg = bnpy.learnalg.MemoizedOnlineVBLearnAlg(savedir=None, seed=0, 
                              algParams=algP, outputParams=outP)
  

class TestHDPSuffStats_BirthMoveAtFirstTwoBatches(TestHDPSuffStats):

  def setUp(self):
    nLap = 5
    Data = U.getBarsData('BarsK6V9')
    self.model, _, _ = U.MakeModelWithFiveTopics(Data)
    DataIterator = bnpy.data.AdmixMinibatchIterator(Data, nBatch=5, nLap=nLap,
                                                    dataorderseed=42)
    self.DataIterator = DataIterator

    outP = dict(**U.outArgs)
    outP['customFuncPath'] = CustomMOVBInspector

    algP = dict(**U.algArgs)
    algP['nLap'] = nLap
    algP['birth'] = dict(**U.birthArgs)
    algP['birth']['Kfresh'] = 2
    algP['birth']['birthPerLap'] = 1
    algP['birth']['birthBatchFrac'] = 0.4
    algP['birth']['birthBatchLapLimit'] = nLap

    algP['birth']['birthRetainExtraMass'] = 1
    algP['birth']['expandAdjustSuffStats'] = 0
    self.learnAlg = bnpy.learnalg.MemoizedOnlineVBLearnAlg(savedir=None, seed=0, 
                              algParams=algP, outputParams=outP)
  
class TestHDPSuffStats_BirthMove_NoRetainExtraMass(TestHDPSuffStats):

  def setUp(self):
    nLap = 5
    Data = U.getBarsData('BarsK6V9')
    self.model, _, _ = U.MakeModelWithFiveTopics(Data)
    DataIterator = bnpy.data.AdmixMinibatchIterator(Data, nBatch=4, nLap=nLap,
                                                    dataorderseed=42)
    self.DataIterator = DataIterator

    outP = dict(**U.outArgs)
    outP['customFuncPath'] = CustomMOVBInspector

    algP = dict(**U.algArgs)
    algP['nLap'] = nLap
    algP['birth'] = dict(**U.birthArgs)
    algP['birth']['birthPerLap'] = 2
    algP['birth']['birthRetainExtraMass'] = 0
    algP['birth']['expandAdjustSuffStats'] = 0
    self.learnAlg = bnpy.learnalg.MemoizedOnlineVBLearnAlg(savedir=None, seed=0, 
                              algParams=algP, outputParams=outP)
  

class TestHDPSuffStats_BirthMove_Adjusted(TestHDPSuffStats):

  def setUp(self):
    nLap = 5
    Data = U.getBarsData('BarsK6V9')
    self.model, _, _ = U.MakeModelWithFiveTopics(Data)
    DataIterator = bnpy.data.AdmixMinibatchIterator(Data, nBatch=4, nLap=nLap,
                                                    dataorderseed=42)
    self.DataIterator = DataIterator

    outP = dict(**U.outArgs)
    outP['customFuncPath'] = CustomMOVBInspector

    algP = dict(**U.algArgs)
    algP['nLap'] = nLap
    algP['birth'] = dict(**U.birthArgs)
    algP['birth']['Kfresh'] = 5
    algP['birth']['birthPerLap'] = 1
    algP['birth']['birthRetainExtraMass'] = 0
    algP['birth']['expandAdjustSuffStats'] = 1
    self.learnAlg = bnpy.learnalg.MemoizedOnlineVBLearnAlg(savedir=None, seed=0, 
                              algParams=algP, outputParams=outP)


class TestHDPSuffStats_BirthMovex2_Adjusted(TestHDPSuffStats):

  def setUp(self):
    nLap = 5
    Data = U.getBarsData('BarsK6V9')
    self.model, _, _ = U.MakeModelWithFiveTopics(Data)
    DataIterator = bnpy.data.AdmixMinibatchIterator(Data, nBatch=4, nLap=nLap,
                                                    dataorderseed=42)
    self.DataIterator = DataIterator

    outP = dict(**U.outArgs)
    outP['customFuncPath'] = CustomMOVBInspector

    algP = dict(**U.algArgs)
    algP['nLap'] = nLap
    algP['birth'] = dict(**U.birthArgs)
    algP['birth']['Kfresh'] = 5
    algP['birth']['birthPerLap'] = 2
    algP['birth']['birthRetainExtraMass'] = 0
    algP['birth']['expandAdjustSuffStats'] = 1
    self.learnAlg = bnpy.learnalg.MemoizedOnlineVBLearnAlg(savedir=None, seed=0, 
                              algParams=algP, outputParams=outP)



class TestHDPSuffStats_BirthMoveWithDelete_Adjusted(TestHDPSuffStats):

  def setUp(self):
    nLap = 5
    Data = U.getBarsData('BarsK6V9')
    self.model, _, _ = U.MakeModelWithFiveTopics(Data)
    DataIterator = bnpy.data.AdmixMinibatchIterator(Data, nBatch=4, nLap=nLap,
                                                    dataorderseed=42)
    self.DataIterator = DataIterator

    outP = dict(**U.outArgs)
    outP['customFuncPath'] = CustomMOVBInspector

    algP = dict(**U.algArgs)
    algP['nLap'] = nLap
    algP['birth'] = dict(**U.birthArgs)
    algP['birth']['Kfresh'] = 5
    algP['birth']['birthPerLap'] = 1
    algP['birth']['birthRetainExtraMass'] = 0
    algP['birth']['expandAdjustSuffStats'] = 1
    algP['birth']['cleanupDeleteEmpty'] = 1
    algP['birth']['cleanupDeleteToImprove'] = 1
    self.learnAlg = bnpy.learnalg.MemoizedOnlineVBLearnAlg(savedir=None, seed=0, 
                              algParams=algP, outputParams=outP)

  

class TestHDPSuffStats_BirthMoveAtFirstTwoBatches_Adjusted(TestHDPSuffStats):

  def setUp(self):
    nLap = 5
    Data = U.getBarsData('BarsK6V9')
    self.model, _, _ = U.MakeModelWithFiveTopics(Data)
    DataIterator = bnpy.data.AdmixMinibatchIterator(Data, nBatch=5, nLap=nLap,
                                                    dataorderseed=42)
    self.DataIterator = DataIterator

    outP = dict(**U.outArgs)
    outP['customFuncPath'] = CustomMOVBInspector

    algP = dict(**U.algArgs)
    algP['nLap'] = nLap
    algP['birth'] = dict(**U.birthArgs)
    algP['birth']['Kfresh'] = 2
    algP['birth']['birthPerLap'] = 1
    algP['birth']['birthBatchFrac'] = 0.4
    algP['birth']['birthBatchLapLimit'] = nLap

    algP['birth']['birthRetainExtraMass'] = 0
    algP['birth']['expandAdjustSuffStats'] = 1
    algP['birth']['cleanupDeleteEmpty'] = 1
    algP['birth']['cleanupDeleteToImprove'] = 1
    self.learnAlg = bnpy.learnalg.MemoizedOnlineVBLearnAlg(savedir=None, seed=0, 
                              algParams=algP, outputParams=outP)


class TestHDPSuffStats_BirthMovex3_AtFirstTwoBatches_Adjusted(TestHDPSuffStats):

  def setUp(self):
    nLap = 5
    Data = U.getBarsData('BarsK6V9')
    self.model, _, _ = U.MakeModelWithFiveTopics(Data)
    DataIterator = bnpy.data.AdmixMinibatchIterator(Data, nBatch=5, nLap=nLap,
                                                    dataorderseed=42)
    self.DataIterator = DataIterator

    outP = dict(**U.outArgs)
    outP['customFuncPath'] = CustomMOVBInspector

    algP = dict(**U.algArgs)
    algP['nLap'] = nLap
    algP['birth'] = dict(**U.birthArgs)
    algP['birth']['Kfresh'] = 2
    algP['birth']['birthPerLap'] = 3
    algP['birth']['birthBatchFrac'] = 0.4
    algP['birth']['birthBatchLapLimit'] = nLap

    algP['birth']['birthRetainExtraMass'] = 0
    algP['birth']['expandAdjustSuffStats'] = 1
    algP['birth']['cleanupDeleteEmpty'] = 0
    algP['birth']['cleanupDeleteToImprove'] = 0
    self.learnAlg = bnpy.learnalg.MemoizedOnlineVBLearnAlg(savedir=None, seed=0, 
                              algParams=algP, outputParams=outP)
