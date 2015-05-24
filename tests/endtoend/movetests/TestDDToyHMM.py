import numpy as np
import unittest
from collections import OrderedDict

import bnpy
from MergeMoveEndToEndTest import MergeMoveEndToEndTest


class Test(MergeMoveEndToEndTest):
    __test__ = True

    def setUp(self):
        """ Create the dataset
        """
        self.datasetArg = dict(
            name='DDToyHMM', 
            nDocTotal=16, 
            nPerDoc=500,
            )
        datasetMod = __import__(self.datasetArg['name'], fromlist=[])
        self.Data = datasetMod.get_data(**self.datasetArg)

    def nextObsKwArgsForVB(self):
        for sF in [0.1]:
            for ECovMat in ['eye']:
                kwargs = OrderedDict()
                kwargs['name'] = 'Gauss'
                kwargs['ECovMat'] = ECovMat
                kwargs['sF'] = sF
                yield kwargs

    def nextAllocKwArgsForVB(self):
        alpha = 0.5
        startAlpha = 10
        for gamma in [10.0]:
            for hmmKappa in [0, 50.0]:
                kwargs = OrderedDict()
                kwargs['name'] = 'HDPHMM'
                kwargs['gamma'] = gamma
                kwargs['alpha'] = alpha
                kwargs['hmmKappa'] = hmmKappa
                kwargs['startAlpha'] = startAlpha
                yield kwargs

    def test_initStateSeq(self):
        from bnpy.init.SingleSeqStateCreator import initSingleSeq_SeqAllocContigBlocks
        for aKwArgs in self.nextAllocKwArgsForVB():
            for oKwArgs in self.nextObsKwArgsForVB():
                hmodel = bnpy.HModel.CreateEntireModel(
                    'VB', aKwArgs['name'], oKwArgs['name'],
                    aKwArgs, oKwArgs, self.Data)
                SS = None
                for n in range(self.Data.nDoc):
                    hmodel, SS = initSingleSeq_SeqAllocContigBlocks(
                        n, self.Data, hmodel,
                        SS=SS,
                        verbose=1)
                assert SS.K >= self.Data.TrueParams['K']