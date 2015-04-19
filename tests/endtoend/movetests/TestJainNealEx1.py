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
        import JainNealEx1
        self.Data = JainNealEx1.get_data(nPerState=100)
        self.datasetArg = dict(
            name='JainNealEx1', 
            nPerState=100,
            )
    def nextObsKwArgsForVB(self):
        for lam in [0.5, 0.01]:
            kwargs = OrderedDict()
            kwargs['name'] = 'Bern'
            kwargs['lam1'] = lam
            kwargs['lam0'] = lam            
            yield kwargs

    def nextAllocKwArgsForVB(self):
        for gamma in [1.0, 50.0]:
            kwargs = OrderedDict()
            kwargs['name'] = 'DPMixtureModel'
            kwargs['gamma0'] = gamma
            yield kwargs
