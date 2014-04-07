'''
Unit tests for BirthCreate.py

Verifies that births produce valid models with expected new components.

'''
import numpy as np
from scipy.special import digamma
import unittest

import bnpy
DMove = bnpy.deletemove.DeleteMoveBagOfWords

import UtilForDeleteTest as DU

class TestBarsK6V9(unittest.TestCase):
  def shortDescription(self):
    return None

  def setUp(self):
    self.dataName = 'BarsK6V9'
    mykwargs = dict(**DU.kwargs)
    self.kwargs = mykwargs

  def test_run_delete_move__true_components_remain(self):
    Data = DU.getBarsData(self.dataName)

    model, SS, LP = DU.MakeModelWithTrueTopics(Data)
    LP = model.calc_local_params(Data)
    SS = model.get_global_suff_stats(Data, LP, doPrecompEntropy=True)

    for betamethod in ['current', 'prior']:

      for cIDs in range(SS.K):

        newModel, newSS, Info = DMove.run_delete_move(Data, model, SS, LP,
                                          compIDs=cIDs, 
                                          betamethod=betamethod)
        assert Info['didRemoveComps'] == 0
        assert newSS.K == SS.K
        print Info['msg']

