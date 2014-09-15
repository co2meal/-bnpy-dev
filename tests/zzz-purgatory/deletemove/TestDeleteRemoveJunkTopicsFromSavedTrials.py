'''
Unit tests for BirthCreate.py

Verifies that births produce valid models with expected new components.

'''
import numpy as np
from scipy.special import digamma
import unittest
import os

import bnpy
DMove = bnpy.deletemove.DeleteMoveBagOfWords

import UtilForDeleteTest as DU

def load_data_and_model():
  import BarsK50V2500
  Data = BarsK50V2500.get_data()
  modelpath = '/data/liv/liv-x/topic_models/results/bnpy/BarsK50V2500/HDPModel/Mult/moVB/quebec-K1-datadriven-allwords-smooth1-fixoutlier1-marglik'
  model, lap = bnpy.ioutil.ModelReader.loadModelForLap(os.path.join(modelpath,'1'), 20)
  return model, Data  

class TestBarsK50V2500(unittest.TestCase):
  def shortDescription(self):
    return None

  def setUp(self):
    pass
  
  def test_run_delete_move(self):
    model, Data = load_data_and_model()
    LP = model.calc_local_params(Data)
    SS = model.get_global_suff_stats(Data, LP)
    sortIDs = np.argsort(SS.N)
    smallest = [x for x in sortIDs[:20]]

    newModel, newSS, Info = DMove.run_delete_move(Data, model, SS, LP, compIDs=smallest)
    print Info['msg']

