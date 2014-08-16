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
  
  def test_run_delete_move__remove_single_empty_junk_topics(self):
    Data = DU.getBarsData(self.dataName)

    model, SS, LP = DU.MakeModelWithTrueTopics(Data)
    TrueK = SS.K

    # Add some empty junk!
    SS.insertEmptyComps(2)
    model.update_global_params(SS)

    LP = model.calc_local_params(Data)
    SS = model.get_global_suff_stats(Data, LP, doPrecompEntropy=True)

    for betamethod in ['current', 'prior']:

      for cIDs in range(TrueK, SS.K):

        newModel, newSS, Info = DMove.run_delete_move(Data, model, SS, LP,
                                          compIDs=cIDs, 
                                          betamethod=betamethod)
        assert Info['didRemoveComps'] == 1
        assert newSS.K == SS.K - 1
        print Info['msg']

        newLP = newModel.calc_local_params(Data)
        newSS = newModel.get_global_suff_stats(Data, newLP)
        evBound = newModel.calc_evidence(Data, newSS, newLP)
        print 'proposed ELBO %.4e' % (Info['elbo'])
        print 'revised  ELBO %.4e' % (evBound)
        assert evBound > Info['elbo']

  def test_run_delete_move__remove_many_empty_junk_topics_at_once(self):
    print ''
    Data = DU.getBarsData(self.dataName)
    model, SS, LP = DU.MakeModelWithTrueTopics(Data)
    TrueK = SS.K

    # Add some empty junk!
    SS.insertEmptyComps(4)
    model.update_global_params(SS)

    LP = model.calc_local_params(Data)
    SS = model.get_global_suff_stats(Data, LP, doPrecompEntropy=True)

    for betamethod in ['current', 'prior']:

      cIDs = range(TrueK, SS.K)

      newModel, newSS, Info = DMove.run_delete_move(Data, model, SS, LP,
                                          compIDs=cIDs, 
                                          betamethod=betamethod)
      assert Info['didRemoveComps'] == 1
      print betamethod, Info['msg']


      newLP = newModel.calc_local_params(Data)
      newSS = newModel.get_global_suff_stats(Data, newLP)
      evBound = newModel.calc_evidence(Data, newSS, newLP)
      print 'proposed ELBO %.4e' % (Info['elbo'])
      print 'revised  ELBO %.4e' % (evBound)
      assert evBound > Info['elbo']


  
  def test_run_delete_move__remove_many_random_junk_topics_at_once(self):
    print ''
    Data = DU.getBarsData(self.dataName)
    model, SS, LP = DU.MakeModelWithTrueTopics(Data)
    for seed in range(5):
      self.run_delete_move__verify_junk_removed(Data, model, SS, seed=seed)
    for seed in range(1234, 1234+10):
      self.run_delete_move__verify_junk_removed(Data, model, SS, seed=seed)


  def run_delete_move__verify_junk_removed(self, Data, model, SS, seed=0,
                                                betamethod='current'):
    print '------------- seed=', seed
    LP = model.calc_local_params(Data)
    SS = model.get_global_suff_stats(Data, LP, doPrecompEntropy=True)

    TrueK = SS.K
    PRNG = np.random.RandomState(seed)
    from bnpy.birthmove.BirthMove import run_birth_move
    freshData = Data.get_random_sample(100, randstate=PRNG)
    birthargs = dict(Kfresh=4, Kmax=25,
                     creationRoutine='randexamples',
                     randstate=PRNG, 
                     expandOrder='expandThenRefine',
                     expandAdjustSuffStats=0,
                     refineNumIters=15,
                     cleanupDeleteEmpty=0,
                     cleanupDeleteToImprove=0,
                     birthRetainExtraMass=0,
                     birthVerifyELBOIncrease=0,
                     ) 
    xmodel, xSS, Info = run_birth_move(model, SS, freshData, **birthargs)

    # Redo local/global steps to make all consistent
    xLP = xmodel.calc_local_params(Data)
    xSS = xmodel.get_global_suff_stats(Data, xLP)
    xmodel.update_global_params(xSS)
    xLP = xmodel.calc_local_params(Data)
    xSS = xmodel.get_global_suff_stats(Data, xLP, doPrecompEntropy=True)
    
    print SS.N
    print xSS.N

    cIDs = range(TrueK, xSS.K)

    newModel, newSS, Info = DMove.run_delete_move(Data, xmodel, xSS, xLP,
                                          compIDs=cIDs, 
                                          betamethod=betamethod)

    print newSS.N
    print Info['msg']
    print 'Orig ELBO:', model.calc_evidence(SS=SS)
    
    print model.allocModel.Ebeta
    print xmodel.allocModel.Ebeta

    propModel = Info['propModel']
    propSS = Info['propSS']

    assert Info['didRemoveComps'] == 1
    assert newSS.K == TrueK


class TestBarsK10V900(TestBarsK6V9):
  def shortDescription(self):
    return None

  def setUp(self):
    self.dataName = 'BarsK10V900'
    mykwargs = dict(**DU.kwargs)
    self.kwargs = mykwargs
