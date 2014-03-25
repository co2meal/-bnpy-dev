'''
Unit tests for BirthCreate.py

Verifies that births produce valid models with expected new components.

'''
import numpy as np
import unittest

import bnpy
BirthRefine = bnpy.birthmove.BirthRefine
BirthCreate = bnpy.birthmove.BirthCreate
BirthProposalError = bnpy.birthmove.BirthProposalError

''' Create simple toy bars dataset for testing.
'''
import BarsK6V9
Data = BarsK6V9.get_data(nDocTotal=100)

kwargs = dict(creationroutine='randexamples', cleanupMinSize=25,
               expandorder='expandthenrefine', refineNumIters=10,
               Kfresh=10, randstate=np.random.RandomState(0),
               cleanupDeleteEmpty=1, cleanupDeleteToImprove=0
             )

class TestBirthMove(unittest.TestCase):
  def shortDescription(self):
    return None

  def setUp(self):
    pass

  def MakeModelWithTrueTopics(self):
    ''' Create new model.
    '''
    aDict = dict(alpha0=5.0, gamma=0.5)
    oDict = {'lambda':0.1}
    hmodel = bnpy.HModel.CreateEntireModel('VB', 'HDPModel', 'Mult', 
                                            aDict, oDict, Data)
    hmodel.init_global_params(Data, initname='trueparams')
    LP = hmodel.calc_local_params(Data)
    SS = hmodel.get_global_suff_stats(Data, LP)
    hmodel.update_global_params(SS)
    return hmodel, SS, LP

  def MakeModelWithOneTopic(self):
    ''' Create new model.
    '''
    aDict = dict(alpha0=5.0, gamma=0.5)
    oDict = {'lambda':0.1}
    hmodel = bnpy.HModel.CreateEntireModel('VB', 'HDPModel', 'Mult', 
                                            aDict, oDict, Data)
    hmodel.init_global_params(Data, K=1, initname='randomfromprior',
                                    seed=0)
    LP = hmodel.calc_local_params(Data)
    SS = hmodel.get_global_suff_stats(Data, LP)
    hmodel.update_global_params(SS)
    return hmodel, SS, LP

  ######################################################### create_model
  #########################################################
  def test_create_model_with_new_comps(self):
    model, SS, LP = self.MakeModelWithOneTopic()
    
    freshModel, freshSS = BirthCreate.create_model_with_new_comps(
                                          model, SS, Data, **kwargs)
    assert freshModel.obsModel.K == freshSS.K
    assert freshSS.K > SS.K
    assert freshSS.K == kwargs['Kfresh']

  def test_create_model_with_new_comps_cleanupMinSize(self):
    model, SS, LP = self.MakeModelWithOneTopic()
    mykwargs = dict(**kwargs)
    mykwargs['cleanupDeleteEmpty'] = 1
    mykwargs['cleanupMinSize'] = 10000
    with self.assertRaises(BirthProposalError):
      freshModel, freshSS = BirthCreate.create_model_with_new_comps(
                                          model, SS, Data, **mykwargs)


  ######################################################### refine_model
  #########################################################
  def test_expand_then_refine(self):
    model, SS, LP = self.MakeModelWithOneTopic()  
    freshModel, freshSS = BirthCreate.create_model_with_new_comps(
                                          model, SS, Data, **kwargs)
    mykwargs = dict(**kwargs)
    mykwargs['refineNumIters'] = 1
    mykwargs['cleanupDeleteEmpty'] = 0
    mykwargs['cleanupDeleteToImprove'] = 0
    newModel, newSS = BirthRefine.expand_then_refine(
                              freshModel, freshSS, Data,
                              model, SS, **mykwargs)
    assert newSS.K == SS.K + freshSS.K