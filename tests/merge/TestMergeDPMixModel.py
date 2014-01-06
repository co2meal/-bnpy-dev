
import numpy as np
import unittest

from bnpy import HModel
from bnpy.learnalg import MergeMove, BirthMove
from bnpy.util.RandUtil import mvnrand
from bnpy.data import XData

class TestMergeMove(unittest.TestCase):
  def shortDescription(self):
    return None

  def setUp(self):
    self.MakeData()
    self.MakeModelWithTrueComps()
    self.MakeModelWithDuplicatedComps()

  def MakeModelWithTrueComps(self):
    aDict = dict(alpha0=1.0, truncType='z')
    oDict = dict(kappa=1e-5, dF=1, ECovMat='eye', sF=1e-3)
    self.hmodel = HModel.CreateEntireModel('VB', 'DPMixModel', 'Gauss', aDict, oDict, self.Data)
    LP = dict(resp=self.TrueResp)
    SS = self.hmodel.get_global_suff_stats(self.Data, LP)
    self.hmodel.update_global_params(SS)
    LP = self.hmodel.calc_local_params(self.Data)
    flagDict = dict(doPrecompEntropy=True, doPrecompMergeEntropy=True)
    self.SS = self.hmodel.get_global_suff_stats(self.Data, LP, **flagDict)

  def MakeModelWithDuplicatedComps(self):
    aDict = dict(alpha0=1.0, truncType='z')
    oDict = dict(kappa=1e-5, dF=1, ECovMat='eye', sF=1e-3)
    self.dupModel = HModel.CreateEntireModel('VB', 'DPMixModel', 'Gauss', aDict, oDict, self.Data)

    LP = dict(resp=self.DupResp)
    SS = self.hmodel.get_global_suff_stats(self.Data, LP)
    self.dupModel.update_global_params(SS)

    LP = self.dupModel.calc_local_params(self.Data)
    flagDict = dict(doPrecompEntropy=True, doPrecompMergeEntropy=True)
    self.dupSS = self.dupModel.get_global_suff_stats(self.Data, LP, **flagDict)

  def MakeData(self, K=3, Nperclass=1000):
    ''' Simple 3 component data with eye covar and distinct, well-sep means
        mu0 = [-10, -10]
        mu1 = [0, 0]
        mu2 = [10, 10]
    '''
    Mu = np.zeros((3,2))
    Mu[0] = Mu[0] - 10
    Mu[2] = Mu[2] + 10
    Sigma = np.eye(2)    
    self.TrueResp = np.zeros((K*Nperclass,K))
    self.DupResp = np.zeros((K*Nperclass, 2*K))
    Xlist = list()
    for k in range(K):
      Xcur = mvnrand( Mu[k], Sigma, Nperclass)    
      Xlist.append(Xcur)
      self.TrueResp[k*Nperclass:(k+1)*Nperclass, k] = 1.0
      start = k*Nperclass
      stop = (k+1)*Nperclass
      half = 0.5*(start + stop)      
      self.DupResp[start:half, k] = 1.0
      self.DupResp[half:stop, K+k] = 1.0
    X = np.vstack(Xlist)
    self.Data = XData(X=X)
    self.Mu = Mu
    assert np.abs(self.TrueResp.sum() - self.Data.nObs) < 1e-2
    assert np.abs(self.DupResp.sum() - self.Data.nObs) < 1e-2
    

  def test_run_birth_move(self):
    '''
    '''
    PRNG = np.random.RandomState(12345)
    birthArgs = dict(Kfresh=10, freshInitName='randexamples', 
                      freshAlgName='VB', nFreshLap=50)
    newModel, newSS, MInfo = BirthMove.run_birth_move(self.hmodel, 
                      self.Data, self.SS, randstate=PRNG, **birthArgs)
    assert newModel.obsModel.K > self.hmodel.obsModel.K

  
  def test_run_many_merge_moves(self):
    ''' 
    '''
    PRNG = np.random.RandomState(12345)
    newModel, newSS, Info = MergeMove.run_many_merge_moves(
                    self.dupModel, self.Data, self.dupSS, nMergeTrials=6,
                    randstate=PRNG)
    assert newSS.K == self.SS.K
    