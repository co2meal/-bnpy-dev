'''
Unit tests for MergeMove.py

Verification merging works as expected and produces valid models.


'''
import numpy as np
import unittest

from bnpy import HModel
from bnpy.suffstats import SuffStatDict
from bnpy.learn import MergeMove
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
    oDict = dict(kappa=1e-5, dF=1, smatname='eye', sF=1e-3)
    self.hmodel = HModel.CreateEntireModel('VB', 'DPMixModel', 'Gauss', aDict, oDict, self.Data)
    LP = dict(resp=self.TrueResp)
    SS = self.hmodel.get_global_suff_stats(self.Data, LP)
    self.hmodel.update_global_params(SS)
    LP = self.hmodel.calc_local_params(self.Data)
    flagDict = dict(doPrecompEntropy=True, doPrecompMergeEntropy=True)
    self.SS = self.hmodel.get_global_suff_stats(self.Data, LP, **flagDict)

  def MakeModelWithDuplicatedComps(self):
    aDict = dict(alpha0=1.0, truncType='z')
    oDict = dict(kappa=1e-5, dF=1, smatname='eye', sF=1e-3)
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
    
  def verify_selected_component_ids(self, kA, kB):
    assert kA < self.SS.K
    assert kB < self.SS.K
    assert kA < kB

  def test_model_matches_ground_truth_as_precheck(self):
    ''' Before learning can proceed, need to ensure the model
          is able to learn ground truth.
    '''
    for k in range(self.hmodel.obsModel.K):
      muHat = self.hmodel.obsModel.get_mean_for_comp(k)
      print muHat, self.Mu[k]
      assert np.max(np.abs(muHat - self.Mu[k])) < 0.5
    LP = self.hmodel.calc_local_params(self.Data)
    absDiff = np.abs(LP['resp'] - self.TrueResp)
    maxDiff = np.max(absDiff, axis=1)
    assert np.sum( maxDiff < 0.1 ) > 0.5 * self.Data.nObs

  def test_select_merge_components_random_in_bounds(self):
    kA, kB = MergeMove.select_merge_components(self.hmodel, self.Data, self.SS, mergename='random')
    self.verify_selected_component_ids(kA, kB)

  def test_select_merge_components_random_seed_reproduceable(self):
    Alist = list()
    Blist = list()  
    for trial in range(10):
      kA, kB = MergeMove.select_merge_components(self.hmodel, self.Data, self.SS, mergename='random', randstate=np.random.RandomState(0))
      Alist.append(kA)
      Blist.append(kB)
    assert np.all( np.asarray(Alist) == Alist[0])
    assert np.all( np.asarray(Blist) == Blist[0])
  
  def test_select_merge_components_random_seed_updates_over_time(self):
    ''' Verify repeated calls with the same PRNG random generator object
        do NOT always have the same result (meaning the generator gets updated)
    '''
    Alist = list()
    Blist = list()  
    PRNG = np.random.RandomState(867)
    for trial in range(10):
      kA, kB = MergeMove.select_merge_components(self.hmodel, self.Data, self.SS, mergename='random', randstate=PRNG)
      Alist.append(kA)
      Blist.append(kB)
    assert not np.all( np.asarray(Alist) == Alist[0])
    assert not np.all( np.asarray(Blist) == Blist[0])

  def verify_proposed_model(self, propModel, propSS):
    assert propModel.allocModel.K == propSS.K
    assert propModel.obsModel.K == propSS.K
    assert len(propModel.obsModel.comp) == propSS.K
    if hasattr(propSS, 'N'):
      assert propSS.N.size == propSS.K
    if hasattr(propSS, 'x'):
      assert propSS.x.shape[0] == propSS.K
    # Check stick-break params
    if hasattr(propSS, 'qalpha0'):
      assert propModel.allocModel.qalpha0.size == propSS.K
      assert propModel.allocModel.qalpha1.size == propSS.K

  def test_propose_merge_candidate_produces_valid_model(self):
    ''' Test a proposal where we explicitly choose which comps to merge (kA,kB)
        and verify it produces a valid model
    '''
    propModel, propSS = MergeMove.propose_merge_candidate(self.hmodel, self.SS, kA=0, kB=1)
    # Check number of components!    
    assert propSS.K == self.SS.K - 1
    self.verify_proposed_model(propModel, propSS)
    # Check that we can now do further E-steps and get sensible results
    LP = propModel.calc_local_params(self.Data)
    R = LP['resp']
    assert R.shape[0] == self.Data.nObs
    assert R.shape[1] == propSS.K
    assert not np.any(np.isnan(R))
    Neff = np.sum(R, axis=0)
    # Merger should cause new component "0" to explain twice as much data as
    # component 1. Let's verify this
    assert Neff[0] > 1.5*Neff[1]
    # Merged model should have mu for comp 0 (merged one) around -5
    mu0 = propModel.obsModel.get_mean_for_comp(0)
    distFromExpected = np.abs(mu0[0] - -5.0)
    assert distFromExpected < 1.0
    # Merged model should have mu for comp 1 near where only comp 2 was
    newMu1 = propModel.obsModel.get_mean_for_comp(1)
    oldMu2 = self.hmodel.obsModel.get_mean_for_comp(2)
    distFromExpected = np.max(np.abs(newMu1 - oldMu2))
    assert distFromExpected < 1.0

  def test_run_merge_move_on_true_comps_fails(self):
    ''' 
    '''
    for trial in range(10):
      newModel, newSS, newEv, MoveInfo = MergeMove.run_merge_move(self.hmodel, self.Data, self.SS, mergename='random')
      assert newModel.allocModel.K == self.hmodel.allocModel.K
      assert newModel.obsModel.K == self.hmodel.obsModel.K

  def test_run_merge_move_on_duplicated_comps_succeeds_with_ideal_choice(self):
    ''' Consider duplicated comps model.
        Attempt merge move on each pair of known "duplicates", 
          these are comp IDs  (0,3),  (1,4), and (2,5)
        This is "ideal", since in practice we wont know which components to merge.
        This isolates whether the merges work even in the best of circumstances.
    '''
    Ktrue = self.hmodel.obsModel.K
    for k in range(Ktrue):
      newModel, newSS, newEv, MoveInfo = MergeMove.run_merge_move(self.dupModel, self.Data, self.dupSS, kA=k, kB=Ktrue+k)
      assert newModel.obsModel.K == self.dupModel.obsModel.K - 1


  def test_run_merge_move_on_duplicated_comps_succeeds_with_random_choice(self):
    ''' Consider Duplicated Comps model.
        Out of (6 choose 2) = 15 possible pairs, exactly 3 produce sensible merges.
        Verify that over many random trials where kA,kB drawn uniformly,
          we obtain a success rate not too different from 3 / 15 = 0.2.
    '''
    nTrial = 100
    nSuccess = 0
    PRNG = np.random.RandomState(0)
    for trial in range(nTrial):
      newModel, newSS, newEv, MoveInfo = MergeMove.run_merge_move(self.dupModel, self.Data, self.dupSS, mergename='random', randstate=PRNG)
      if MoveInfo['didAccept']:
        nSuccess += 1
    assert nSuccess > 0
    rate = float(nSuccess)/float(nTrial)
    assert rate > 0.1
    assert rate < 0.3

  def test_run_merge_move_on_duplicated_comps_succeeds_with_marglik_choice(self):
    ''' Consider Duplicated Comps model.
        Instead of random choice use marglik criteria to select candidates kA, kB.
        Verify that the merge accept rate is much much higher than choosing uniformly at random.  The accept rate should actually be near perfect!
    '''
    nTrial = 100
    nSuccess = 0
    PRNG = np.random.RandomState(0)
    for trial in range(nTrial):
      newModel, newSS, newEv, MoveInfo = MergeMove.run_merge_move(self.dupModel, self.Data, self.dupSS, mergename='marglik', randstate=PRNG)
      if MoveInfo['didAccept']:
        nSuccess += 1
    assert nSuccess > 0
    rate = float(nSuccess)/float(nTrial)
    print rate
    assert rate > 0.8

  

