'''
Unit tests for MergeMove.py for HDPTopicModels

Verification merging works as expected and produces valid models.


'''
import numpy as np
import unittest

import bnpy
from bnpy.learnalg import MergeMove
from scipy.special import digamma
import copy

class TestMergeHDP(unittest.TestCase):
  def shortDescription(self):
    return None

  def setUp(self):
    self.MakeData()
    self.MakeModelWithTrueComps()
    self.MakeMinibatches()

  def MakeMinibatches(self):
    PRNG = np.random.RandomState(1234)
    permIDs =  PRNG.permutation(self.Data.nDocTotal)
    bIDs1 = permIDs[:len(permIDs)/2]
    bIDs2 = permIDs[len(permIDs)/2:]
    self.batchData1 = self.Data.select_subset_by_mask(bIDs1)
    self.batchData2 = self.Data.select_subset_by_mask(bIDs2)

  def getTrueLP(self):
    return self.getLPfromResp(self.trueResp)

  def getDupLP(self):
    Data = self.Data
    K = self.trueResp.shape[1]
    dupResp = np.zeros((Data.nObs, 2*K))
    dupResp[:Data.nObs/2,:K] = self.trueResp[:Data.nObs/2]
    dupResp[Data.nObs/2:,K:] = self.trueResp[Data.nObs/2:]
    return self.getLPfromResp(dupResp)

  def getLPfromResp(self, Resp, smoothMass=0.001):
    Data = self.Data
    D = Data.nDoc
    K = Resp.shape[1]
    # DocTopicCount matrix : D x K matrix
    DocTopicC = np.zeros((D, K))
    for dd in range(D):
      start,stop = Data.doc_range[dd,:]
      DocTopicC[dd,:] = np.dot(Data.word_count[start:stop],        
                               Resp[start:stop,:]
                               )
    assert np.allclose(DocTopicC.sum(), Data.word_count.sum())
    # Alpha and ElogPi : D x K+1 matrices
    padCol = smoothMass * np.ones((D,1))
    alph = np.hstack( [DocTopicC + smoothMass, padCol])    
    ElogPi = digamma(alph) - digamma(alph.sum(axis=1))[:,np.newaxis]
    assert ElogPi.shape == (D,K+1)
    return dict(word_variational =Resp, 
              E_logPi=ElogPi, alphaPi=alph,
              DocTopicCount=DocTopicC)    

  def MakeModelWithTrueComps(self):
    aDict = dict(alpha0=1.0, gamma=0.1)
    oDict = {'lambda':0.05}
    self.hmodel = bnpy.HModel.CreateEntireModel('VB', 'HDPModel', 'Mult',
                                             aDict, oDict, self.Data)
    LP = self.getTrueLP()
    SS = self.hmodel.get_global_suff_stats(self.Data, LP)
    self.hmodel.update_global_params(SS)

  def MakeModelWithDuplicatedComps(self):
    aDict = dict(alpha0=1.0, gamma=0.1)
    oDict = {'lambda':0.05}
    self.dupModel = bnpy.HModel.CreateEntireModel('VB', 'HDPModel', 'Mult',
                                             aDict, oDict, self.Data)
    dupLP = self.getDupLP()
    dupSS = self.dupModel.get_global_suff_stats(self.Data, dupLP)
    self.dupModel.update_global_params(dupSS)

  def run_Estep_then_Mstep(self):
    ''' Run Estep, calc suff stats, then do an Mstep
    '''
    LP = self.hmodel.calc_local_params(self.Data)
    flagDict = dict(doPrecompEntropy=True, doPrecompMergeEntropy=True)
    SS = self.hmodel.get_global_suff_stats(self.Data, LP, **flagDict)
    self.hmodel.update_global_params(SS)
    return LP, SS

  def MakeData(self, K=4, D=2500, nWordsPerDoc=50):
    ''' Simple 4 component data on 6 word vocabulary
        
    '''
    TopicWord = np.zeros((K,6))
    TopicWord[0] = [0.48, 0.48, 0.01, 0.01, 0.01, 0.01]
    TopicWord[1] = [0.01, 0.01, 0.48, 0.48, 0.01, 0.01]
    TopicWord[2] = [0.01, 0.01, 0.01, 0.01, 0.48, 0.48]
    TopicWord[3] = [0.01, 0.33, 0.01, 0.32, 0.01, 0.32]
    docTopicParamVec = 0.1 * np.ones(4)
    Data = bnpy.data.WordsData.genToyData(TopicWordProbs=TopicWord,
                      docTopicParamVec=docTopicParamVec,
                      nDocTotal=D,
                      nWordsPerDoc=nWordsPerDoc, seed=123)
    # "Make up" the right local params for this data
    self.trueResp = np.zeros( (Data.nObs, K))
    docID = 0
    for nn in range(Data.nObs):
      if nn >= Data.doc_range[docID,1]:
        docID += 1
      topicProbs = Data.true_td[:,docID] 
      self.trueResp[nn,:] = topicProbs * TopicWord[:,Data.word_id[nn]]
    self.trueResp /= np.sum(self.trueResp,axis=1)[:,np.newaxis]
    assert np.allclose(1.0,np.sum(self.trueResp,axis=1))
    self.Data = Data
  

  ######################################################### run_merge_move
  #########################################################  full tests
  def test_model_matches_ground_truth_as_precheck(self):
    ''' Verify HDPmodel is able to learn ground truth parameters
          and maintain stable estimates after several E/M steps
    '''
    np.set_printoptions(precision=3,suppress=True)
    # Advance the model several iterations
    for rr in range(5):
      self.run_Estep_then_Mstep()
    for k in range(self.hmodel.obsModel.K):
      logtopicWordHat = self.hmodel.obsModel.comp[k].Elogphi
      topicWordHat = np.exp(logtopicWordHat)
      print topicWordHat
      print self.Data.true_tw[k]
      diffVec = np.abs(topicWordHat - self.Data.true_tw[k])
      print diffVec
      print ' '
      assert np.max(diffVec) < 0.04

  ######################################################### Direct merge LP params
  #########################################################  (for comparison)
  def calc_mergeLP(self, LP, kA, kB):
    K = LP['DocTopicCount'].shape[1]
    assert kA < kB
    LP = copy.deepcopy(LP)
    for key in ['DocTopicCount', 'word_variational', 'alphaPi']:
      LP[key][:,kA] = LP[key][:,kA] + LP[key][:,kB]
      LP[key] = np.delete(LP[key], kB, axis=1)
    LP['E_logPi'] = digamma(LP['alphaPi']) \
                    - digamma(LP['alphaPi'].sum(axis=1))[:,np.newaxis]
    assert LP['word_variational'].shape[1] == K-1
    return LP


  ######################################################### Verify Z terms
  
  def test_ELBO_Z_terms_are_correct(self):
    ''' Verify that the ELBO terms for ElogpZ, ElogqZ are correct
    '''
    aModel = self.hmodel.allocModel
    LP = self.hmodel.calc_local_params(self.Data)
    # 1) Calculate the ELBO terms directly
    directElogpZ = np.sum(aModel.E_logpZ(self.Data, LP))
    directElogqZ = np.sum(aModel.E_logqZ(self.Data, LP))
    # 2) Calculate the terms via suff stats
    SS = self.hmodel.get_global_suff_stats(self.Data, LP, doPrecompEntropy=True)
    memoElogpZ = np.sum(SS.getELBOTerm('ElogpZ'))
    memoElogqZ = np.sum(SS.getELBOTerm('ElogqZ'))
    assert np.allclose(directElogpZ, memoElogpZ)
    assert np.allclose(directElogqZ, memoElogqZ)

  def test_ELBO_Z_terms_are_correct_memoized(self):
    ''' Verify that the ELBO terms for ElogpZ, ElogqZ are correct
        when aggregated across batches!
    '''
    aModel = self.hmodel.allocModel
    LP1 = self.hmodel.calc_local_params(self.batchData1)
    LP2 = self.hmodel.calc_local_params(self.batchData2)
    # 1) Calculate the ELBO terms directly
    directElogpZ = np.sum(aModel.E_logpZ(self.batchData1, LP1) \
                         + aModel.E_logpZ(self.batchData2, LP2)
                         )
    directElogqZ = np.sum(aModel.E_logqZ(self.batchData1, LP1) \
                         + aModel.E_logqZ(self.batchData2, LP2)
                         )
    # 2) Calculate via aggregated suff stats
    SS1 = self.hmodel.get_global_suff_stats(self.batchData1, LP1, doPrecompEntropy=True)
    SS2 = self.hmodel.get_global_suff_stats(self.batchData2, LP2, doPrecompEntropy=True)
    SS = SS1 + SS2
    memoElogpZ = np.sum(SS.getELBOTerm('ElogpZ'))
    memoElogqZ = np.sum(SS.getELBOTerm('ElogqZ'))
    assert np.allclose(directElogpZ, memoElogpZ)
    assert np.allclose(directElogqZ, memoElogqZ)

  def test_ELBO_Z_terms_are_correct_merge(self):
    ''' Verify that the ELBO terms for ElogpZ, ElogqZ are correct
        when a merge occurs
    '''
    aModel = self.hmodel.allocModel
    LP1 = self.hmodel.calc_local_params(self.batchData1)
    LP2 = self.hmodel.calc_local_params(self.batchData2)
 
    beforeElogqZ = aModel.E_logqZ(self.batchData1, LP1) \
                 + aModel.E_logqZ(self.batchData2, LP2)
    print beforeElogqZ
    # ----------------------------------------  Consider merge of comps 0 and 1
    kA = 0
    kB = 1
    mLP1 = self.calc_mergeLP(LP1, kA, kB)
    mLP2 = self.calc_mergeLP(LP2, kA, kB)
    # 1) Calculate the ELBO terms directly
    directElogpZ = aModel.E_logpZ(self.batchData1, mLP1) \
                 + aModel.E_logpZ(self.batchData2, mLP2)
                         
    directElogqZ = aModel.E_logqZ(self.batchData1, mLP1) \
                 + aModel.E_logqZ(self.batchData2, mLP2)
                         
    assert LP1['DocTopicCount'].shape[1] == aModel.K # still have originals!
    assert LP2['DocTopicCount'].shape[1] == aModel.K # still have originals!
    # 2) Calculate via suff stats
    SS1 = self.hmodel.get_global_suff_stats(self.batchData1, LP1,
                           doPrecompEntropy=True, doPrecompMergeEntropy=True)
    SS2 = self.hmodel.get_global_suff_stats(self.batchData2, LP2,
                           doPrecompEntropy=True, doPrecompMergeEntropy=True)
    SS = SS1 + SS2
    SS.mergeComps(kA, kB)
    memoElogpZ = SS.getELBOTerm('ElogpZ')
    memoElogqZ = SS.getELBOTerm('ElogqZ')
    assert np.allclose(directElogpZ, memoElogpZ)
    assert np.allclose(directElogqZ, memoElogqZ)
    # ----------------------------------------  Follow-up merge of comps 2 and 3
    # since we've collapsed 0,1 into 0 previously,
    #  original comps 2,3 have been "renamed" 1,2
    kA = 1
    kB = 2
    mLP1 = self.calc_mergeLP(mLP1, kA, kB)
    mLP2 = self.calc_mergeLP(mLP2, kA, kB)
    assert mLP1['DocTopicCount'].shape[1] == aModel.K - 2 # we've done two merges
    # 1) Calculate the ELBO terms directly
    directElogpZ = aModel.E_logpZ(self.batchData1, mLP1) \
                 + aModel.E_logpZ(self.batchData2, mLP2)
                         
    directElogqZ = aModel.E_logqZ(self.batchData1, mLP1) \
                 + aModel.E_logqZ(self.batchData2, mLP2)
    # 2) Calculate via suff stats (carried over from earlier merge!)
    SS.mergeComps(kA, kB)
    memoElogpZ = SS.getELBOTerm('ElogpZ')
    memoElogqZ = SS.getELBOTerm('ElogqZ')
    assert np.allclose(directElogpZ, memoElogpZ)
    assert np.allclose(directElogqZ, memoElogqZ)
    # Make sure we don't have any more valid precomputed merge terms
    WMat = SS.getMergeTerm('ElogpZ')
    numZeros = np.sum( WMat == 0)
    numNans = np.isnan(WMat).sum()
    assert numZeros + numNans == WMat.size

  ######################################################### Verify Pi terms
  ######################################################### (doc-topic prob vector)

  def test_ELBO_Pi_terms_are_correct(self):
    ''' Verify that the ELBO terms for ElogpPi, ElogqPi are correct
    '''
    aModel = self.hmodel.allocModel
    LP = self.hmodel.calc_local_params(self.Data)
    SS = self.hmodel.get_global_suff_stats(self.Data, LP, doPrecompEntropy=True)
    # 1) Calculate the ELBO terms directly
    directElogpPi = np.sum(aModel.E_logpPi(SS))
    directElogqPi = np.sum(aModel.E_logqPi(LP))
    # 2) Calculate the terms via suff stats
    memoElogpPi = np.sum(aModel.E_logpPi(SS))
    memoElogqPi = np.sum(SS.getELBOTerm('ElogqPiActive')) \
                    + SS.getELBOTerm('ElogqPiConst') \
                    + SS.getELBOTerm('ElogqPiUnused')
    assert np.allclose(directElogpPi, memoElogpPi)
    assert np.allclose(directElogqPi, memoElogqPi)

  def test_ELBO_Pi_terms_are_correct_memoized(self):
    ''' Verify that the ELBO terms for ElogpPi, ElogqPi are correct
    '''
    aModel = self.hmodel.allocModel
    LP1 = self.hmodel.calc_local_params(self.batchData1)
    LP2 = self.hmodel.calc_local_params(self.batchData2)

    SS1 = self.hmodel.get_global_suff_stats(self.batchData1, LP1,   doPrecompEntropy=True)
    SS2 = self.hmodel.get_global_suff_stats(self.batchData2, LP2, doPrecompEntropy=True)
    SS = SS1 + SS2
    assert SS.nDoc == self.Data.nDocTotal

    # 1) Calculate the ELBO terms directly
    directElogpPi = np.sum(aModel.E_logpPi(SS))
    directElogqPi = np.sum(aModel.E_logqPi(LP1) + aModel.E_logqPi(LP2))

    # 2) Calculate the terms via suff stats
    memoElogpPi = np.sum(aModel.E_logpPi(SS))
    memoElogqPi = np.sum(SS.getELBOTerm('ElogqPiActive')) \
                    + SS.getELBOTerm('ElogqPiConst') \
                    + SS.getELBOTerm('ElogqPiUnused')

    assert np.allclose(directElogpPi, memoElogpPi)
    assert np.allclose(directElogqPi, memoElogqPi)

  

  def test_ELBO_Pi_terms_are_correct_merge(self):
    ''' Verify that the ELBO terms for ElogpPi, ElogqPi are correct
    '''
    aModel = self.hmodel.allocModel
    LP1 = self.hmodel.calc_local_params(self.batchData1)
    LP2 = self.hmodel.calc_local_params(self.batchData2)

    SS1 = self.hmodel.get_global_suff_stats(self.batchData1, LP1,   doPrecompEntropy=True, doPrecompMergeEntropy=True)
    SS2 = self.hmodel.get_global_suff_stats(self.batchData2, LP2, doPrecompEntropy=True, doPrecompMergeEntropy=True)

    SS = SS1 + SS2
    assert SS.nDoc == self.Data.nDocTotal
    cTerm = SS.getELBOTerm('ElogqPiConst')
    cTerm1 = SS1.getELBOTerm('ElogqPiConst')
    cTerm2 = SS2.getELBOTerm('ElogqPiConst')
    assert np.allclose(cTerm, cTerm1 + cTerm2)

    # Consider merge of comps 0 and 1
    K = SS.K
    print "BEFORE sumLogPiActive", SS.sumLogPiActive
    kA = 0
    kB = 1
    mLP1 = self.calc_mergeLP(LP1, kA, kB)
    mLP2 = self.calc_mergeLP(LP2, kA, kB)
    mergeSumLogPi_kA = SS.getMergeTerm('sumLogPiActive')[kA,kB]

    print "EXPECTED new entry: ", SS.getMergeTerm('sumLogPiActive')[kA,kB]

    SS.mergeComps(kA, kB)
    print "AFTER sumLogPiActive", SS.sumLogPiActive

    assert np.allclose(SS.getELBOTerm('ElogqPiConst'), cTerm1 + cTerm2)

    sumLogPi = mLP1['E_logPi'].sum(axis=0) + mLP2['E_logPi'].sum(axis=0)
    sumLogPiVec = np.hstack([SS.sumLogPiActive, SS.sumLogPiUnused])

    assert sumLogPi[kA] == mergeSumLogPi_kA
    assert sumLogPiVec[kA] == mergeSumLogPi_kA

    print sumLogPi, sumLogPiVec
    assert np.allclose(sumLogPi, sumLogPiVec)

    # 1) Calculate the ELBO terms directly
    directElogqPi = np.sum(aModel.E_logqPi(mLP1) + aModel.E_logqPi(mLP2))

    # 2) Calculate the terms via suff stats
    memoElogqPi = np.sum(SS.getELBOTerm('ElogqPiActive')) \
                    + SS.getELBOTerm('ElogqPiConst') \
                    + SS.getELBOTerm('ElogqPiUnused')
    print directElogqPi
    print memoElogqPi
    assert np.allclose(directElogqPi, memoElogqPi)

    # ----------------------------------------  Follow-up merge of comps 2 and 3
    # since we've collapsed 0,1 into 0 previously,
    #  original comps 2,3 have been "renamed" 1,2
    kA = 1
    kB = 2
    mLP1 = self.calc_mergeLP(mLP1, kA, kB)
    mLP2 = self.calc_mergeLP(mLP2, kA, kB)
    assert mLP1['DocTopicCount'].shape[1] == aModel.K - 2 # we've done two merges
    # 1) Calculate the ELBO terms directly
    directElogqPi = np.sum(aModel.E_logqPi(mLP1) + aModel.E_logqPi(mLP2))
                         
    # 2) Calculate via suff stats (carried over from earlier merge!)
    SS.mergeComps(kA, kB)
    memoElogqPi = np.sum(SS.getELBOTerm('ElogqPiActive')) \
                    + SS.getELBOTerm('ElogqPiConst')\
                    + SS.getELBOTerm('ElogqPiUnused')
    assert np.allclose(directElogqPi, memoElogqPi)


  def test_ELBO_terms_are_correct_merge_duplicates(self):
    ''' Verify that the ELBO terms for ElogpPi, ElogqPi are correct
    '''
    np.set_printoptions(precision=4, suppress=True)
    kA = 2
    kB = 3
    self.MakeModelWithDuplicatedComps()

    LP = self.dupModel.calc_local_params(self.Data)
    SS = self.dupModel.get_global_suff_stats(self.Data, LP,   doPrecompEntropy=True, doPrecompMergeEntropy=True)
    beforeELBO = self.dupModel.calc_evidence(self.Data, SS, LP)

    # Perform merge via suff stats
    SS.mergeComps(kA,kB)
    newModel = self.dupModel.copy()
    newModel.update_global_params(SS)
    memoELBO = newModel.calc_evidence(SS=SS)
    assert newModel.allocModel.K == 7

    # Perform merge via direct operation on true components
    assert self.dupModel.allocModel.K == 8
    mLP = self.calc_mergeLP(LP, kA, kB)
    mSS = self.dupModel.get_global_suff_stats(self.Data, mLP)
    assert np.allclose(mSS.WordCounts, SS.WordCounts)
    assert np.allclose(mSS.sumLogPiActive, SS.sumLogPiActive)

    self.dupModel.update_global_params(mSS)
    assert self.dupModel.allocModel.K == 7
    directELBO = self.dupModel.calc_evidence(self.Data, mSS, mLP)


    print beforeELBO
    print directELBO
    print memoELBO
    assert np.allclose(directELBO, memoELBO)
    assert beforeELBO > directELBO


  def test_ELBO_terms_are_correct_merge_true(self):
    ''' Verify that entire ELBO is correct
    '''
    np.set_printoptions(precision=4, suppress=True)
    kA = 2
    kB = 3

    LP = self.hmodel.calc_local_params(self.Data)
    SS = self.hmodel.get_global_suff_stats(self.Data, LP,   doPrecompEntropy=True, doPrecompMergeEntropy=True)
    beforeELBO = self.hmodel.calc_evidence(self.Data, SS, LP)

    # Perform merge via suff stats
    SS.mergeComps(kA,kB)
    newModel = self.hmodel.copy()
    newModel.update_global_params(SS)
    memoELBO = newModel.calc_evidence(SS=SS)
    assert newModel.allocModel.K == 3

    # Perform merge via direct operation on true components
    assert self.hmodel.allocModel.K == 4
    mLP = self.calc_mergeLP(LP, kA, kB)
    mSS = self.hmodel.get_global_suff_stats(self.Data, mLP)
    assert np.allclose(mSS.WordCounts, SS.WordCounts)
    assert np.allclose(mSS.sumLogPiActive, SS.sumLogPiActive)

    self.hmodel.update_global_params(mSS)
    assert self.hmodel.allocModel.K == 3
    directELBO = self.hmodel.calc_evidence(self.Data, mSS, mLP)

    print beforeELBO
    print directELBO
    print memoELBO
    assert np.allclose(directELBO, memoELBO)
    assert beforeELBO > directELBO



  ######################################################### run_merge_move
  #########################################################  full tests
  def test_run_merge_move_on_true_comps_fails(self):
    ''' Should not be able to merge "true" components into one another
        Each is necessary to explain (some) data
    '''
    mergeFlags = dict(doPrecompEntropy=True, doPrecompMergeEntropy=True)
    LP = self.hmodel.calc_local_params(self.Data)
    SS = self.hmodel.get_global_suff_stats(self.Data, LP, **mergeFlags)
    for trial in range(10):
      newModel, newSS, newEv, MoveInfo = MergeMove.run_merge_move(self.hmodel, self.Data, SS, mergename='random')
      assert newModel.allocModel.K == self.hmodel.allocModel.K
      assert newModel.obsModel.K == self.hmodel.obsModel.K

  def test_run_merge_move_on_dup_comps_succeeds_with_each_ideal_pair(self):
    ''' Given the duplicated comps model,
          which has a redundant copy of each "true" component,
        We show that deliberately merging each pair does succeed.
        This is "ideal" since we know in advance which merge pair to try
    '''
    self.MakeModelWithDuplicatedComps()
    mergeFlags = dict(doPrecompEntropy=True, doPrecompMergeEntropy=True)
    LP = self.dupModel.calc_local_params(self.Data)
    SS = self.dupModel.get_global_suff_stats(self.Data, LP, **mergeFlags)
    for kA in [0,1,2,3]:
      kB = kA + 4 # Ktrue=4, so kA's best match is kA+4 
      newModel, newSS, newEv, MoveInfo = MergeMove.run_merge_move(self.dupModel,
                                         self.Data, SS, kA=kA, kB=kB)
      print MoveInfo['msg']
      assert newModel.allocModel.K == self.dupModel.allocModel.K - 1
      assert newModel.obsModel.K == self.dupModel.obsModel.K - 1
      assert MoveInfo['didAccept'] == 1

  def test_run_merge_move_on_dup_comps_fails_with_nonideal_pairs(self):
    ''' Given the duplicated comps model,
          which has a redundant copy of each "true" component,
        We show that deliberately merging each pair does succeed.
        This is "ideal" since we know in advance which merge pair to try
    '''
    self.MakeModelWithDuplicatedComps()
    mergeFlags = dict(doPrecompEntropy=True, doPrecompMergeEntropy=True)
    LP = self.dupModel.calc_local_params(self.Data)
    SS = self.dupModel.get_global_suff_stats(self.Data, LP, **mergeFlags)
    for Kstep in [1,2,3,5,6,7]:
      for kA in range(8 - Kstep):
        kB = kA + Kstep
        newM, newSS, newEv, MoveInfo = MergeMove.run_merge_move(self.dupModel,
                                         self.Data, SS, kA=kA, kB=kB)
        print MoveInfo['msg']
        assert MoveInfo['didAccept'] == 0


  def test_run_merge_move_on_dup_comps_succeeds_with_all_ideal_pairs(self):
    self.MakeModelWithDuplicatedComps()
    mergeFlags = dict(doPrecompEntropy=True, doPrecompMergeEntropy=True)
    LP = self.dupModel.calc_local_params(self.Data)
    SS = self.dupModel.get_global_suff_stats(self.Data, LP, **mergeFlags)
    myModel = self.dupModel.copy()
    for kA in [3,2,1,0]: # descend backwards so indexing still works
      kB = kA + 4 # Ktrue=4, so kA's best match is kA+4 
      myModel, SS, newEv, MoveInfo = MergeMove.run_merge_move(myModel,
                                         self.Data, SS, kA=kA, kB=kB)
      print MoveInfo['msg']
      assert MoveInfo['didAccept'] == 1

  def test_run_merge_move_on_dup_comps_succeeds_with_random_choice(self):
    ''' Consider Duplicated Comps model.
        Out of (8 choose 2) = 28 possible pairs, exactly 4 produce sensible merges.
        Verify that over many random trials where kA,kB drawn uniformly,
          we obtain a success rate not too different from 4 / 28 = 0.142857
    '''
    self.MakeModelWithDuplicatedComps()
    mergeFlags = dict(doPrecompEntropy=True, doPrecompMergeEntropy=True)
    LP = self.dupModel.calc_local_params(self.Data)
    SS = self.dupModel.get_global_suff_stats(self.Data, LP, **mergeFlags)
    nTrial = 100
    nSuccess = 0
    PRNG = np.random.RandomState(0)
    for trial in range(nTrial):
      newModel, newSS, newEv, MoveInfo = MergeMove.run_merge_move(self.dupModel, self.Data, SS, mergename='random', randstate=PRNG)
      if MoveInfo['didAccept']:
        print MoveInfo['msg']
        nSuccess += 1
    assert nSuccess > 0
    rate = float(nSuccess)/float(nTrial)
    print "Expected rate: .1428"
    print "Measured rate: %.3f" % (rate)
    assert rate > 0.1
    assert rate < 0.2

  def test_run_merge_move_on_dup_comps_succeeds_with_marglik_choice(self):
    ''' Consider Duplicated Comps model.
        Instead of random choice use marglik criteria to select candidates kA, kB.
        Verify that the merge accept rate is much much higher than choosing uniformly at random.  The accept rate should actually be near perfect!
    '''
    self.MakeModelWithDuplicatedComps()
    mergeFlags = dict(doPrecompEntropy=True, doPrecompMergeEntropy=True)
    LP = self.dupModel.calc_local_params(self.Data)
    SS = self.dupModel.get_global_suff_stats(self.Data, LP, **mergeFlags)
    nTrial = 100
    nSuccess = 0
    PRNG = np.random.RandomState(0)
    for trial in range(nTrial):
      newModel, newSS, newEv, MoveInfo = MergeMove.run_merge_move(self.dupModel, self.Data, SS, mergename='marglik', randstate=PRNG)
      print MoveInfo['msg']
      if MoveInfo['didAccept']:
        nSuccess += 1
    assert nSuccess > 0
    rate = float(nSuccess)/float(nTrial)
    print "Expected rate: >.95"
    print "Measured rate: %.3f" % (rate)
    assert rate > 0.95
