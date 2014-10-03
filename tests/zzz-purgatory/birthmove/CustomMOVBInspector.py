'''
Inspector module implements the bnpy custom_hook interface,
so we can double-check calculations happening within bnpy.run 
without modifying any learning algorithm code.
'''

import numpy as np
import os

def onBatchComplete(lapFrac=0, hmodel=None, learnAlg=None,
                    SS=None, SSchunk=None, Dchunk=None,
                    BirthResults=None, prevBirthResults=None, **kwargs):
    print '====================================', lapFrac
    if SS.hasELBOTerms():
      print '%.9e' % (hmodel.calc_evidence(SS=SS))
    print learnAlg.BirthCompIDs

    if prevBirthResults is None:
      prevBirthResults = list()
    if BirthResults is None:
      BirthResults = list()

    if (learnAlg.hasMove('birth') 
       and (len(BirthResults) > 0 or len(prevBirthResults) > 0)):
        verify_birth__global_equals_sum_of_batch(SS, learnAlg, lapFrac, 
                                               prevBirthResults, BirthResults)
    else:  
      verify_global_summary_equals_sum_of_batch_summaries(SS, learnAlg)

    if lapFrac >= 1.0:
      if len(BirthResults) == 0 or learnAlg.isLastBatch(lapFrac):
        assert SS.nDoc == Dchunk.nDocTotal


def assert_equal(arr, arr2):
  if not np.allclose(arr, arr2):
    print arr
    print arr2
  assert np.allclose(arr, arr2)

def verify_global_summary_equals_sum_of_batch_summaries(SS, learnAlg):
    ''' Verify we can 'rebuild' global SS from all batch-specific summaries
    '''
    assert learnAlg is not None

    SS2 = SS.copy()
    SS2._Fields.setAllFieldsToZero()
    SS2._ELBOTerms.setAllFieldsToZero()
    for batchID, SSchunk in learnAlg.SSmemory.items():
      assert SSchunk.K == SS2.K
      SS2 += SSchunk
    verify_summaries_are_equal(SS, SS2)    

def verify_summaries_are_equal(SS, SS2):
    for key in SS._FieldDims:
      arr = getattr(SS, key)
      arr2 = getattr(SS2, key)
      print '-------------------', key
      assert_equal(arr, arr2)

    for key in SS._ELBOTerms._FieldDims:
      arr = SS.getELBOTerm(key)
      arr2 = SS2.getELBOTerm(key)
      print '-------------------', key
      assert_equal(arr, arr2)

def verify_birth__global_equals_sum_of_batch(SS, learnAlg,
                                             lapFrac, 
                                             prevBirthResults, BirthResults):
    if prevBirthResults is None:
      BR = list()
    else:
      BR = [x for x in prevBirthResults]
    if BirthResults is not None:
      BR.extend(BirthResults)

    K = SS.K
    SS2 = SS.copy()
    SS2._Fields.setAllFieldsToZero()
    SS2._ELBOTerms.setAllFieldsToZero()
    for batchID, SSchunk in learnAlg.SSmemory.items():
      SSchunk = SSchunk.copy() # don't alter anything in learnAlg.SSmemory
      if SSchunk.K != K:
        SSchunk.insertEmptyComps(K - SSchunk.K)
        for MoveInfo in BR:
          if 'AdjustInfo' in MoveInfo and MoveInfo['AdjustInfo'] is not None:
            if 'bchecklist' in MoveInfo:
              if MoveInfo['bchecklist'][batchID] > 0:
                continue
            for key in MoveInfo['AdjustInfo']:
              if hasattr(SSchunk, key):
                arrA = getattr(SSchunk, key)
              else:
                arrA = SSchunk.getELBOTerm(key)
              arrB = SSchunk.nDoc * MoveInfo['AdjustInfo'][key]
              if arrA.size > arrB.size:
                arr = arrA
                arr[:arrB.size] += arrB
              else:
                arr = arrA + arrB
              if hasattr(SSchunk, key):
                SSchunk.setField(key, arr, dims=SSchunk._FieldDims[key])
              else:
                SSchunk.setELBOTerm(key, arr, dims='K')
          if 'ReplaceInfo' in MoveInfo and MoveInfo['ReplaceInfo'] is not None:
            if 'bchecklist' in MoveInfo:
              if MoveInfo['bchecklist'][batchID] > 1:
                continue
            for key in MoveInfo['ReplaceInfo']:
              arr = SSchunk.nDoc * MoveInfo['ReplaceInfo'][key]
              if hasattr(SSchunk, key):
                SSchunk.setField(key, arr, dims=SSchunk._FieldDims[key])
              else:
                SSchunk.setELBOTerm(key, arr, dims=None)
      SS2 += SSchunk
    if not learnAlg.isLastBatch(lapFrac) and BirthResults is not None:
      for MoveInfo in BirthResults:
        if 'extraSS' in MoveInfo:
          extraSS = MoveInfo['extraSS'].copy()
          if extraSS.K < SS2.K:
            extraSS.insertEmptyComps(SS2.K - extraSS.K)
          SS2 += extraSS
          
    verify_summaries_are_equal(SS, SS2)
