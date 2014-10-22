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

    print '====================================', lapFrac, 'K=', SS.K
    if SS.hasELBOTerms():
      print '%.9e' % (hmodel.calc_evidence(SS=SS))
      print ['%.1f' % (x) for x in SS.N[:8]]

    assert learnAlg.hasMove('merge')
    assert not learnAlg.hasMove('birth')

    verify_global_summary_equals_sum_of_batch_summaries(SS, learnAlg)
    if lapFrac >= 1.0:
      if hasattr(Dchunk, 'nObsTotal'):
        assert np.allclose(SS.N.sum(), Dchunk.nObsTotal)

def assert_equal(arr, arr2):
  if not np.allclose(arr, arr2):
    print '!!!!!@@@@'
    print arr
    print arr2
  else:
    print '  verified.'
  assert np.allclose(arr, arr2)

def verify_global_summary_equals_sum_of_batch_summaries(SS, learnAlg):
    ''' Verify we can 'rebuild' global SS from all batch-specific summaries
    '''
    SS2 = SS.copy()
    SS2._Fields.setAllFieldsToZero()
    SS2._ELBOTerms.setAllFieldsToZero()
    for batchID in learnAlg.SSmemory.keys():
      SSchunk = learnAlg.load_batch_suff_stat_from_memory(batchID, SS.K,
                                                          doCopy=1)
      assert SSchunk.K == SS2.K
      SS2 += SSchunk
    verify_fields_and_ELBOTerms_are_equal(SS, SS2)    

def verify_fields_and_ELBOTerms_are_equal(SS, SS2):
    for key in SS._FieldDims:
      arr = getattr(SS, key)
      arr2 = getattr(SS2, key)
      print '-------------------', key,
      assert_equal(arr, arr2)

    for key in SS._ELBOTerms._FieldDims:
      arr = SS.getELBOTerm(key)
      arr2 = SS2.getELBOTerm(key)
      print '-------------------', key,
      assert_equal(arr, arr2)
