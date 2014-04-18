'''
Unit tests for moVB with births, for deciding when births happen.

Coverage
--------
* do_birth_at_lap
  * verify births occur at the expected times (when lap < fracLapsBirth*nLap)
'''
import numpy as np
import unittest

import bnpy
import UtilForBirthTest as U

def np2flatstr(xvec, fmt='% 9.3f', Dmax=8, offset=' '):
  if np.asarray(xvec).ndim == 0:
    msg = fmt % (xvec)
  elif xvec.ndim == 2:
    msg = ' '.join([fmt % (x) for x in xvec[0, :Dmax]])
  else:
    msg = ' '.join([fmt % (x) for x in xvec[:Dmax]])
  return offset + msg

class TestSSAdditivity(unittest.TestCase):

  def setUp(self):
    nLap = 2
    Data = U.getBarsData('BarsK6V9')
    model, _, _ = U.MakeModelWithFiveTopics(Data)
    DataIterator = bnpy.data.AdmixMinibatchIterator(Data, nBatch=4, nLap=nLap,
                                                    dataorderseed=42)
    outP = dict(**U.outArgs)
    algP = dict(**U.algArgs)
    algP['nLap'] = nLap

    self.learnAlg = bnpy.learnalg.MemoizedOnlineVBLearnAlg(savedir=None, seed=0, 
                              algParams=algP, outputParams=outP)    
  
    self.learnAlg.fit(model, DataIterator)

  def test__global_summary_equals_sum_of_batch_summaries(self):
    SS = self.learnAlg.SS

    SS2 = SS.copy()
    SS2._Fields.setAllFieldsToZero()
    SS2._ELBOTerms.setAllFieldsToZero()

    for batchID, SSchunk in self.learnAlg.SSmemory.items():
      SS2 += SSchunk
    
    for key in SS._FieldDims:
      arr = getattr(SS, key)
      arr2 = getattr(SS2, key)
      print '-------------------', key
      print np2flatstr(arr)
      print np2flatstr(arr2)
      assert np.allclose(arr, arr2)

    print '=============================== ELBO terms'
    for key in SS._ELBOTerms._FieldDims:
      arr = SS.getELBOTerm(key)
      arr2 = SS2.getELBOTerm(key)
      print '-------------------', key
      print np2flatstr(arr)
      print np2flatstr(arr2)
      assert np.allclose(arr, arr2)


class TestSSAdditivity_BirthMove(TestSSAdditivity):

  def setUp(self):
    nLap = 2
    Data = U.getBarsData('BarsK6V9')
    model, _, _ = U.MakeModelWithFiveTopics(Data)
    DataIterator = bnpy.data.AdmixMinibatchIterator(Data, nBatch=4, nLap=nLap,
                                                    dataorderseed=42)
    outP = dict(**U.outArgs)
    algP = dict(**U.algArgs)
    algP['nLap'] = nLap
    algP['birth'] = dict(**U.birthArgs)
    self.learnAlg = bnpy.learnalg.MemoizedOnlineVBLearnAlg(savedir=None, seed=0, 
                              algParams=algP, outputParams=outP)    
  
    self.learnAlg.fit(model, DataIterator)

