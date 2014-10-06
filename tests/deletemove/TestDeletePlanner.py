import numpy as np
import unittest 

from bnpy.suffstats import SuffStatBag
from bnpy.deletemove import DeletePlanner

class TestK8(unittest.TestCase):

  def setUp(self):
    Nvec = np.asarray([500, 600, 0, 10, 700, 6, 800, 0])
    self.selectIDs = [2, 7, 5, 3]
    SS = SuffStatBag(K=Nvec.size)
    SS.setField('N', Nvec, dims='K')
    SS.uIDs = np.arange(Nvec.size)
    self.SS = SS
    self.deleteSizeThr = 100
    self.dtargetMaxSize = 1000

  def test_makePlan(self):
    print ''
    Plans = DeletePlanner.makePlans(self.SS, 
                                  deleteSizeThr=self.deleteSizeThr,
                                  dtargetMaxSize=self.dtargetMaxSize)
    print Plans
    if len(Plans) > 0:
      selectIDs =  Plans[0]['selectIDs']
    else:
      selectIDs = []
    print '  RESULT: ', selectIDs
    print 'EXPECTED: ', self.selectIDs
    assert np.allclose(self.selectIDs, selectIDs)

class TestEmpty(TestK8):

  def setUp(self):
    Nvec = np.asarray([500, 600, 700, 800, 10000])
    self.selectIDs = []
    SS = SuffStatBag(K=Nvec.size)
    SS.setField('N', Nvec, dims='K')
    SS.uIDs = np.arange(Nvec.size)
    self.SS = SS
    self.deleteSizeThr = 100
    self.dtargetMaxSize = 1000

class TestTooMany(TestK8):

  def setUp(self):
    Nvec = np.asarray([500, 600, 700, 800, 10000])
    self.selectIDs = [0, 1]
    SS = SuffStatBag(K=Nvec.size)
    SS.setField('N', Nvec, dims='K')
    SS.uIDs = np.arange(Nvec.size)
    self.SS = SS
    self.deleteSizeThr = 15000
    self.dtargetMaxSize = 1500
