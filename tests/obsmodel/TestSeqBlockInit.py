'''
'''
from matplotlib import pylab
import numpy as np
import unittest
import copy

import bnpy
from bnpy.init.FromLP import initHardLP_SeqAllocContigBlocks

class TestSeqBlockInit(unittest.TestCase):

  def setUp(self):
    self.seed = np.random.choice(10000)
    PriorSpec = dict(ECovMat='eye', sF=0.1, nu=0, VMat='same', MMat='eye')
    APriorDict = dict(alpha=0.5, gamma=10.0, hmmKappa=0)
    obsModel = bnpy.obsmodel.GaussObsModel('VB', D=3, **PriorSpec)
    aModel = bnpy.allocmodel.HDPHMM('VB', APriorDict)
    hmodel = bnpy.HModel(aModel, obsModel)

    import ToyARK13
    Data = ToyARK13.get_data(seed=self.seed, nSeq=2, T=500)

    self.hmodel = hmodel
    self.obsModel = obsModel
    self.Data = Data
    self.PRNG = np.random.RandomState(self.seed)    


  def test_sacbLP_init(self):
    Data = self.Data
    LP = initHardLP_SeqAllocContigBlocks(Data, self.obsModel)
    nPlot = np.minimum(Data.nDoc, 8)

    LP['Z'] = bnpy.util.StateSeqUtil.alignEstimatedStateSeqToTruth(LP['Z'],
                                                         Data.TrueParams['Z'])
    for n in xrange(nPlot):
      start = Data.doc_range[n]
      stop = Data.doc_range[n+1]
      T = stop - start
      Ztru_n = Data.TrueParams['Z'][start:stop]
      Zest_n = LP['Z'][start:stop]
      image = np.vstack([Ztru_n, Zest_n])

      pylab.subplot(nPlot, 1, n+1)
      pylab.imshow(image,
                   interpolation='nearest', aspect=T/50.0,
                   vmin=0, vmax=LP['Z'].max())
      pylab.yticks([])
    pylab.show(block=False)
    from IPython import embed; embed()

  '''
  def test_seq_init(self, blockLen=20):
    print ''
    oModel = self.oModel
    Data = self.Data

    start = 0
    stop = Data.doc_range[1]
    T = stop - start
    nBlocks = T // blockLen
    leftoverLen = T - blockLen * nBlocks
    if leftoverLen > 0:
      xtrastartLen = self.PRNG.choice(leftoverLen) 
    else:
      xtrastartLen = 0

    Zest = -1 * np.ones(T, dtype=np.int32)
    kID = 0
    SSagg = None
    for blockID in xrange(nBlocks):
        if blockID == 0:
          a = 0
          b = a + blockLen + xtrastartLen
        elif blockID == nBlocks:
          a = b
          b = stop
        else:
          a = b
          b = a + blockLen

        SSab = oModel.calcSummaryStatsForContigBlock(Data, a=a, b=b)
        if blockID > 1:
          ELBOgap = oModel.calcHardMergeGap_SpecificPairSS(SScur, SSab)
          if ELBOgap >= 0:
            # Positive value means we prefer to merge, so keep going!
            SScur += SSab
          else:
            # Try to combine curSS with some existing comp
            if SSagg is not None:
              mPairIDs = [(k, kID) for k in range(SSagg.K)]
              SSagg.insertComps(SScur)
              oModel.update_global_params(SSagg)
              ELBOgaps = oModel.calcHardMergeGap_SpecificPairs(SSagg, mPairIDs)
              bestID = np.argmax(ELBOgaps)
              if ELBOgaps[bestID] > 0:
                SSagg.mergeComps(*mPairIDs[bestID])
              kID = SSagg.K - 1
            else:
              SSagg = SScur
            SScur = SSab.copy() # make a soft copy / alias
            kID += 1
        else:
          SScur = SSab.copy()
        Zest[a:b] = kID
    '''
