import numpy as np

from bnpy.util import as1D
import munkres

def calcHammingDistance(zTrue, zHat):
  zHat = as1D(zHat)
  zTrue = as1D(zTrue)
  return np.sum(zTrue != zHat)

def buildCostMatrix(zHat, zTrue):
  ''' Construct cost matrix for alignment of estimated and true sequences

  Returns
  --------
  CostMatrix : 2D array, size Ktrue x Kest
               CostMatrix[j,k] = count of events across all timesteps,
               where j is assigned, but k is not.
  '''
  zHat = as1D(zHat)
  zTrue = as1D(zTrue)
  Ktrue = int(np.max(zTrue)) + 1
  Kest = int(np.max(zHat)) + 1
  K = np.maximum(Ktrue, Kest)
  CostMatrix = np.zeros((K, K))
  for ktrue in xrange(K):
    for kest in xrange(K):
      CostMatrix[ktrue, kest] = np.sum(np.logical_and(zTrue == ktrue,
                                                      zHat != kest))
  return CostMatrix     

def alignEstimatedStateSeqToTruth(zHat, zTrue, returnInfo=False):
  ''' Relabel the states in zHat to minimize the hamming-distance to zTrue

  Args
  --------
  zHat : 1D array
         each entry is an integer label in {0, 1, ... Kest-1}
  zTrue : 1D array
          each entry is an integer label in {0, 1, ... Ktrue-1}

  Returns
  --------
  zHatAligned : 1D array, relabeled version of zHat that aligns to zTrue
  AInfo : dict of information about the alignment
  '''
  zHat = as1D(zHat)
  zTrue = as1D(zTrue)

  CostMatrix = buildCostMatrix(zHat, zTrue)
  MunkresAlg = munkres.Munkres()
  AlignedRowColPairs = MunkresAlg.compute(CostMatrix)
  zHatA = -1 * np.ones_like(zHat)
  for (ktrue, kest) in AlignedRowColPairs:
    mask = zHat == kest
    zHatA[mask] = ktrue
  if returnInfo:
    return zHatA, dict(CostMatrix=CostMatrix,
                       AlignedRowColPairs=AlignedRowColPairs)
  else:
    return zHatA

def convertStateSeq_flat2list(zFlat, Data):
  ''' Convert flat, 1D array representation of multiple sequences to list
  '''
  zListBySeq = list()
  for n in xrange(Data.nDoc):
    start = Data.doc_range[n]
    stop = Data.doc_range[n+1]
    zListBySeq.append(zFlat[start:stop])
  return zListBySeq

def convertStateSeq_list2flat(zListBySeq, Data):
  ''' Convert nested list representation of multiple sequences to 1D array
  '''
  zFlat = np.zeros(Data.doc_range[-1])
  for n in xrange(Data.nDoc):
    start = Data.doc_range[n]
    stop = Data.doc_range[n+1]
    zFlat[start:stop] = zListBySeq[n]
  return zFlat

def convertStateSeq_list2MAT(zListBySeq):
  ''' Convert nested list representation to MAT friendly format
  '''
  N = len(zListBySeq)
  zObjArr = np.zeros((N,1), dtype=object)
  for n in xrange(N):
    zObjArr[n,0] = np.asarray(zListBySeq[n][:,np.newaxis], dtype=np.int32)
  return zObjArr

def convertStateSeq_MAT2list(zObjArr):
  N = zObjArr.shape[0]
  zListBySeq = list()
  for n in xrange(N):
    zListBySeq.append(np.squeeze(zObjArr[n,0]))
  return zListBySeq