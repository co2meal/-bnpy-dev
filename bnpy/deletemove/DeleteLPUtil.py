import numpy as np
import warnings


def deleteCompFromResp_Renorm(Resp, compIDs):
  ''' Delete comps from responsibility matrix
  '''
  if type(compIDs) == int:
    compIDs = [compIDs]
  assert type(compIDs) == list or type(compIDs) == np.ndarray
  N, K = Resp.shape
  if K == 1:
    raise ValueError('Cannot delete when K==1')
  if len(compIDs) >= K:
    raise ValueError('Cannot delete all components.')

  if len(compIDs) == 1:
    remComps = range(compIDs[0]) + range(compIDs[0] + 1, K)
  else:
    remComps = [k for k in xrange(K) if k not in compIDs]
  Knew = K - len(compIDs)
  assert len(remComps) == Knew

  if Knew == 1:
    return np.ones((N,1), dtype=Resp.dtype)

  Rnew = Resp[:, remComps].copy()
  rMass = np.sum(Rnew, axis=1)
  goodIDs = rMass > 1e-25
  Rnew[goodIDs] /= rMass[goodIDs][:,np.newaxis]
  Rnew[np.logical_not(goodIDs)] = 1./Knew

  assert np.allclose(1.0, np.sum(Rnew, axis=1))
  return Rnew


def deleteCompFromResp_SoftEv(Resp, SoftEv, compIDs):
  ''' Construct proposal for resp with components deleted, via SoftEv rule.

      Args
      -------
      Resp : 2D array, shape N x K
      SoftEv : log probability matrix

      Returns
      -------
      propResp : 2D array, shape N x Knew
      where Knew = K - len(compIDs)
  '''
  if type(compIDs) == int:
    compIDs = [compIDs]
  assert type(compIDs) == list or type(compIDs) == np.ndarray
  N, K = Resp.shape
  if K == 1:
    raise ValueError('Cannot delete when K==1')
  if len(compIDs) >= K:
    raise ValueError('Cannot delete all components.')

  if len(compIDs) == 1:
    remComps = range(compIDs[0]) + range(compIDs[0] + 1, K)
  else:
    remComps = [k for k in xrange(K) if k not in compIDs]
  Knew = K - len(compIDs)
  assert len(remComps) == Knew

  if Knew == 1:
    return np.ones((N,1), dtype=Resp.dtype)

  Rnew = Resp[:, remComps].copy()
  rMass = np.sum(Rnew, axis=1)
  keepIDs = rMass > 0.5
  replaceIDs = np.logical_not(keepIDs)

  SoftEv = SoftEv[replaceIDs][:, remComps].copy()
  SoftEv -= SoftEv.max(axis=1)[:,np.newaxis]
  np.exp(SoftEv, out=SoftEv)
  SoftEv /= SoftEv.sum(axis=1)[:,np.newaxis]

  Rnew[keepIDs] /= rMass[keepIDs][:,np.newaxis]
  Rnew[replaceIDs] = SoftEv
  assert np.allclose(1.0, np.sum(Rnew, axis=1))
  return Rnew



def deleteCompFromResp_SoftEvOverlap(Resp, SoftEv, Data, SS, compIDs):
  ''' Construct proposed resp with components deleted, via SoftEv+overlap rule.

      Args
      -------
      Resp : 2D array, shape N x K
      SoftEv : log probability matrix
      Data : dataset object

      Returns
      -------
      propResp : 2D array, shape N x Knew
      where Knew = K - len(compIDs)
  '''
  if type(compIDs) == int:
    compIDs = [compIDs]
  assert type(compIDs) == list or type(compIDs) == np.ndarray
  N, K = Resp.shape
  if K == 1:
    raise ValueError('Cannot delete when K==1')
  if len(compIDs) >= K:
    raise ValueError('Cannot delete all components.')

  if len(compIDs) == 1:
    remComps = range(compIDs[0]) + range(compIDs[0] + 1, K)
  else:
    remComps = [k for k in xrange(K) if k not in compIDs]
  Knew = K - len(compIDs)
  assert len(remComps) == Knew

  if Knew == 1:
    return np.ones((N,1), dtype=Resp.dtype)

  Rnew = Resp[:, remComps].copy()
  rMass = np.sum(Rnew, axis=1)
  keepIDs = rMass > 0.5
  replaceIDs = np.logical_not(keepIDs)

  SoftEv = SoftEv[replaceIDs][:, remComps].copy()
  SoftEv -= SoftEv.max(axis=1)[:,np.newaxis]
  np.exp(SoftEv, out=SoftEv)
  SoftEv /= SoftEv.sum(axis=1)[:,np.newaxis]

  ## loop over all the docs that had things replaced,
  prevdocID = -1
  remWordCounts = SS.WordCounts[remComps, :]
  remWordSums = SS.SumWordCounts[remComps]
  for rr,replaceID in enumerate(np.flatnonzero(replaceIDs)):
    docID = np.searchsorted(Data.doc_range, replaceID)
    docID = np.maximum(0, docID-1)
    docWordTypes = Data.word_id[Data.doc_range[docID]:Data.doc_range[docID+1]]
    if docID != prevdocID:
      overlapScore = remWordCounts[:,docWordTypes].sum(axis=1) / remWordSums
      overlapScore /= overlapScore.sum()
    print docID
    print overlapScore
    SoftEv[rr] *= overlapScore
    prevdocID = docID
  # Normalize it, since multiplying ruined the sum-to-one constraint
  SoftEv /= SoftEv.sum(axis=1)[:,np.newaxis]

  Rnew[keepIDs] /= rMass[keepIDs][:,np.newaxis]
  Rnew[replaceIDs] = SoftEv
  assert np.allclose(1.0, np.sum(Rnew, axis=1))
  return Rnew
