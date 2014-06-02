import numpy as np
import warnings

import DeleteMoveStickBreak as DMSB

def MakeLP(Data, model, LP, DeleteInfo, **kwargs):
  ''' Create new local parameters from given generic LP dict
  '''
  ktarget = DeleteInfo['ktarget']

  if 'word_variational' in LP and 'U1' in LP:
    LP = DMSB.construct_LP_with_comps_removed(Data, model, ktarget, LP,
                                           neighbors=None,
                                           #neighbors=DeleteInfo['neighbors'],
                                           **kwargs)
  return LP

def delete_comps_from_resp_matrix(Resp, compIDs):
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