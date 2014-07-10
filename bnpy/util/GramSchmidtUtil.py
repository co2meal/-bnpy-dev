'''
'''
import numpy as np

hasRandProjModule = False
try:
  from random_projection import Random_Projection
  hasRandProjModule = True
except ImportError:
  pass

def FindAnchorsForSizeKBasis(Q, K, lowerDim=None, seed=0, candidateRows=None):
  ''' Find K rows of Q that best span the entire rowspace of Q

      Args
      --------
      Q : N x V matrix
      K : int, size of desired basis
      candidateRows : list of ints, identifying distinct rows of Q

      Returns
      --------
      basisRows : 1D array, size K, type int32
                  basisRows[k] gives row of Q that forms k-th basis
      dist : 1D array, size K
  '''
  if candidateRows is None:
    Q = Q.copy()
  else:
    candidateRows = np.asarray(candidateRows, dtype=np.int32)
    Q = Q[candidateRows,:].copy()

  Q /= Q.sum(axis=1)[:,np.newaxis]

  if hasRandProjModule and lowerDim is not None:
    if lowerDim > 0 and lowerDim < Q.shape[1]:
      PRNG = np.random.RandomState(seed)
      Q = Random_Projection(Q.T, lowerDim, PRNG)
      Q = Q.T

  N, V = Q.shape
  dist = np.zeros(K)
  B = np.zeros((K-1, V), dtype=np.float64)
  basisRows = np.zeros(K, dtype=np.int32)

  # Temporarily store the vector in first row of B
  basisRows[0], dist[0], B[0,:] = FindFarthestFromOrigin(Q)
  Q -= B[0,:][np.newaxis,:]

  # basisRows[0] is now the "origin" of our coordinate system
  basisRows[1], dist[1], B[0,:] = FindFarthestFromOrigin(Q)
  B[0,:] /= np.sqrt(np.inner(B[0], B[0]))

  dist[0] = dist[1] # dist[0] is kind of nonsensical, so just duplicate
  distToOrigin = np.zeros(Q.shape[0])
  for k in xrange(1, K-1):
    #qVec = np.dot(Q, B[k-1])
    #Q -= qVec[:,np.newaxis] * B[k-1][np.newaxis,:]
    for rowID in xrange(Q.shape[0]):
      Q[rowID] -= np.inner(Q[rowID], B[k-1]) * B[k-1]
      distToOrigin[rowID] = np.inner(Q[rowID], Q[rowID])
    basisRows[k+1], dist[k+1], B[k,:] = FindFarthestFromOrigin(Q, distToOrigin)
    B[k,:] /= np.sqrt(np.inner(B[k], B[k]))

  if candidateRows is not None:
    basisRows = candidateRows[basisRows]
  return basisRows, dist

def FindFarthestFromOrigin(Q, distToOrigin=None):
  if distToOrigin is None:
    distToOrigin = np.zeros(Q.shape[0])
    for rowID in xrange(Q.shape[0]):
      distToOrigin[rowID] = np.inner(Q[rowID], Q[rowID])
    # above loop is faster than this, due to no memory allocation
    #distToOrigin = np.sum(np.square(Q), axis=1)
  maxID = np.argmax(distToOrigin)
  maxDist = distToOrigin[maxID]
  return maxID, maxDist, Q[maxID]

###########################################################
###########################################################
def FindAnchorsForExpandedBasis(Q, Topics, Kextra, lowerDim=None, 
                                           seed=0, candidateRows=None):
  ''' Find Kextra rows of Q that best expand the given existing basis.

      Returns
      -------
      newRows : 1D array of int32, size Kextra
                where Q[newRows] and Topics together span the basis
                      that best covers the row space of Q
  '''
  if candidateRows is not None:
    candidateRows = np.asarray(candidateRows, dtype=np.int32)
    Q = Q[candidateRows]
  Q = Q / np.sum(Q,axis=1)[:,np.newaxis]
  Topics = Topics / np.sum(Topics, axis=1)[:,np.newaxis]

  if lowerDim is not None:
    PRNG = np.random.RandomState(seed)
    Q = Random_Projection(Q.T, lowerDim, PRNG)
    Q = Q.T
    PRNG = np.random.RandomState(seed)
    Topics = Random_Projection(Topics.T, lowerDim, PRNG)
    Topics = Topics.T

  # Allocate basis
  N, V = Q.shape
  K, V2 = Topics.shape
  assert V == V2
  B = np.zeros( (K+Kextra-1, V))

  ## Build basis for existing Q to reflect the existing Topics
  ## Basically, play usual alg forward, as if we selected each row of Topics
  Q -= Topics[0][np.newaxis,:]
  Topics = Topics - Topics[0][np.newaxis,:]

  B[0,:] = Topics[1]
  B[0,:] /= np.sqrt(np.inner(B[0],B[0]))

  for k in xrange(1, K-1):
    Q -= np.dot(Q, B[k-1])[:,np.newaxis] * B[k-1][np.newaxis,:]
    Topics -= np.dot(Topics, B[k-1])[:,np.newaxis] * B[k-1][np.newaxis,:]
    B[k,:] = Topics[k+1]  
    B[k,:] /= np.sqrt(np.inner(B[k],B[k]))

  ## Now, expand by adding Kextra new basis vectors
  newRows = np.zeros(Kextra, dtype=np.int32)
  for knew in xrange(Kextra):
    k = K + knew - 1
    for i in xrange(Q.shape[0]):
      Q[i] -= np.inner(Q[i], B[k-1]) * B[k-1]
    newRows[knew], _, B[k] = FindFarthestFromOrigin(Q)
    B[k,:] /= np.sqrt(np.inner(B[k], B[k]))

  if candidateRows is not None:
    newRows = candidateRows[newRows]
  return newRows

########################################################### original impl
###########################################################
def _FindBasisOrig(Q, K, candidateRows=None):
    ''' direct port of original implementation
    '''
    if candidateRows is None:
        candidateRows = np.arange(Q.shape[0])

    M_orig = Q.copy()
    M_orig /= M_orig.sum(axis=1)[:,np.newaxis]

    r = K
    candidates = candidateRows

    n = M_orig[:, 0].size
    dim = M_orig[0, :].size

    M = M_orig.copy()

    # stored recovered anchor words
    anchor_words = np.zeros((r, dim))
    anchor_indices = np.zeros(r, dtype=np.int)

    # store the basis vectors of the subspace spanned by the anchor word vectors
    basis = np.zeros((r-1, dim))


    # find the farthest point p1 from the origin
    max_dist = 0
    for i in candidates:
        dist = np.dot(M[i], M[i])
        if dist > max_dist:
            max_dist = dist
            anchor_words[0] = M_orig[i]
            anchor_indices[0] = i

    # let p1 be the origin of our coordinate system
    #for i in range(0, n):
    for i in candidates:
        M[i] = M[i] - anchor_words[0]


    # find the farthest point from p1
    max_dist = 0
    #for i in range(0, n):
    for i in candidates:
        dist = np.dot(M[i], M[i])
        if dist > max_dist:
            max_dist = dist
            anchor_words[1] = M_orig[i]
            anchor_indices[1] = i
            basis[0] = M[i]/np.sqrt(np.dot(M[i], M[i]))


    # stabilized gram-schmidt which finds new anchor words to expand our subspace
    for j in range(1, r - 1):

        # project all the points onto our basis and find the farthest point
        max_dist = 0
        #for i in range(0, n):
        for i in candidates:
            M[i] = M[i] - np.dot(M[i], basis[j-1])*basis[j-1]
            dist = np.dot(M[i], M[i])
            if dist > max_dist:
                max_dist = dist
                anchor_words[j + 1] = M_orig[i]
                anchor_indices[j + 1] = i
                basis[j] = M[i]/np.sqrt(np.dot(M[i], M[i]))
    return anchor_indices, basis



"""
def ExpandBasisByKextra2(Q, Topics, Kextra, candidateRows=None):
  N, V = Q.shape
  K, V2 = Topics.shape
  assert V == V2

  B = np.zeros( (K+Kextra-1, V))
  if candidateRows is None:
    candidateRows = np.arange(N)
  Q = Q[candidateRows].copy()

  ## Build basis for existing Q to reflect the existing Topics
  ## Basically, play FindBasis forward, as if we selected each row of Topics
  Q -= Topics[0][np.newaxis,:]
  Topics = Topics - Topics[0][np.newaxis,:]

  B[0,:] = Topics[1]
  B[0,:] /= np.sqrt(np.inner(B[0],B[0]))

  for k in xrange(1, K-1):
    Q -= np.dot(Q, B[k-1])[:,np.newaxis] * B[k-1][np.newaxis,:]
    Topics -= np.dot(Topics, B[k-1])[:,np.newaxis] * B[k-1][np.newaxis,:]
    B[k,:] = Topics[k+1]  
    B[k,:] /= np.sqrt(np.inner(B[k],B[k]))

  ## Now, expand to new topics
  newRows = np.zeros(Kextra, dtype=np.int32)
  
  for knew in xrange(Kextra):
    k = K + knew - 1
    newRows[knew], _, B[k] = AddNextVectorToBasis(k, B, Q)
    B[k,:] /= np.sqrt(np.inner(B[k], B[k]))

  return newRows

def FindSizeKBasis2(Q, K, candidateRows=None):
  Q = Q.copy()  
  Q /= Q.sum(axis=1)[:,np.newaxis]

  N, V = Q.shape
  if candidateRows is None:
    candidateRows = np.arange(N)

  dist = np.zeros(K)
  B = np.zeros( (K-1, V), dtype=np.float64)
  basisRows = np.zeros(K, dtype=np.int32)

  basisRows[0], dist[0], originVec = FindFarthestFromOrigin2(Q, candidateRows)
  Q[candidateRows,:] -= originVec[np.newaxis,:]

  # basisRows[0] is now the "origin" of our coordinate system
  basisRows[1], dist[1], B[0,:] = FindFarthestFromOrigin2(Q, candidateRows)
  B[0,:] /= np.sqrt(np.inner(B[0], B[0]))

  dist[0] = dist[1] # dist[0] is kind of nonsensical, so just duplicate 
  for k in xrange(1, K-1):
    basisRows[k+1], dist[k+1], B[k,:] = AddNextVectorToBasis2(k, B, Q, candidateRows)
    B[k,:] /= np.sqrt(np.inner(B[k], B[k]))
  
  return basisRows, dist

def FindFarthestFromOrigin2(Q, candidateRows):
  dist = np.sum(np.square(Q[candidateRows,:]), axis=1)
  maxID = np.argmax(dist)
  maxDist = dist[maxID]
  maxID = candidateRows[maxID]
  return maxID, maxDist, Q[maxID]

def AddNextVectorToBasis2(k, B, Q, candidateRows=None):
  if candidateRows is None:
    candidateRows = np.arange(Q.shape[0])
  for i in candidateRows:
    Q[i] -= np.inner(Q[i], B[k-1]) * B[k-1]
  return FindFarthestFromOrigin2(Q, candidateRows)
"""

