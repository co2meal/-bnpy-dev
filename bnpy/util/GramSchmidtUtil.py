'''
'''
import numpy as np

def FindSizeKBasis(Q, K, candidateRows=None):
  '''
      Args
      --------
      Q : N x V matrix
      K : int, size of desired basis
      candidateRows : list of ints, identifying distinct rows of Q

      Returns
      --------
      basisRows : 1D array, size K, type int32
                  basisRows[k] gives row of Q that forms k-th basis
      B : K-1 x V matrix
  '''
  Q = Q.copy()  
  N, V = Q.shape
  if candidateRows is None:
    candidateRows = np.arange(N)

  dist = np.zeros(K)
  B = np.zeros( (K-1, V), dtype=np.float64)
  basisRows = np.zeros(K, dtype=np.int32)

  basisRows[0], dist[0], originVec = FindFarthestFromOrigin(Q, candidateRows)
  Q[candidateRows,:] -= originVec[np.newaxis,:]

  # basisRows[0] is now the "origin" of our coordinate system
  basisRows[1], dist[1], B[0,:] = FindFarthestFromOrigin(Q, candidateRows)
  B[0,:] /= np.sqrt(np.inner(B[0], B[0]))

  dist[0] = dist[1] # dist[0] is kind of nonsensical, so just duplicate 
  for k in xrange(1, K-1):
    basisRows[k+1], dist[k+1], B[k,:] = AddNextVectorToBasis(k, B, Q, candidateRows)
    B[k,:] /= np.sqrt(np.inner(B[k], B[k]))
  
  return basisRows, dist

def FindFarthestFromOrigin(Q, candidateRows):
  dist = np.sum(np.square(Q[candidateRows,:]), axis=1)
  maxID = np.argmax(dist)
  maxDist = dist[maxID]
  maxID = candidateRows[maxID]
  return maxID, maxDist, Q[maxID]

def AddNextVectorToBasis(k, B, Q, candidateRows=None):
  if candidateRows is None:
    candidateRows = np.arange(Q.shape[0])
  for i in candidateRows:
    Q[i] -= np.inner(Q[i], B[k-1]) * B[k-1]
  return FindFarthestFromOrigin(Q, candidateRows)

def ExpandBasisByKextra(Q, Topics, Kextra, candidateRows=None):
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

########################################################### original impl
###########################################################
def _FindBasisOrig(Q, K, candidateRows=None):
    ''' direct port of original implementation
    '''
    if candidateRows is None:
        candidateRows = np.arange(Q.shape[0])

    M_orig = Q
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