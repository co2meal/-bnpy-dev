'''
BinTreeUtil.py

Provides sum-product algorithm for binary HMTs
'''
import numpy as np
import math


def SumProductAlg_BinTree(PiInit, PiMat, logSoftEv):
  '''Execute sum-product algorithm for binary trees
  '''
  PiInit, PiMat, K = _parseInput_TransParams(PiInit, PiMat)
  logSoftEv = _parseInput_SoftEv(logSoftEv, K)
  N = logSoftEv.shape[0]

  SoftEv, lognormC = expLogLik(logSoftEv)
  umsg, margPrObs, fmsg = UpwardPass(PiInit, PiMat, SoftEv)
  dmsg = DownwardPass(PiInit, PiMat, SoftEv, umsg, fmsg)
  resp = np.empty( (N,K) )
  start = find_last_nonleaf_node(N)
  for n in xrange(N):
    message = np.ones( K )
    if n > start-1:
      message *= SoftEv[n]
    else:
      message *= fmsg[n,:]
    message *= dmsg[n,:]
    resp[n,:] = message / np.sum(message)
  respPair = np.zeros( (N,K,K) )
  for n in xrange(1,N,1):
    parent = get_parent_index(n)
    branch = get_branch(n)
    up = np.ones( K )
    up *= dmsg[parent,:]
    eps = np.finfo(float).eps
    divisor = np.ones( K ) * umsg[n, :]
    divisor[divisor < eps] = eps
    up *= fmsg[parent,:] / divisor
    down = np.ones( K )
    if n > start-1:
      down *= SoftEv[n,:]
    else:
      down *= fmsg[n,:]
    respPair[n] = PiMat[branch,:,:] * np.outer(up, down)
    respPair[n] /= np.sum(respPair[n])
  logMargPrSeq = np.log(margPrObs).sum() + lognormC.sum()
  return resp, respPair, logMargPrSeq


def UpwardPass(PiInit, PiMat, SoftEv):
  '''Propagate messages upwards along the tree, starting from the leaves
  '''
  N = SoftEv.shape[0]
  K = PiInit.size
  umsg = np.ones( (N, K) )
  margPrObs = np.empty( N )
  start = find_last_nonleaf_node(N)
  fmsg = np.ones ( (start, K) )
  for n in xrange(N-1, 0, -1):
    branch = get_branch(n)
    if n > start-1:
      message = np.ones( K )
      message *= SoftEv[n]
      message = np.dot(PiMat[branch,:,:], message)
      margPrObs[n] = np.sum(message)
      umsg[n, :] = message / margPrObs[n]
    else:
      children = get_children_indices(n, N)
      message = np.ones( K )
      for child in children:
        message *= umsg[child,:]
      message *= SoftEv[n]
      fmsg[n] = np.ones( K ) * message
      message = np.dot(PiMat[branch,:,:], message)
      margPrObs[n] = np.sum(message)
      umsg[n,:] = message / margPrObs[n]
  message = np.ones( K )
  for child in xrange(1, 3):
    message *= umsg[child, :]
  message *= SoftEv[0]
  fmsg[0] = np.ones( K ) * message
  message *= PiInit
  margPrObs[0] = np.sum(message)
  return umsg, margPrObs, fmsg


def DownwardPass(PiInit, PiMat, SoftEv, umsg):
  '''Propagate messages downwards along the tree, starting from the root
  '''
  N = SoftEv.shape[0]
  K = PiInit.size
  PiTMat = np.empty( (2,K,K) )
  for d in xrange(0, 2):
    PiTMat[d,:,:] = PiMat[d,:,:].T
  dmsg = np.empty( (N,K) )
  for n in xrange( 0, N ):
    if n == 0:
      dmsg[n] = PiInit
    else:
      parent = get_parent_index(n)
      branch = get_branch(n)
      message = np.ones( K )
      eps = np.finfo(float).eps
      divisor = np.ones( K ) * umsg[n, :]
      divisor[divisor < eps] = eps
      message = fmsg[parent,:] / divisor
      message *= dmsg[parent,:]
      message = np.dot(PiTMat[branch,:,:], message)
      dmsg[n,:] = message / np.sum(message)
  return dmsg

########################################################### Brute Force
###########################################################
def calcRespByBruteForce(PiInit, PiMat, logSoftEv):
  ''' Calculate marginal state-assignments for binary trees via brute force.

    Returns
    -------
    resp : 2D array, size N x K
    respPair : 3D array, size N x K x K
    logPrObs : scalar log probability
  '''
  N = logSoftEv.shape[0]
  PiInit, PiMat, K = _parseInput_TransParams(PiInit, PiMat)
  logSoftEv = _parseInput_SoftEv(logSoftEv, K)
  SoftEv, lognormC = expLogLik(logSoftEv)
  if N > 10:
    raise ValueError("Brute force is too expensive for N=%d!" % (N))
  resp = np.zeros((N,K))
  respPair = np.zeros((N,K,K))
  margPrObs = 0
  for configID in xrange(K ** N):
    Ztree = makeZTree(configID, N, K)
    prTree = calcProbOfTree(Ztree, PiInit, PiMat, SoftEv)
    margPrObs += prTree
    for n in range(N):
      resp[n, Ztree[n]] += prTree
      if n > 0:
        pa = get_parent_index(n)
        respPair[n, Ztree[pa], Ztree[n]] += prTree
  resp /= resp.sum(axis=1)[:,np.newaxis]
  for n in range(1, N):
    respPair[n] /= respPair[n].sum()

  return resp, respPair, np.log(margPrObs) + np.sum(lognormC)

def makeZTree(configID, N, K):
  '''Create configuration of hidden state variables for all nodes in tree.

    Examples
    --------
    >>> makeZTree(0, 3, 2)
    [0, 0, 0]
       >>> makeZTree(7, 3, 2)
       [1, 1, 1]
       >>> makeZTree(7, 4, 2)
       [0, 1, 1, 1]
       >>> makeZTree(7, 4, 3)
       [0, 0, 2, 1]
    '''
  Ztree = np.zeros( N, dtype=np.int32)
  for n in range(N-1, -1, -1):
    posID = N - n - 1
    Ztree[posID] = configID / (K**n)   
    configID = configID - Ztree[posID] * (K**n)
  return Ztree

def calcProbOfTree(Ztree, PiInit, PiMat, SoftEv):
  ''' Calculate joint prob of assignments and observations for entire tree.
  '''
  N = SoftEv.shape[0]
  prTree = PiInit[Ztree[0]] * SoftEv[0,Ztree[0]]
  for n in xrange(1, N):
    parent = get_parent_index(n)
    get_branch(n)
    prTree *= PiMat[branch, Ztree[parent], Ztree[n]] * SoftEv[n,Ztree[n]]
  return prTree

########################################################### tree utilities
###########################################################
def get_parent_index(n):
  if n == 0:
    return None # it is a root
  elif n % 2 == 0:
    return (n-1)/2
  else:
    return n/2

def get_children_indices(n, N):
  if 2 * n + 1 >= N:
    return [] # it is a leaf
  else:
    return [2 * n + j + 1 for j in range(2)]

def get_branch(child_index):
  '''Find on which branch of its parent this child lies
  '''
  if (child_index%4 == 0):
    return 3
  else:
    return (child_index%4 - 1)

def find_last_nonleaf_node(N):
    '''Get the index of last nonleaf node in the data
    '''
  if N == 1:
    return None
  else:
    height = 1
    total = 1
    while (total + 2**height) < N:
      total += 2**height
      height += 1
    return total

########################################################### expLogLik
###########################################################
def expLogLik(logSoftEv, axis=1):
  ''' Return element-wise exp of input log likelihood
        guaranteed not to underflow
    
        Returns
        --------
        SoftEv : 2D array, size NxK
                  equal to exp(logSoftEv), up to prop constant for each row
        lognormC : 1D array, size N
                gives log of the prop constant for each row
    '''
  lognormC = np.max(logSoftEv, axis)
  if axis==0:
    logSoftEv = logSoftEv - lognormC[np.newaxis,:]
  elif axis==1:
    logSoftEv = logSoftEv - lognormC[:,np.newaxis]
  SoftEv = np.exp(logSoftEv)
  return SoftEv, lognormC

########################################################### Parse input
###########################################################
def _parseInput_TransParams(PiInit, PiMat):
  PiInit = np.asarray(PiInit, dtype=np.float64)
  PiMat = np.asarray(PiMat, dtype=np.float64)
  assert PiInit.ndim == 1
  assert PiMat.ndim == 3
  I, J, K = PiMat.shape
  assert I == 2
  assert J == K
  return PiInit, PiMat, K

def _parseInput_SoftEv(logSoftEv, K):
  logSoftEv = np.asarray(logSoftEv, dtype=np.float64)
  Nl, Kl = logSoftEv.shape
  assert Kl == K
  return logSoftEv
