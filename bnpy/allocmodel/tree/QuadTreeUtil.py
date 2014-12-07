'''
QuadTreeUtil.py 

Provides sum-product and brute-force algorithms for HMTs
'''
import numpy as np
import math
from bnpy.util import EPS


def SumProductAlg_QuadTree(PiInit, PiMat, logSoftEv):
  '''Execute sum-product algorithm given HMT state
     transition params and log likelihoods of each observation

     Args
     -------
     piInit : 1D array, size K
            initial transition distribution to each of the K states
            must be valid probability vector (positive entries, sums to one)
     piMat  : 4 2D arrays, size KxK
            piMat[i, j] is the transition distribution on branch i from state j to all
            K states. each row must be probability vector (positive entries, sums to one)
     logSoftEv : 2D array, size NxK
            logSoftEv[n] := log p( x[n] | z[nk] = 1)
                         log likelihood of observation n under state k
     Returns
     -------
     resp : 2D array, size N x K
            resp[n,k] = marg. prob. that step t assigned to state K
                        p( z[n,k] = 1 | x[1], x[2], ... x[N])
     respPair : 2D array, size N x K x K
            respPair[n,j,k] = marg. prob. that both
                              * node n assigned to state k
                              * parent of node n assigned to state j
                        p( z[pa(n),j] = 1, z[n,k] = 1 | x[1], x[2], ... x[N])
            respPair[0,:,:] is undefined, but kept to match indexing consistent.
      logMargPrSeq : scalar real
            logMargPrSeq = joint log probability of the observed sequence
                        log p( x[1], x[2], ... x[T] ) 

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
    Args
     -------
     piInit : 1D array, size K
            initial transition distribution to each of the K states
            must be valid probability vector (positive entries, sums to one)
     piMat  : 4 2D array, size 4xKxK
            piMat[i,j] is transition distribution on branch ifrom state j to all K states.
            each row must be probability vector (positive entries, sums to one)
     SoftEv : 2D array, size NxK
            SoftEv[n] := p( x[n] | z[nk] = 1)
                         likelihood of observation n under state k
                         given up to an additive constant for each n
     Returns
     -------
        umsg : 2D array, size N x K
                  probability of state k on latent variable n, given all 
                  observations from its predecessors and its observation
                  umsg[n,k] = p( z[n,k] = 1 | x[c(c(n))]...x[c(n)] ... x[n] )
        margPrObs : 1D array, size N
                  margPrObs[n] = p( x[n] | x[c(c(n))]...x[c(n)] )
        fmsg : 2D array, size T x K, where T is the number of nonleaf nodes
                  fmsg[n] = p(x[n]|z[n]) * umsg[c(n)[1]] * umsg[c(n)[2]] * umsg[c(n)[3]] * umsg[c(n)[4]], 
                  where c(n)[k] is the kth child of node n

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
  for child in xrange(1, 5):
    message *= umsg[child, :]
  message *= SoftEv[0]
  fmsg[0] = np.ones( K ) * message
  message *= PiInit
  margPrObs[0] = np.sum(message)
  return umsg, margPrObs, fmsg


def DownwardPass(PiInit, PiMat, SoftEv, umsg, fmsg):
  '''Propagate messages downwards along the tree, starting from the root

    Args
     -------
    piInit : 1D array, size K
            initial transition distribution to each of the K states
            must be valid probability vector (positive entries, sums to one)
    piMat  : 4 2D array, size 4xKxK
            piMat[i,j] is transition distribution on branch ifrom state j to all K states.
            each row must be probability vector (positive entries, sums to one)
    SoftEv : 2D array, size NxK
            SoftEv[n] := p( x[n] | z[nk] = 1)
                         likelihood of observation n under state k
                         given up to an additive constant for each n
    umsg : 2D array, size N x K
          probability of state k on latent variable n, given all 
          observations from its predecessors and its observation
          umsg[n,k] = p( z[n,k] = 1 | x[c(c(n))]...x[c(n)] ... x[n] )
    fmsg : 2D array, size T x K, where T is the number of nonleaf nodes
          fmsg[n] = p(x[n]|z[n]) * umsg[c(n)[1]] * umsg[c(n)[2]] * umsg[c(n)[3]] * umsg[c(n)[4]], 
          where c(n)[k] is the kth child of node n
     Returns
     -------
        dmsg : 2D array, size N x K
                  dmsg[n,k] = p( x[p(n)], x[p(p(n))], ... x[1] |  z[n,k] = 1 )
  '''
  N = SoftEv.shape[0]
  K = PiInit.size
  PiTMat = np.empty( (4,K,K) )
  for d in xrange(0, 4):
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
  if N > 21:
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
    branch = get_branch(n)
    prTree *= PiMat[branch, Ztree[parent], Ztree[n]] * SoftEv[n,Ztree[n]]
  return prTree

def calcEntropyFromResp(resp, respPair, Data, eps=1e-100):
  ''' Calculate entropy E_q(z) [ log q(z) ] for all trees
  '''
  startLocIDs = Data.doc_range[:-1]

  idx = filter(lambda x: x not in set(startLocIDs), xrange(np.size(respPair,0)))
  sigma = respPair / (respPair.sum(axis=2)[:,:,np.newaxis] + eps)
  firstH = -1 * np.sum(resp[startLocIDs] * np.log(resp[startLocIDs]+eps))
  restH = -1 * np.sum(respPair[idx,:,:] * np.log(sigma[idx,:,:] + EPS))
  return firstH + restH

def calcEntropyFromResp_bySeq(resp, respPair, Data, eps=1e-100):
  totalH = 0
  for n in xrange(Data.nDoc):
    start = Data.doc_range[n]
    stop = Data.doc_range[n+1]
    resp_n = resp[start:stop]
    respPair_n = respPair[start:stop]

    # sigma_n : conditional prob of each adjacent pair of states
    # sums to one over the final dimension: sigma_n[t, j, :].sum()
    sigma_n = respPair_n / (respPair_n.sum(axis=2)[:,:,np.newaxis] + eps)

    # Entropy of the first step
    firstH_n = -1 * np.inner(resp_n[0], np.log(resp_n[0] + eps))
    
    # Entropy of the remaining steps 2, 3, ... T
    restH_n = -1 * np.sum(respPair_n * np.log(sigma_n + eps))
    totalH += firstH_n + restH_n
  return totalH

########################################################### tree utilities
###########################################################
def get_parent_index(child_index):
  if child_index == 0:
    return None #it is a root
  elif child_index%4 == 0:
    return (child_index-1)/4
  else:
    return child_index/4

def get_children_indices(parent, N):
  if 4*parent+1 > N:
    return []
  else:
    myList = [4*parent+j+1 for j in range(4)]
    return myList

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
    while (total + 4**height) < N:
      total += 4**height
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
  K0 = PiInit.shape[0]
  assert PiMat.ndim == 3
  I, J, K = PiMat.shape
  assert J == K
  assert K0 == K
  assert I == 4
  return PiInit, PiMat, K

def _parseInput_SoftEv(logSoftEv, K):
  logSoftEv = np.asarray(logSoftEv, dtype=np.float64)
  Nl, Kl = logSoftEv.shape
  assert Kl == K
  return logSoftEv
