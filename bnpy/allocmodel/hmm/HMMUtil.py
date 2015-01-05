'''
HMMUtil.py

Provides standard message-passing algorithms for inference in HMMs,
  such as the forward-backward algorithm

Intentionally separated from rest of HMM code, so that we can swap in 
  any fast routine for this calculation with ease.
'''
import numpy as np
from bnpy.util import EPS

from bnpy.util.NumericUtil import Config as PlatformConfig
from lib.LibFwdBwd import FwdAlg_cpp, BwdAlg_cpp

def FwdBwdAlg(PiInit, PiMat, logSoftEv):
  '''Execute forward-backward message passing algorithm
       given HMM state transition params and log likelihoods of each observation

     Args
     -------
     piInit : 1D array, size K
            initial transition distribution to each of the K states
            must be valid probability vector (positive entries, sums to one)
     piMat  : 2D array, size KxK
            piMat[j] is transition distribution from state j to all K states.
            each row must be probability vector (positive entries, sums to one)
     logSoftEv : 2D array, size TxK
            logSoftEv[t] := log p( x[t] | z[tk] = 1)
                         log likelihood of observation t under state k
                         if given exactly, 
                           * resp, respPair will be exact
                           * logMargPrSeq will be exact
                         if given up to an additive constant,
                           * resp, respPair will be exact
                           * logMargPrSeq will be off by an additive constant
     Returns
     -------
     resp : 2D array, size T x K
            resp[t,k] = marg. prob. that step t assigned to state K
                        p( z[t,k] = 1 | x[1], x[2], ... x[T])
     respPair : 2D array, size T x K x K
            respPair[t,j,k] = marg. prob. that both
                              * step t-1 assigned to state j
                              * step t assigned to state k
                        p( z[t-1,j] = 1, z[t,k] = 1 | x[1], x[2], ... x[T])
            respPair[0,:,:] is undefined, but kept to match indexing consistent.

     logMargPrSeq : scalar real
            logMargPrSeq = joint log probability of the observed sequence
                        log p( x[1], x[2], ... x[T] )  
  '''
  PiInit, PiMat, K = _parseInput_TransParams(PiInit, PiMat)
  logSoftEv = _parseInput_SoftEv(logSoftEv, K)
  T = logSoftEv.shape[0]

  SoftEv, lognormC = expLogLik(logSoftEv)
  
  fmsg, margPrObs = FwdAlg(PiInit, PiMat, SoftEv)
  bmsg = BwdAlg(PiInit, PiMat, SoftEv, margPrObs)

  respPair = np.zeros((T,K,K))
  for t in xrange(1, T):
    respPair[t] = np.outer(fmsg[t-1], bmsg[t] * SoftEv[t])
    respPair[t] *= PiMat / margPrObs[t]
    #assert np.allclose(respPair[t].sum(), 1.0)
  logMargPrSeq = np.log(margPrObs).sum() + lognormC.sum()
  resp = fmsg * bmsg
  return resp, respPair, logMargPrSeq

########################################################### FwdAlg, BwdAlg Wrappers
###########################################################
def FwdAlg(PiInit, PiMat, SoftEv):
  ''' Forward algorithm for a single HMM sequence. Wrapper for FwdAlg_py/FwdAlg_cpp.

     Related
     -------
     FwdAlg_py

     Returns
     -------
        fmsg : 2D array, size T x K
                  fmsg[t,k] = p( z[t,k] = 1 | x[1] ... x[t] )
        margPrObs : 1D array, size T
                  margPrObs[t] = p( x[t] | x[1], x[2], ... x[t-1] )
  '''
  if PlatformConfig['FwdBwdImpl'] == "cpp":
    return FwdAlg_cpp(PiInit, PiMat, SoftEv)
  else:
    return FwdAlg_py(PiInit, PiMat, SoftEv)

def BwdAlg(PiInit, PiMat, SoftEv, margPrObs):
  ''' Backward algorithm for a single HMM sequence. Wrapper for BwdAlg_py/BwdAlg_cpp.

     Related
     -------
     BwdAlg_py

     Returns
     -------
     bmsg : 2D array, size TxK
              bmsg[t,k] = p( x[t+1], x[t+2], ... x[T] |  z[t,k] = 1 )
                          -------------------------------------
                          p( x[t+1], x[t+2], ... x[T] |  x[1] ... x[t])
  '''
  if PlatformConfig['FwdBwdImpl'] == "cpp":
    return BwdAlg_cpp(PiInit, PiMat, SoftEv, margPrObs)
  else:
    return BwdAlg_py(PiInit, PiMat, SoftEv, margPrObs)

########################################################### Python implementations
###########################################################

def FwdAlg_py(PiInit, PiMat, SoftEv):
  ''' Forward algorithm for a single HMM sequence. In pure python.

      Execute forward message-passing on an observed sequence
       given HMM state transition params and likelihoods of each observation

     Args
     -------
     piInit : 1D array, size K
            initial transition distribution to each of the K states
            must be valid probability vector (positive entries, sums to one)
     piMat  : 2D array, size KxK
            piMat[j] is transition distribution from state j to all K states.
            each row must be probability vector (positive entries, sums to one)
     SoftEv : 2D array, size TxK
            SoftEv[t] := p( x[t] | z[tk] = 1)
                         likelihood of observation t under state k
                         given up to an additive constant for each t
     Returns
     -------
        fmsg : 2D array, size T x K
                  fmsg[t,k] = p( z[t,k] = 1 | x[1] ... x[t] )
        margPrObs : 1D array, size T
                  margPrObs[t] = p( x[t] | x[1], x[2], ... x[t-1] )
  '''
  T = SoftEv.shape[0]
  K = PiInit.size
  PiTMat = PiMat.T

  fmsg = np.empty( (T,K) )
  margPrObs = np.zeros( T )
  for t in xrange( 0, T ):
    if t == 0:
      fmsg[t] = PiInit * SoftEv[0]
    else:
      fmsg[t] = np.dot(PiTMat, fmsg[t-1]) * SoftEv[t]
    margPrObs[t] = np.sum( fmsg[t] )
    fmsg[t] /= margPrObs[t]
  return fmsg, margPrObs
  

def BwdAlg_py(PiInit, PiMat, SoftEv, margPrObs):
  '''Backward algorithm for a single HMM sequence. In pure python.

     Takes as input the HMM state transition params, initial probabilities,
           and likelihoods of each observation
     Requires running forward filtering first, to obtain correct scaling.

     Args
     -------
     piInit : 1D array, size K
            initial transition distribution to each of the K states
            must be valid probability vector (positive entries, sums to one)
     piMat  : 2D array, size KxK
            piMat[j] is transition distribution from state j to all K states.
            each row must be probability vector (positive entries, sums to one)
     SoftEv : 2D array, size TxK
            SoftEv[t] := p( x[t] | z[tk] = 1)
                         likelihood of observation t under state k
                         given up to an additive constant for each t
     margPrObs : 1D array, size T
            margPrObs[t] := p( x[t] | x[1], x[2], ... x[t-1] )
            this is returned by FwdAlg

     Returns
     -------
     bmsg : 2D array, size TxK
              bmsg[t,k] = p( x[t+1], x[t+2], ... x[T] |  z[t,k] = 1 )
                          -------------------------------------
                          p( x[t+1], x[t+2], ... x[T] |  x[1] ... x[t])
  '''
  T = SoftEv.shape[0]
  K = PiInit.size
  bmsg = np.ones( (T,K) )
  for t in xrange( T-2, -1, -1 ):
    bmsg[t] = np.dot(PiMat, bmsg[t+1] * SoftEv[t+1] )
    bmsg[t] /= margPrObs[t+1]
  return bmsg


########################################################### expLogLik
###########################################################
def expLogLik(logSoftEv, axis=1):
  ''' Return element-wise exp of input log likelihood
        guaranteed not to underflow
    
      Returns
      --------
      SoftEv : 2D array, size TxK
                equal to exp(logSoftEv), up to prop constant for each row
      lognormC : 1D array, size T
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
  assert PiMat.ndim == 2
  J, K = PiMat.shape
  assert J == K
  assert K0 == K
  return PiInit, PiMat, K

def _parseInput_SoftEv(logSoftEv, K):
  logSoftEv = np.asarray(logSoftEv, dtype=np.float64)
  Tl, Kl = logSoftEv.shape
  assert Kl == K
  return logSoftEv


def viterbi(logSoftEv, pi0, pi):
  '''
  Input : The log evidence matrix (logSoftEv[n,k] = log(p(x_n | z_n = k))), as 
  well as the starting distribution and transition matrix.

  '''
  logPi0 = np.log(pi0 + EPS)
  logPi = np.log(pi + EPS)
  T, K = np.shape(logSoftEv)
 
  V = np.zeros((T, K))
  prev = np.zeros((T, K))


  for t in xrange(T):
    biggest = -np.inf
    for l in xrange(K):
      
      if t == 0:
        V[0, l] = logSoftEv[t,l] + logPi0[l]
        prev[0,l] = -1
        continue
      for k in xrange(K):
        logpr = logPi[k,l] + V[t-1, k]
        if logpr > biggest:
          biggest = logpr
          prev[t,l] = k

      V[t, l] = logSoftEv[t,l] + biggest

  #Find most likely sequence of z's
  z = np.zeros(T)
  for t in reversed(xrange(T)):
    if t == T-1:
      z[t] = np.argmax(V[t,:])
    else:
      z[t] = prev[t+1, z[t+1]]

  return z


def calcEntropyFromResp(resp, respPair, Data, eps=1e-100):
  ''' Calculate entropy E_q(z) [ log q(z) ] for all sequences
  '''
  startLocIDs = Data.doc_range[:-1]

  sigma = respPair / (respPair.sum(axis=2)[:,:,np.newaxis] + eps)
  firstH = -1 * np.sum(resp[startLocIDs] * np.log(resp[startLocIDs]+eps))
  restH = -1 * np.sum(respPair[1:,:,:] * np.log(sigma[1:,:,:] + EPS))
  return firstH + restH


def calcEntropyFromResp_bySeq(resp, respPair, Data, eps=1e-100):
  ''' Calculate entropy E_q(z) [ log q(z) ] for all sequences, using loop over seqs

      This is simply a way to verify correctness for fast, vectorized calculations.
  '''
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
