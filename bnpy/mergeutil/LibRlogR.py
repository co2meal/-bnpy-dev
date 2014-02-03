'''
LibRlogR.py

Library of routines for computing assignment entropies for merges

Notes
-------
Eigen expects the matrices to be fortran formated (column-major ordering).
In contrast, Numpy defaults to C-format (row-major ordering)
All functions here take care of this under the hood (so end-user doesn't need to worry about alignment)
This explains the mysterious line: X=np.asarray(X, order='F')
However, we do *return* values that are F-ordered by default. 
'''
import os
import numpy as np
import numexpr as ne
from numpy.ctypeslib import ndpointer
import ctypes
if 'OMP_NUM_THREADS' in os.environ:
  ne.set_num_threads(os.environ['OMP_NUM_THREADS'])

libpath = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-1])

doUseLib = True
try:
  lib = ctypes.cdll.LoadLibrary(os.path.join(libpath,'librlogr.so'))
  lib.CalcRlogR_AllPairs.restype = None
  lib.CalcRlogR_AllPairs.argtypes = \
               [ndpointer(ctypes.c_double),
                ndpointer(ctypes.c_double),
                ctypes.c_int, ctypes.c_int]

  lib.CalcRlogR_AllPairsDotV.restype = None
  lib.CalcRlogR_AllPairsDotV.argtypes = \
               [ndpointer(ctypes.c_double),
                ndpointer(ctypes.c_double), ndpointer(ctypes.c_double),
                ctypes.c_int, ctypes.c_int]

  lib.CalcRlogR_SpecificPairsDotV.restype = None
  lib.CalcRlogR_SpecificPairsDotV.argtypes = \
               [ndpointer(ctypes.c_double),
                ndpointer(ctypes.c_double), ndpointer(ctypes.c_double),
                ndpointer(ctypes.c_double), ndpointer(ctypes.c_double),
                ctypes.c_int, ctypes.c_int, ctypes.c_int]

except OSError:
  # No compiled C++ library exists
  doUseLib = False  



########################################################### safeExpAndNormalizeRows
###########################################################
def safeExpAndNormalizeRows_numpy(R):
  # Take exp of wv in numerically stable manner (first subtract the max)
  #  in-place so no new allocations occur
  R -= np.max(R, axis=1)[:,np.newaxis]
  np.exp(R, out=R)
  # Normalize, so rows of wv sum to one
  R /= R.sum(axis=1)[:,np.newaxis]

def safeExpAndNormalizeRows_numexpr(R):
  # Take exp of wv in numerically stable manner (first subtract the max)
  #  in-place so no new allocations occur
  R -= np.max(R, axis=1)[:,np.newaxis]
  ne.evaluate("exp(R)", out=R)
  # Normalize, so rows of wv sum to one
  R /= R.sum(axis=1)[:,np.newaxis]

########################################################### standard R * log(R)
###########################################################
def calcRlogR_numpy(R):
  return np.sum(R * np.log(R), axis=0)

def calcRlogR_numexpr(R):
  return ne.evaluate("sum(R*log(R), axis=0)")


########################################################### standard R * log(R)
###########################################################
def calcRlogRdotv_numpy(R, v):
  return np.dot( v, R * np.log(R))

def calcRlogRdotv_numexpr(R, v):
  RlogR = ne.evaluate("R*log(R)")
  return np.dot(v, RlogR)

########################################################### all-pairs
###########################################################
def calcRlogR_allpairs_c(R):
  if not doUseLib:
    return calcRlogR_allpairs_numpy(R, v, mPairs)
  R = np.asarray(R, order='F')
  N,K = R.shape
  Z = np.zeros((K,K), order='F' )
  lib.CalcRlogR_AllPairs( R, Z, N, K)
  return Z

def calcRlogR_allpairs_numpy(R):
  K = R.shape[1]
  Z = np.zeros((K,K))
  for jj in xrange(K-1):
    curR = R[:,jj][:,np.newaxis] + R[:, jj+1:]
    curR *= np.log(curR)
    Z[jj,jj+1:] = np.sum(curR, axis=0)
  return Z

def calcRlogR_allpairs_numexpr(R):
  K = R.shape[1]
  Z = np.zeros((K,K))
  for jj in xrange(K-1):
    curR = R[:,jj][:,np.newaxis] + R[:, jj+1:]
    curZ = ne.evaluate("sum(curR * log(curR), axis=0)")
    Z[jj,jj+1:] = curZ
  return Z


########################################################### all-pairs
###########################################################  with vector
def calcRlogRdotv_allpairs_c(R, v):
  if not doUseLib:
    return calcRlogRdotv_allpairs_numexpr(R,v)
  R = np.asarray(R, order='F')
  v = np.asarray(v, order='F')
  N,K = R.shape
  Z = np.zeros((K,K), order='F' )
  lib.CalcRlogR_AllPairsDotV( R, v, Z, N, K)
  return Z

def calcRlogRdotv_allpairs_numpy(R, v):
  K = R.shape[1]
  Z = np.zeros((K,K))
  for jj in range(K):
    curR = R[:,jj][:,np.newaxis] + R[:, jj+1:]
    curR *= np.log(curR)
    Z[jj,jj+1:] = np.dot(v,curR)
  return Z

def calcRlogRdotv_allpairs_numexpr(R, v):
  K = R.shape[1]
  Z = np.zeros((K,K))
  for jj in xrange(K-1):
    curR = R[:,jj][:,np.newaxis] + R[:, jj+1:]
    ne.evaluate("curR * log(curR)", out=curR)
    curZ = np.dot(v, curR)
    Z[jj,jj+1:] = curZ
  return Z


########################################################### specific-pairs
###########################################################  with vector

def calcRlogRdotv_specificpairs_numpy(R, v, mPairs):
  K = R.shape[1]
  ElogqZMat = np.zeros((K, K))
  if K == 1:
    return Z
  for (kA, kB) in mPairs:
    curWV = R[:,kA] + R[:, kB]
    curWV *= np.log(curWV)
    ElogqZMat[kA,kB] = np.dot(v, curWV)
  return ElogqZMat

def calcRlogRdotv_specificpairs_numexpr(R, v, mPairs):
  K = R.shape[1]
  ElogqZMat = np.zeros((K, K))
  if K == 1:
    return Z
  for (kA, kB) in mPairs:
    curR = R[:,kA] + R[:, kB]
    ne.evaluate("curR * log(curR)", out=curR)
    ElogqZMat[kA,kB] = np.dot(v, curR)
  return ElogqZMat

def calcRlogRdotv_specificpairs_c(R, v, mPairs):
  if not doUseLib:
    return calcRlogR_specificpairsdotv_numpy(R, v, mPairs)
  R = np.asarray(R, order='F')
  v = np.asarray(v, order='F')
  N,K = R.shape
  Z = np.zeros((K,K), order='F' )
  if K == 1 or len(mPairs) == 0:
    return Z
  aList, bList = zip(*mPairs)
  avec = np.asarray(aList, order='F', dtype=np.float64)
  bvec = np.asarray(bList, order='F', dtype=np.float64)

  lib.CalcRlogR_SpecificPairsDotV( R, v, avec, bvec, Z, N, len(avec), K)
  return Z

