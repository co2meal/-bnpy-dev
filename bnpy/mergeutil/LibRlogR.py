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
from numpy.ctypeslib import ndpointer
import ctypes

lib = ctypes.cdll.LoadLibrary('librlogr.so')
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

########################################################### all-pairs
###########################################################  with vector

def calcRlogR_allpairsdotv_c(R, v):
  R = np.asarray(R, order='F')
  v = np.asarray(v, order='F')
  N,K = R.shape
  Z = np.zeros((K,K), order='F' )
  lib.CalcRlogR_AllPairsDotV( R, v, Z, N, K)
  return Z

def calcRlogR_allpairsdotv_numpy(R, v):
  K = R.shape[1]
  Z = np.zeros((K,K))
  for jj in range(K):
    curR = R[:,jj][:,np.newaxis] + R[:, jj+1:]
    curR *= np.log(curR)
    Z[jj,jj+1:] = np.dot(v,curR)
  return Z

########################################################### all-pairs
###########################################################

def calcRlogR_allpairs_c(R):
  R = np.asarray(R, order='F')
  N,K = R.shape
  Z = np.zeros((K,K), order='F' )
  lib.CalcRlogR_AllPairs( R, Z, N, K)
  return Z

def calcRlogR_allpairs_numpy(R):
  K = R.shape[1]
  Z = np.zeros((K,K))
  for jj in range(K):
    curR = R[:,jj][:,np.newaxis] + R[:, jj+1:]
    curR *= np.log(curR)
    Z[jj,jj+1:] = np.sum(curR, axis=0)
  return Z

"""
def CalcRlogR_Vectorized(R):
  R = np.asarray(R, order='F')
  N,K = R.shape
  Z = np.zeros((K,K), order='F')
  lib.CalcRlogR_Vectorized(R, Z, N, K);
  return Z
lib.CalcRlogR_Vectorized.restype = None
lib.CalcRlogR_Vectorized.argtypes = \
               [ndpointer(ctypes.c_double),
                ndpointer(ctypes.c_double),
                ctypes.c_int, ctypes.c_int]
"""