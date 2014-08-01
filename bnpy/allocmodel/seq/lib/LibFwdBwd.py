import os
import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer

def FwdAlg_cpp(initPi, transPi, SoftEv, order='C'):
  ''' Forward algorithm for a single HMM sequence. Implemented in C++/Eigen.
  '''
  if not doUseLib:
    raise ValueError("Cannot find library %s. Please recompile." \
                      % (libfilename))
  if order != 'C':
    raise NotImplementedError("LibFwdBwd only supports row-major order.")
  T, K = SoftEv.shape
  ## Prep inputs
  initPi = np.asarray(initPi, order=order)
  transPi = np.asarray(transPi, order=order)
  SoftEv = np.asarray(SoftEv, order=order)

  ## Allocate outputs
  fwdMsg = np.zeros((T,K), order=order)
  margPrObs = np.zeros(T, order=order)

  ## Execute C++ code (fills in outputs in-place)
  lib.FwdAlg(initPi, transPi, SoftEv, fwdMsg, margPrObs, K, T)
  return fwdMsg, margPrObs

def BwdAlg_cpp(initPi, transPi, SoftEv, margPrObs, order='C'):
  ''' Backward algorithm for a single HMM sequence. Implemented in C++/Eigen.
  '''
  if not doUseLib:
    raise ValueError("Cannot find library %s. Please recompile." \
                      % (libfilename))
  if order != 'C':
    raise NotImplementedError("LibFwdBwd only supports row-major order.")

  ## Prep inputs
  T, K = SoftEv.shape
  initPi = np.asarray(initPi, order=order)
  transPi = np.asarray(transPi, order=order)
  SoftEv = np.asarray(SoftEv, order=order)
  margPrObs = np.asarray(margPrObs, order=order)

  ## Allocate outputs
  bMsg = np.zeros((T,K), order=order)

  ## Execute C++ code for backward pass (fills in bMsg in-place)
  lib.BwdAlg(initPi, transPi, SoftEv, margPrObs, bMsg, K, T)
  return bMsg


########################################################### C++ interface code
###########################################################
''' This block of code loads the shared library and defines wrapper functions
    that can take numpy array objects.
'''
libpath = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-1])
libfilename = 'libfwdbwd.so'
doUseLib = True

try:
  lib = ctypes.cdll.LoadLibrary(os.path.join(libpath, libfilename))
  lib.FwdAlg.restype = None
  lib.FwdAlg.argtypes = \
               [ndpointer(ctypes.c_double),
                ndpointer(ctypes.c_double),
                ndpointer(ctypes.c_double),
                ndpointer(ctypes.c_double),
                ndpointer(ctypes.c_double),
                ctypes.c_int, ctypes.c_int]

  lib.BwdAlg.restype = None
  lib.BwdAlg.argtypes = \
               [ndpointer(ctypes.c_double),
                ndpointer(ctypes.c_double),
                ndpointer(ctypes.c_double),
                ndpointer(ctypes.c_double),
                ndpointer(ctypes.c_double),
                ctypes.c_int, ctypes.c_int]

except OSError:
  # No compiled C++ library exists
  doUseLib = False  
