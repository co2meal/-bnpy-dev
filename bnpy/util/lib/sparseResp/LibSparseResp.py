'''
LibSparseResp.py
'''
import os
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes
import scipy.sparse

def sparsifyResp_cpp(Resp, nnzPerRow, order='C'):
    ''' Forward algorithm for a single HMM sequence. Implemented in C++/Eigen.
    '''
    if not hasEigenLibReady:
        raise ValueError("Cannot find library %s. Please recompile."
                         % (libfilename))
    if order != 'C':
        raise NotImplementedError("LibFwdBwd only supports row-major order.")
    N, K = Resp.shape

    if nnzPerRow == 1:
        # Fast case. No need for C++ code.
        spR_colids = np.argmax(Resp, axis=1)
        spR_data = np.ones(N, dtype=np.float64)
    else:
        # Prep input to C++ routine. Verify correct byte-order (row-major).
        Resp = np.asarray(Resp, order=order)
        # Allocate output arrays, initialized to all zeros
        spR_data = np.zeros(N * nnzPerRow, dtype=np.float64, order=order)
        spR_colids = np.zeros(N * nnzPerRow, dtype=np.int32, order=order)
        # Execute C++ code (fills in outputs in-place)
        lib.sparsifyResp(Resp, nnzPerRow, N, K, spR_data, spR_colids)

    # Here, both spR_data and spR_colids have been created
    # Assemble these into a row-based sparse matrix (scipy object)
    spR_indptr = np.arange(0, N * nnzPerRow + nnzPerRow, 
                           step=nnzPerRow, dtype=spR_colids.dtype)
    spR = scipy.sparse.csr_matrix(
        (spR_data, spR_colids, spR_indptr),
        shape=(N,K),
        )
    return spR


def calcRXXT_withSparseRespCSR_cpp(
        X=None, spR_csr=None, order='C', **kwargs):
    if not hasEigenLibReady:
        raise ValueError("Cannot find library %s. Please recompile."
                         % (libfilename))
    if order != 'C':
        raise NotImplementedError("LibFwdBwd only supports row-major order.")
    N, K = spR_csr.shape
    N1, D = X.shape
    assert N == N1
    nnzPerRow = spR_csr.data.size // N
    X = np.asarray(X, order=order)
    stat_RXX = np.zeros((K, D, D), order=order)
    lib.calcRXXT_withSparseRespCSR(
        X, spR_csr.data, spR_csr.indices, spR_csr.indptr,
        D, K, N, nnzPerRow,
        stat_RXX)
    return stat_RXX


def calcRXX_withSparseRespCSC_cpp(
        X=None, spR_csc=None, order='C', **kwargs):
    if not hasEigenLibReady:
        raise ValueError("Cannot find library %s. Please recompile."
                         % (libfilename))
    if order != 'C':
        raise NotImplementedError("LibFwdBwd only supports row-major order.")
    N, K = spR_csc.shape
    N1, D = X.shape
    assert N == N1
    L = spR_csc.data.size

    X = np.asarray(X, order=order)
    
    stat_RXX = np.zeros((K, D), order=order)

    lib.calcRXX_withSparseRespCSC(
        X, spR_csc.data, spR_csc.indices, spR_csc.indptr,
        D, K, L, N,
        stat_RXX)
    return stat_RXX

def calcRXX_withSparseRespCSR_cpp(
        X=None, spR_csr=None, order='C', **kwargs):
    if not hasEigenLibReady:
        raise ValueError("Cannot find library %s. Please recompile."
                         % (libfilename))
    if order != 'C':
        raise NotImplementedError("LibFwdBwd only supports row-major order.")
    N, K = spR_csr.shape
    N1, D = X.shape
    assert N == N1
    nnzPerRow = spR_csr.data.size // N

    X = np.asarray(X, order=order)
    stat_RXX = np.zeros((K, D), order=order)

    lib.calcRXX_withSparseRespCSR(
        X, spR_csr.data, spR_csr.indices, spR_csr.indptr,
        D, K, N, nnzPerRow,
        stat_RXX)
    return stat_RXX



''' This block of code loads the shared library and defines wrapper functions
    that can take numpy array objects.
'''
libpath = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-1])
libfilename = 'libsparseresp.so'
hasEigenLibReady = True

try:
    lib = ctypes.cdll.LoadLibrary(os.path.join(libpath, libfilename))
    lib.sparsifyResp.restype = None
    lib.sparsifyResp.argtypes = \
        [ndpointer(ctypes.c_double),
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_int),
         ]

    lib.calcRXXT_withSparseRespCSR.restype = None
    lib.calcRXXT_withSparseRespCSR.argtypes = \
        [ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_int),
         ndpointer(ctypes.c_int),
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ndpointer(ctypes.c_double),
         ]

    lib.calcRXX_withSparseRespCSR.restype = None
    lib.calcRXX_withSparseRespCSR.argtypes = \
        [ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_int),
         ndpointer(ctypes.c_int),
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ndpointer(ctypes.c_double),
         ]

    lib.calcRXX_withSparseRespCSC.restype = None
    lib.calcRXX_withSparseRespCSC.argtypes = \
        [ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_int),
         ndpointer(ctypes.c_int),
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ndpointer(ctypes.c_double),
         ]
except OSError:
    # No compiled C++ library exists
    hasEigenLibReady = False


if __name__ == "__main__":
    from bnpy.util.SparseRespUtil import sparsifyResp_numpy_vectorized

    for nnzPerRow in [1, 2, 3]:
        R = np.random.rand(5,6)
        R /= R.sum(axis=1)[:,np.newaxis]
        print R

        spR = sparsifyResp_cpp(R, nnzPerRow).toarray()
        print spR

        spR2 = sparsifyResp_numpy_vectorized(R, nnzPerRow).toarray()
        print spR2

        assert np.allclose(spR, spR2)