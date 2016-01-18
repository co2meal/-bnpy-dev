'''
LibSparseResp.py
'''
import os
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes
import scipy.sparse

def sparsifyResp_cpp(Resp, nnzPerRow, order='C'):
    '''
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

def sparsifyLogResp_cpp(logResp, nnzPerRow, order='C'):
    '''
    '''
    if not hasEigenLibReady:
        raise ValueError("Cannot find library %s. Please recompile."
                         % (libfilename))
    if order != 'C':
        raise NotImplementedError("LibFwdBwd only supports row-major order.")
    N, K = logResp.shape
    if nnzPerRow == 1:
        # Fast case. No need for C++ code.
        spR_colids = np.argmax(logResp, axis=1)
        spR_data = np.ones(N, dtype=np.float64)
    else:
        # Prep input to C++ routine. Verify correct byte-order (row-major).
        logResp = np.asarray(logResp, order=order)
        # Allocate output arrays, initialized to all zeros
        spR_data = np.zeros(N * nnzPerRow, dtype=np.float64, order=order)
        spR_colids = np.zeros(N * nnzPerRow, dtype=np.int32, order=order)
        # Execute C++ code (fills in outputs in-place)
        lib.sparsifyLogResp(logResp, nnzPerRow, N, K, spR_data, spR_colids)

    # Here, both spR_data and spR_colids have been created
    # Assemble these into a row-based sparse matrix (scipy object)
    spR_indptr = np.arange(0, N * nnzPerRow + nnzPerRow, 
                           step=nnzPerRow, dtype=spR_colids.dtype)
    spR = scipy.sparse.csr_matrix(
        (spR_data, spR_colids, spR_indptr),
        shape=(N,K),
        )
    return spR


def calcRlogR_withSparseRespCSR_cpp(
        spR_csr=None, nnzPerRow=-1, order='C', **kwargs):
    '''
    '''
    if not hasEigenLibReady:
        raise ValueError("Cannot find library %s. Please recompile."
                         % (libfilename))
    if order != 'C':
        raise NotImplementedError("LibFwdBwd only supports row-major order.")

    assert spR_csr is not None
    N, K = spR_csr.shape
    if nnzPerRow == 1:
        # Fast case. No need for C++ code.
        return 0.0
    elif nnzPerRow > 1 and nnzPerRow <= K:
        # Preallocate memory
        Hvec_OUT = np.zeros(K, dtype=np.float64)
        # Execute C++ code (fills in output array Hvec_OUT in-place)
        lib.calcRlogR_withSparseRespCSR(
            spR_csr.data,
            spR_csr.indices,
            spR_csr.indptr,
            K,
            N,
            nnzPerRow,
            Hvec_OUT)
        return Hvec_OUT
    else:
        raise ValueError("Bad nnzPerRow value %d. Need >= 1" % (nnzPerRow))

def calcRlogRdotv_withSparseRespCSR_cpp(
        spR_csr=None, v=None, nnzPerRow=-1, order='C', **kwargs):
    '''
    '''
    if not hasEigenLibReady:
        raise ValueError("Cannot find library %s. Please recompile."
                         % (libfilename))
    if order != 'C':
        raise NotImplementedError("LibFwdBwd only supports row-major order.")
    v = np.asarray(v, order=order)
    assert spR_csr is not None
    N, K = spR_csr.shape
    if nnzPerRow == 1:
        # Fast case. No need for C++ code.
        return 0.0
    elif nnzPerRow > 1 and nnzPerRow <= K:
        # Preallocate memory
        Hvec_OUT = np.zeros(K, dtype=np.float64)
        # Execute C++ code (fills in output array Hvec_OUT in-place)
        lib.calcRlogRdotv_withSparseRespCSR(
            spR_csr.data,
            spR_csr.indices,
            spR_csr.indptr,
            v,
            K,
            N,
            nnzPerRow,
            Hvec_OUT)
        return Hvec_OUT
    else:
        raise ValueError("Bad nnzPerRow value %d. Need >= 1" % (nnzPerRow))

def calcMergeRlogR_withSparseRespCSR_cpp(
        spR_csr=None, nnzPerRow=-1, order='C', mPairIDs=None, **kwargs):
    '''
    '''
    if not hasEigenLibReady:
        raise ValueError("Cannot find library %s. Please recompile."
                         % (libfilename))
    assert spR_csr is not None
    N, K = spR_csr.shape
    if nnzPerRow == 1:
        # Fast case. No need for C++ code.
        return None
    elif nnzPerRow > 1 and nnzPerRow <= K:
        # Preallocate memory
        m_Hvec_OUT = np.zeros(len(mPairIDs), dtype=np.float64)
        for mID, (kA, kB) in enumerate(mPairIDs): 
            # Execute C++ code (fills in output array Hvec_OUT in-place)
            m_Hvec_OUT[mID] = lib.calcMergeRlogR_withSparseRespCSR(
                spR_csr.data,
                spR_csr.indices,
                spR_csr.indptr,
                K,
                N,
                nnzPerRow,
                kA, kB)
        return m_Hvec_OUT
    else:
        raise ValueError("Bad nnzPerRow value %d. Need >= 1" % (nnzPerRow))

def calcMergeRlogRdotv_withSparseRespCSR_cpp(
        spR_csr=None, nnzPerRow=-1, v=None, 
        order='C', mPairIDs=None, **kwargs):
    '''
    '''
    if not hasEigenLibReady:
        raise ValueError("Cannot find library %s. Please recompile."
                         % (libfilename))
    assert spR_csr is not None
    N, K = spR_csr.shape
    if nnzPerRow == 1:
        # Fast case. No need for C++ code.
        return None
    elif nnzPerRow > 1 and nnzPerRow <= K:
        # Preallocate memory
        m_Hvec_OUT = np.zeros(len(mPairIDs), dtype=np.float64)
        for mID, (kA, kB) in enumerate(mPairIDs): 
            # Execute C++ code (fills in output array Hvec_OUT in-place)
            m_Hvec_OUT[mID] = lib.calcMergeRlogRdotv_withSparseRespCSR(
                spR_csr.data,
                spR_csr.indices,
                spR_csr.indptr,
                v,
                K,
                N,
                nnzPerRow,
                kA, kB)
        return m_Hvec_OUT
    else:
        raise ValueError("Bad nnzPerRow value %d. Need >= 1" % (nnzPerRow))


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


def calcSparseLocalParams_SingleDoc(
        wc_d, Lik_d, alphaEbeta, alphaEbetaRem=None,
        topicCount_d_OUT=None, 
        spResp_data_OUT=None,
        spResp_colids_OUT=None,
        nCoordAscentItersLP=10, convThrLP=0.001,
        nnzPerRowLP=2,
        restartLP=0,
        initDocTopicCountLP='setDocProbsToEGlobalProbs',
        **kwargs):
    N, K = Lik_d.shape
    K1 = alphaEbeta.size
    assert K == K1
    assert topicCount_d_OUT.size == K
    assert spResp_data_OUT.size == N * nnzPerRowLP
    assert spResp_colids_OUT.size == N * nnzPerRowLP
    nnzPerRowLP = np.minimum(nnzPerRowLP, K)
    if initDocTopicCountLP.count("setDocProbsToEGlobalProbs"):
        initProbsToEbeta = 1
    else:
        initProbsToEbeta = 0
    if isinstance(wc_d, np.ndarray) and wc_d.size == N:
        libTopics.sparseLocalStepSingleDocWithWordCounts(
            wc_d, Lik_d, alphaEbeta,
            nnzPerRowLP, N, K, nCoordAscentItersLP, convThrLP,
            initProbsToEbeta,
            topicCount_d_OUT,
            spResp_data_OUT,
            spResp_colids_OUT,
            )
    else:
        libTopics.sparseLocalStepSingleDoc(
            Lik_d, alphaEbeta,
            nnzPerRowLP, N, K, nCoordAscentItersLP, convThrLP,
            initProbsToEbeta,
            topicCount_d_OUT,
            spResp_data_OUT,
            spResp_colids_OUT,
            )


''' This block of code loads the shared library and defines wrapper functions
    that can take numpy array objects.
'''
libpath = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-1])
libfilename = 'libsparseresp.so'
libfilename2 = 'libsparsetopics.so'
hasEigenLibReady = True

try:
    # Load the compiled C++ library from disk
    lib = ctypes.cdll.LoadLibrary(os.path.join(libpath, libfilename))

    # Now specify each function's signature
    lib.sparsifyResp.restype = None
    lib.sparsifyResp.argtypes = \
        [ndpointer(ctypes.c_double),
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_int),
         ]

    lib.sparsifyLogResp.restype = None
    lib.sparsifyLogResp.argtypes = \
        [ndpointer(ctypes.c_double),
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_int),
         ]

    lib.calcRlogR_withSparseRespCSR.restype = None
    lib.calcRlogR_withSparseRespCSR.argtypes = \
        [ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_int),
         ndpointer(ctypes.c_int),
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ndpointer(ctypes.c_double),
         ]

    lib.calcRlogRdotv_withSparseRespCSR.restype = None
    lib.calcRlogRdotv_withSparseRespCSR.argtypes = \
        [ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_int),
         ndpointer(ctypes.c_int),
         ndpointer(ctypes.c_double),
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ndpointer(ctypes.c_double),
         ]

    lib.calcMergeRlogR_withSparseRespCSR.restype = ctypes.c_double
    lib.calcMergeRlogR_withSparseRespCSR.argtypes = \
        [ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_int),
         ndpointer(ctypes.c_int),
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ]

    lib.calcMergeRlogRdotv_withSparseRespCSR.restype = ctypes.c_double
    lib.calcMergeRlogRdotv_withSparseRespCSR.argtypes = \
        [ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_int),
         ndpointer(ctypes.c_int),
         ndpointer(ctypes.c_double),
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
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

    libTopics = ctypes.cdll.LoadLibrary(os.path.join(libpath, libfilename2))
    libTopics.sparseLocalStepSingleDoc.restype = None
    libTopics.sparseLocalStepSingleDoc.argtypes = \
        [ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_double,
         ctypes.c_int,
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_int),
         ]

    libTopics.sparseLocalStepSingleDocWithWordCounts.restype = None
    libTopics.sparseLocalStepSingleDocWithWordCounts.argtypes = \
        [ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_int,
         ctypes.c_double,
         ctypes.c_int,
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_int),
         ]
except OSError as e:
    print str(e)
    # No compiled C++ library exists
    hasEigenLibReady = False


if __name__ == "__main__":
    from scipy.special import digamma
    N = 3
    K = 7
    nnzPerRow = 2
    MAXITER = 50
    convThr = 0.005
    alphaEbeta = np.random.rand(K)
    logLik_d = np.log(np.random.rand(N,K) **2)
    wc_d = np.float64(np.arange(1, N+1))
    D = 10
    topicCount_d = np.zeros(K)
    spResp_data = np.zeros(N * D * nnzPerRow)
    spResp_colids = np.zeros(N * D * nnzPerRow, dtype=np.int32)
    for d in [0, 1, 2, 3]:
        print nnzPerRow
        start = d * (N * nnzPerRow)
        stop = (d+1) * (N * nnzPerRow)
        libTopics.sparseLocalStepSingleDocWithWordCounts(
            wc_d, logLik_d,
            alphaEbeta,
            nnzPerRow,
            N,
            K,
            MAXITER,
            convThr,
            topicCount_d,
            spResp_data[start:stop],
            spResp_colids[start:stop],
            )
        print ' '.join(['%5.2f' % (x) for x in topicCount_d])
        print 'sum(topicCount_d)=', topicCount_d.sum()
        print 'sum(wc_d)=', np.sum(wc_d)
    '''
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
    '''