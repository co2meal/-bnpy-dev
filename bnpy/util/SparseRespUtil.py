import argparse
import numpy as np
import scipy.sparse
import timeit
import time
import sys

from SparseRespUtilX import calcSpRData_cython
from lib.sparseResp.LibSparseResp import sparsifyResp_cpp

def sparsifyResp(resp, nnzPerRow=1, sparse_format='csc'):
	spR_csr = sparsifyResp_cpp(resp, nnzPerRow)
	if sparse_format == 'csc':
		return spR_csr.tocsc()
	return spR_csr

def sparsifyResp_numpy_forloop(resp, nnzPerRow=1):
	'''
	Returns
	-------
	spR : sparse csr matrix, shape N x K
	'''
	N, K = resp.shape

	if nnzPerRow == 1:
		spR_colids = np.argmax(resp, axis=1)
		spR_data = np.ones(N, dtype=resp.dtype)
	else:
		spR_data = np.zeros(N * nnzPerRow)
		spR_colids = np.zeros(N * nnzPerRow, dtype=np.int32)
		for n in xrange(N):
			start = n * nnzPerRow
			stop = start + nnzPerRow
			top_colids_n = np.argpartition(resp[n], -nnzPerRow)[-nnzPerRow:]
			spR_colids[start:stop] = top_colids_n

			top_rowsum = resp[n, top_colids_n].sum()
			spR_data[start:stop] = resp[n, top_colids_n] / top_rowsum
	# Assemble into common sparse matrix
	spR_indptr = np.arange(0, N * nnzPerRow + nnzPerRow, 
						   step=nnzPerRow, dtype=spR_colids.dtype)
	spR = scipy.sparse.csr_matrix(
		(spR_data, spR_colids, spR_indptr),
		shape=(N,K),
		)
	return spR

def sparsifyResp_numpy_vectorized(resp, nnzPerRow=1):
	'''
	Returns
	-------
	spR : sparse csr matrix, shape N x K
	'''
	N, K = resp.shape

	if nnzPerRow == 1:
		spR_colids = np.argmax(resp, axis=1)
		spR_data = np.ones(N, dtype=resp.dtype)
	else:
		spR_data = np.zeros(N * nnzPerRow)
		top_colids = np.argpartition(resp, K-nnzPerRow, axis=1)
		top_colids = top_colids[:, -nnzPerRow:]
		for n in xrange(N):
			start = n * nnzPerRow
			stop = start + nnzPerRow
			top_rowsum = resp[n, top_colids[n]].sum()
			spR_data[start:stop] = resp[n, top_colids[n]] / top_rowsum
		spR_colids = top_colids.flatten()

	# Assemble into common sparse matrix
	spR_indptr = np.arange(0, N * nnzPerRow + nnzPerRow, 
						   step=nnzPerRow, dtype=spR_colids.dtype)
	spR = scipy.sparse.csr_matrix(
		(spR_data, spR_colids, spR_indptr),
		shape=(N,K),
		)
	return spR


def sparsifyResp_numpy_with_cython(resp, nnzPerRow=1):
	'''
	Returns
	-------
	spR : sparse csr matrix, shape N x K
	'''
	N, K = resp.shape
	if nnzPerRow == 1:
		spR_data = np.ones(N, dtype=resp.dtype)
		spR_colids = np.argmax(resp, axis=1)
	else:
		top_colids = np.argpartition(resp, K-nnzPerRow, axis=1)
		top_colids = top_colids[:, -nnzPerRow:]
		spR_data = calcSpRData_cython(resp, top_colids, nnzPerRow)
		spR_colids = top_colids.flatten()
	# Assemble into common sparse matrix
	spR_indptr = np.arange(0, N * nnzPerRow + nnzPerRow, 
						   step=nnzPerRow, dtype=spR_colids.dtype)
	spR = scipy.sparse.csr_matrix(
		(spR_data, spR_colids, spR_indptr),
		shape=(N,K),
		)
	return spR

def default_funcList(prefix='sparsifyResp_'):
	funcList = []
	for key, val in globals().items():
		if key.startswith(prefix):
			funcList.append(val)
	return funcList

def test_correctness(R=None, N=3, K=10, 
					 funcList=default_funcList(),
					 nnzPerRow=None, nnzList=None):
	if R is None:
		R = np.random.rand(N, K)
	if nnzPerRow is None:
		nnzPerRow = 1

	if nnzList is None:
		nnzList = [nnzPerRow]

	for nnzPerRow in nnzList:
		nnzPerRow = np.minimum(nnzPerRow, R.shape[1])
		nnzPerRow = np.maximum(nnzPerRow, 1)

		print 'nnzPerRow=', nnzPerRow
		for i in range(len(funcList)):
			for j in range(i+1, len(funcList)):
				func_i = funcList[i]
				func_j = funcList[j]

				ans1 = func_i(R, nnzPerRow).toarray()
				ans2 = func_j(R, nnzPerRow).toarray()
				assert np.allclose(ans1, ans2)
				assert np.allclose(np.sum(ans1 > 1e-5, axis=1), nnzPerRow)
		print '  all pairs of funcs give same answer'


def test_speed(R=None, N=3, K=10,
			   funcList=default_funcList(),
			   nnzPerRow=None, nnzList=None, nRep=1):
	if R is None:
		R = np.random.rand(N, K)
	if funcList is None:
		funcList = [sparsifyResp_numpy_forloop, sparsifyResp_numpy_vectorized]
	if nnzPerRow is None:
		nnzPerRow = 1

	if nnzList is None:
		nnzList = [nnzPerRow]

	for nnzPerRow in nnzList:
		nnzPerRow = np.minimum(nnzPerRow, R.shape[1])
		nnzPerRow = np.maximum(nnzPerRow, 1)
		print 'nnzPerRow=', nnzPerRow
		for func in funcList:
			if func.__name__.count('forloop') and N * K > 1e6:
				print 'SKIPPED | ', func.__name__
				continue
			do_timing_test_for_func(func, (R,nnzPerRow), nRep=nRep)

def do_timing_test_for_func(func, args, nRep=1):
    times = list()
    for trial in xrange(nRep):
        tstart = time.time()
        func(*args)
        tstop = time.time()
        times.append(tstop - tstart)
    print " AVG %.4f sec  MEDIAN %.4f sec | %s" % (
        np.mean(times), np.median(times), func.__name__)

def test_speed_np_builtins(size, nLoop, nRep=1):
	setupCode = (
		"import numpy as np;" +
		"PRNG = np.random.RandomState(0);" + 
		"x = PRNG.rand(%d);" % (size)
		)
	pprint_timeit(
		stmt='np.argmax(x)',
		setup=setupCode, number=nLoop, repeat=nRep)

	pprint_timeit(
		stmt='np.argsort(x)',
		setup=setupCode, number=nLoop, repeat=nRep)

	nnzPerRows = [0]
	for expval in np.arange(0, np.ceil(np.log2(size/2))):
		nnzPerRows.append(2**expval)

	for nnzPerRow in nnzPerRows:
		funcCode = 'np.argpartition(x, %d)' % (nnzPerRow)
		pprint_timeit(
			stmt=funcCode, setup=setupCode, number=nLoop, repeat=nRep)

def pprint_timeit(*args, **kwargs):
	print kwargs['stmt']
	result_list = timeit.repeat(*args, **kwargs)
	print '  %9.6f sec' % (np.min(result_list))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--N', type=int, default=3)
	parser.add_argument('--K', type=int, default=10)
	parser.add_argument('--nnzList', type=str, default='1,2,4,8')

	parser.add_argument('--size', type=int, default=100)
	parser.add_argument('--bestOf', type=int, default=5)
	parser.add_argument('--nLoop', type=int, default=100)
	args = parser.parse_args()

	args.nnzList = [int(i) for i in args.nnzList.split(',')]

	if args.N * args.K < 1e4:
		test_correctness(N=args.N, K=args.K, nnzList=args.nnzList)
	test_speed(N=args.N, K=args.K, nnzList=args.nnzList, nRep=args.nLoop)
