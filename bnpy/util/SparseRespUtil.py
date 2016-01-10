import argparse
import numpy as np
import scipy.sparse
import timeit

def denseResp2sparse_csr(resp, nnzPerRow=1):
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

def test_speed_argsort(size, nLoop, nRep=1):
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
	parser.add_argument('--size', type=int, default=100)
	parser.add_argument('--bestOf', type=int, default=5)
	parser.add_argument('--nLoop', type=int, default=100)
	args = parser.parse_args()

	test_speed_argsort(args.size, args.nLoop, args.bestOf)

	'''
	resp = np.random.rand(10, 5)
	resp *= resp
	resp /= resp.sum(axis=1)[:,np.newaxis]
	assert np.allclose(np.sum(resp, axis=1), 1)

	spR_2 = denseResp2sparse_csr(resp, nnzPerRow=2)
	R2 = spR_2.toarray()

	spR_4 = denseResp2sparse_csr(resp, nnzPerRow=4)
	R4 = spR_4.toarray()

	from IPython import embed; embed()
	'''