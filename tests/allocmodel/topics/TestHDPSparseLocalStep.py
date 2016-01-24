import numpy as np
import time
import bnpy


if __name__ == '__main__':
	import AdmixAsteriskK8
	Data = AdmixAsteriskK8.get_data(nDocTotal=50, nObsPerDoc=333)
	hmodel, Info = bnpy.run(Data, 'HDPTopicModel', 'Gauss', 'memoVB',
		ECovMat='diagcovdata', sF=0.1,
		nLap=5, initname='randexamples', K=200, nBatch=1)

	nnzPerRowLP = 9
	convThrLP = 0.0001 # Never converge early!
	nCoordAscentItersLP = 50

	tstart = time.time()
	yesaLP = hmodel.calc_local_params(Data, 
		doSparseOnlyAtFinalLP=0, nnzPerRowLP=nnzPerRowLP, activeonlyLP=1,
		convThrLP=convThrLP,
		nCoordAscentItersLP=nCoordAscentItersLP,
		initDocTopicCountLP='setDocProbsToEGlobalProbs')
	a_elapsed = time.time() - tstart
	
	tstart = time.time()
	noaLP = hmodel.calc_local_params(Data, 
		doSparseOnlyAtFinalLP=0, nnzPerRowLP=nnzPerRowLP, activeonlyLP=0,
		convThrLP=convThrLP,
		nCoordAscentItersLP=nCoordAscentItersLP,
		initDocTopicCountLP='setDocProbsToEGlobalProbs')
	b_elapsed = time.time() - tstart

	try:
		assert np.allclose(yesaLP['DocTopicCount'], noaLP['DocTopicCount'])
		assert np.allclose(yesaLP['spR'].toarray(), noaLP['spR'].toarray())
		#assert np.allclose(yesaLP['spR'].data, noaLP['spR'].data)
	except AssertionError:
		print "BADNESS!"

	print "ACTIVE: ", a_elapsed
	print "ALL:    ", b_elapsed
	print "LOCAL STEP DONE!"
	from IPython import embed; embed()