import numpy as np
import bnpy
runBregKMeans = bnpy.init.FromScratchGauss.runKMeans_bregmanDiv

from bnpy.viz.PlotUtil import pylab

def main(K, N=1000):

	import BarsK10V900
	Data = BarsK10V900.get_data(nWordsPerDoc=33, nDocTotal=N)
	X = Data.getDocTypeCountMatrix()

	hmodel = bnpy.HModel.CreateEntireModel(
		'VB', 'DPMixtureModel', 'Mult',
		dict(gamma0=10),
		dict(lam=0.1),
		Data)
	
	Z, Mu, Lscores = runBregKMeans(X, K, hmodel.obsModel)
	assert np.all(np.diff(Lscores) <= 0)

if __name__ == '__main__':
	for N in [200, 333, 500, 1000]:
		for K in [3, 5, 10, 20, 50]:
			main(K, N)