import numpy as np
import bnpy
runBregKMeans = bnpy.init.FromScratchGauss.runKMeans_bregmanDiv

from bnpy.viz.PlotUtil import pylab

def test_ZeroMeanGauss(K, N=1000, D=1):
	import StarCovarK5
	Data = StarCovarK5.get_data(nObsTotal=N)
	if D < Data.X.shape[1]:
		Data = bnpy.data.XData(X=Data.X[:,:D])
	hmodel = bnpy.HModel.CreateEntireModel(
		'VB', 'DPMixtureModel', 'ZeroMeanGauss',
		dict(gamma0=10),
		dict(ECovMat='eye', sF=0.0001, nu=0.5),
		Data)
	Z, Mu, Lscores = runBregKMeans(
		Data.X, K, hmodel.obsModel, smoothFrac=0)
	assert np.all(np.diff(Lscores) <= 0)

def test_Mult(K, N=1000):
	import BarsK10V900
	Data = BarsK10V900.get_data(nWordsPerDoc=33, nDocTotal=N)
	X = Data.getDocTypeCountMatrix()
	hmodel = bnpy.HModel.CreateEntireModel(
		'VB', 'DPMixtureModel', 'Mult',
		dict(gamma0=10),
		dict(lam=0.003),
		Data)
	Z, Mu, Lscores = runBregKMeans(
		X, K, hmodel.obsModel, smoothFrac=0)
	assert np.all(np.diff(Lscores) <= 0)

if __name__ == '__main__':
	for N in [5, 10, 200, 333, 500, 1000]:
		for K in [5, 10, 20, 50]:
			if K > N:
				continue
			test_Mult(K, N)
			# test_ZeroMeanGauss(K, N)