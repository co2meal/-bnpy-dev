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
		Data.X, K, hmodel.obsModel, smoothFrac=0, smoothFracInit=1.0)
	assert np.all(np.diff(Lscores) <= 0)

def test_Bern(K, N=1000, W=None):
	import SeqOfBinBars9x9
	Data = SeqOfBinBars9x9.get_data(nDocTotal=N, T=1)
	hmodel = bnpy.HModel.CreateEntireModel(
		'VB', 'DPMixtureModel', 'Bern',
		dict(gamma0=10),
		dict(lam1=0.1, lam0=0.1),
		Data)
	if W and W.size != N:	
		PRNG = np.random.RandomState(0)
		W = PRNG.rand(Data.nObs)
	Z, Mu, Lscores = runBregKMeans(
		Data.X, K, hmodel.obsModel,
		W=W, smoothFrac=0.0, smoothFracInit=1.0)
	assert np.all(np.diff(Lscores) <= 0)

def test_Mult(K, N=1000, W=None):
	import BarsK10V900
	Data = BarsK10V900.get_data(nWordsPerDoc=33, nDocTotal=N)
	X = Data.getDocTypeCountMatrix()
	hmodel = bnpy.HModel.CreateEntireModel(
		'VB', 'DPMixtureModel', 'Mult',
		dict(gamma0=10),
		dict(lam=0.01),
		Data)
	if W and np.asarray(W).size != N:	
		PRNG = np.random.RandomState(0)
		W = PRNG.rand(X.shape[0])
	Z, Mu, Lscores = runBregKMeans(
		X, K, hmodel.obsModel,
		W=W, smoothFrac=0.0, smoothFracInit=1.0)
	assert np.all(np.diff(Lscores) <= 0)

if __name__ == '__main__':
	for N in [5, 10, 211, 333, 500, 1000]:
		print ''
		print ''
		for K in [3, 5, 7, 10, 20, 50]:
			if K > N:
				continue
			test_Mult(K, N, W=1)
			# test_ZeroMeanGauss(K, N)