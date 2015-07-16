import numpy as np
import bnpy
from bnpy.birthmove.SCreateFromScratch import createSplitStats
from bnpy.birthmove.SAssignToExisting import assignSplitStats_DPMixture

def main():
	nBatch = 10

	import AsteriskK8
	Data = AsteriskK8.get_data(nObsTotal=10000)
	Data.alwaysTrackTruth = 1
	DataIterator = Data.to_iterator(nBatch=nBatch, nLap=10)

	hmodel = bnpy.HModel.CreateEntireModel(
		'moVB', 'DPMixtureModel', 'Gauss',
		dict(), dict(ECovMat='eye'), Data)
	hmodel.init_global_params(Data, K=1, initname='kmeans')

	for batchID in xrange(nBatch):
		Dbatch = DataIterator.getBatch(batchID)

		LPbatch = hmodel.calc_local_params(Dbatch)
		SSbatch = hmodel.get_global_suff_stats(
			Dbatch, LPbatch, doPrecompEntropy=1)

		
		if batchID == 0:
			SS = SSbatch.copy()

			xSSbatch = createSplitStats(
				Dbatch, hmodel, LPbatch, curSSwhole=SS,
				creationProposalName='truelabels',
				targetUID=0,
				newUIDs=np.arange(100, 100+25))
			xSS = xSSbatch.copy()
		else:
			xSSbatch = assignSplitStats_DPMixture(
				Dbatch, hmodel, LPbatch, SS, xSS,
				targetUID=0)
			xSS += xSSbatch
			SS += SSbatch

		hmodel.update_global_params(SS)

	from IPython import embed; embed()

if __name__ == '__main__':
	main()