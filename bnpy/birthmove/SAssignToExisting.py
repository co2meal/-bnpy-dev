import numpy as np

def assignSplitStats_DPMixture(
		Dslice, hmodel, curLPslice, curSSwhole, propXSS,
		targetUID=0,
		**kwargs):
	''' Reassign target comp. using an existing set of proposal states.

	Returns
	-------
	xSSslice : stats for reassigned mass
		total count is equal to SS.N[ktarget]
		number of components is Kx
	'''
	ktarget = curSSwhole.uid2k(targetUID)

	tmpModel = hmodel.copy()
	tmpModel.update_global_params(propXSS)

	xLPslice = tmpModel.calc_local_params(Dslice)
	xLPslice['resp'] /= curLPslice['resp'][:, ktarget][:, np.newaxis]

	xSSslice = tmpModel.get_global_suff_stats(
		Dslice, xLPslice, doPrecompEntropy=1)
	xSSslice.setUIDs(propXSS.uids)
	return xSSslice