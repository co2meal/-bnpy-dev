import numpy as np

def createSplitStats(
		Dslice, hmodel, curLPslice, curSSwhole=None,
		creationProposalName='truelabels',
		**kwargs):
	''' Reassign target component to new states.

	Returns
	-------
	xSSslice : stats for reassigned mass
		total count is equal to SS.N[ktarget]
		number of components is Kx
	'''
	createSplitStatsMap = dict([
		(k,v) for (k,v) in globals().items()
		if str(k).count('createSplitStats')])
	if str(hmodel.allocModel).count('DPMixture'):
		funcName = 'createSplitStats_DPMixture_' + creationProposalName
	else:
		raise NotImplementedError('TODO')

	createSplitStatsFunc = createSplitStatsMap[funcName]
	xSSslice = createSplitStatsFunc(
		Dslice, hmodel, curLPslice, curSSwhole=curSSwhole,
		**kwargs)
	return xSSslice


def createSplitStats_DPMixture_truelabels(
		Dslice, hmodel, curLPslice, curSSwhole=None,
		targetUID=0, LPkwargs=dict(),
		newUIDs=None,
		**kwargs):
	''' Reassign target component to new states, based on true labels.

	Returns
	-------
	xSSslice : stats for reassigned mass
		total count is equal to SS.N[ktarget]
		number of components is Kx
	'''
	ktarget = curSSwhole.uid2k(targetUID)

	uLabels = np.unique(Dslice.TrueParams['Z'])
	Ktrue = uLabels.size
	trueResp = np.zeros((Dslice.nObs, Ktrue))
	for k in range(Ktrue):
		trueResp[Dslice.TrueParams['Z'] == k, k] = 1.0
	scaledResp = trueResp
	scaledResp /= curLPslice['resp'][:, ktarget][:, np.newaxis]
	np.maximum(scaledResp, 1e-100, out=scaledResp)

	xLPslice = dict(resp=scaledResp)
	xSSslice = hmodel.get_global_suff_stats(
		Dslice, xLPslice, doPrecompEntropy=1)
	xSSslice.setUIDs(newUIDs[:Ktrue])
	return xSSslice