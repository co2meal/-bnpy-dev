import numpy as np

def expandLP_singleNewState(
		Data_t, curLP_t, tmpModel, curSS_nott,
		**Plan):
	''' Create single new state for all target data.

	Returns
	-------
	propLP_t : dict of local params, with K + 1 states
	'''
	xcurSS_nott = curSS_nott.copy(includeELBOTerms=1, includeMergeTerms=0)
	xcurSS_nott.insertEmptyComps(1)

	propK = curSS_nott.K + 1
	propResp = np.zeros((curLP_t['resp'].shape[0], propK))
	propResp[:, -1] = 1.0

	propLP_t = dict(resp=propResp)
	if hasattr(tmpModel.allocModel, 'initLPFromResp'):
		propLP_t = tmpModel.allocModel.initLPFromResp(Data_t, propLP_t)

	return propLP_t, xcurSS_nott

def expandLP_randomSplit(
		Data_t, curLP_t, tmpModel, curSS_nott,
		PRNG=np.random, **Plan):
	''' Divide target data into two new states, completely at random.

	Returns
	-------
	propLP_t : dict of local params, with K + 2 states
	'''
	Kfresh = 2
	xcurSS_nott = curSS_nott.copy(includeELBOTerms=1, includeMergeTerms=0)
	xcurSS_nott.insertEmptyComps(Kfresh)
	
	propK = curSS_nott.K + Kfresh
	propResp = np.zeros((curLP_t['resp'].shape[0], propK))

	allids = np.arange(Data_t.get_size())
	Aids = PRNG.choice(allids, replace=False, size=10)
	Bids = PRNG.choice(np.setdiff1d(allids, Aids), replace=False, size=10)
	propResp[Aids, -2] = 1.0
	propResp[Bids, -1] = 1.0

	propLP_t = dict(resp=propResp)
	if hasattr(tmpModel.allocModel, 'initLPFromResp'):
		propLP_t = tmpModel.allocModel.initLPFromResp(Data_t, propLP_t)

	propSS = tmpModel.get_global_suff_stats(Data_t, propLP_t)
	propSS += xcurSS_nott 
	tmpModel.update_global_params(propSS)

	propLP_t = tmpModel.calc_local_params(Data_t, propLP_t)
	return propLP_t, xcurSS_nott
