import numpy as np

def tryDeleteProposalForSpecificTarget(
		Data, hmodel,
		LPkwargs=dict(),
		ktarget=0,
		verbose=True,
		nUpdateSteps=50,
		**kwargs):
	''' Execute merge for specific whole dataset

	Returns
	-------
	propModel : HModel
	propSS : SuffStatBag
	propLscore : scalar real
		ELBO score of proposed model
	curModel : HModel
	curSS : SuffStatBag
	curLscore : scalar real
		ELBO score of current model
	'''
	curModel = hmodel.copy()
	propModel = hmodel.copy()

	# Update current
	curLP = curModel.calc_local_params(Data, **LPkwargs)
	curSS = curModel.get_global_suff_stats(Data, curLP, doPrecompEntropy=1)
	curModel.update_global_params(curSS)
	curLscore = curModel.calc_evidence(SS=curSS)

	# Do Delete Proposal
	propResp = np.delete(curLP['resp'], ktarget, axis=1)
	propResp /= propResp.sum(axis=1)[:,np.newaxis]
	assert np.allclose(1.0, propResp.sum(axis=1))
	propLP = dict(resp=propResp)

	propLscoreList = list()
	for step in range(nUpdateSteps):
		if step > 0:
			propLP = propModel.calc_local_params(Data, **LPkwargs)
		propSS = propModel.get_global_suff_stats(
			Data, propLP, doPrecompEntropy=1)
		propModel.update_global_params(propSS)
		propLscore = propModel.calc_evidence(SS=propSS)
		propLscoreList.append(propLscore)
	if verbose:
		print "Deleting cluster %d..." % (ktarget)
		if propLscore - curLscore > 0:
			print "  ACCEPTED"
		else:
			print "  REJECTED"
		print "%.4e  cur ELBO score" % (curLscore)
		print "%.4e prop ELBO score" % (propLscore)
		print "Change in ELBO score: %.4e" % (propLscore - curLscore)
		print ""
	return (
		propModel,
		propSS,
		propLscoreList,
		curModel,
		curSS,
		curLscore)