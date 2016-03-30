import numpy as np

def tryMergeProposalForSpecificTarget(
		Data, hmodel,
		LPkwargs=dict(),
		kA=0,
		kB=1,
		verbose=True,
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

	# Update proposal
	propResp = np.delete(curLP['resp'], kB, axis=1)
	propResp[:, kA] += curLP['resp'][:, kB]
	assert np.allclose(1.0, propResp.sum(axis=1))
	propLP = dict(resp=propResp)
	propSS = propModel.get_global_suff_stats(Data, propLP, doPrecompEntropy=1)
	propModel.update_global_params(propSS)
	propLscore = propModel.calc_evidence(SS=propSS)

	if verbose:
		print "Merging cluster %d and %d ..." % (kA, kB)
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
		propLscore,
		curModel,
		curSS,
		curLscore)
