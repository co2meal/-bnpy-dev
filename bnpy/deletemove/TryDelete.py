import argparse
import numpy as np

def tryDeleteProposalForSpecificTarget_DPMixtureModel(
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



def tryDeleteProposalForSpecificTarget_HDPTopicModel(
		Data, hmodel,
		LPkwargs=dict(),
		ktarget=0,
		kabsorbList=[1],
		verbose=True,
		doPlotComps=True,
		doPlotELBO=True,
		nELBOSteps=3,
		nUpdateSteps=5,
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
	from bnpy.allocmodel.topics.HDPTopicRestrictedLocalStep2 \
		import summarizeRestrictedLocalStep_HDPTopicModel
	curModel = hmodel.copy()
	propModel = hmodel.copy()

	# Update current
	curLP = curModel.calc_local_params(Data, **LPkwargs)
	curSS = curModel.get_global_suff_stats(Data, curLP, doPrecompEntropy=1)
	curModel.update_global_params(curSS)
	curLscore = curModel.calc_evidence(SS=curSS)

	# Create init observation model for absorbing states
	xObsModel = propModel.obsModel.copy()
	xinitSS = curSS.copy()
	for k in reversed(np.arange(xObsModel.K)):
		if k not in kabsorbList:
			xinitSS.removeComp(k)
	xObsModel.update_global_params(xinitSS)

	# Create init pi vector for absorbing states
	piVec = propModel.allocModel.get_active_comp_probs()
	xPiVec = piVec[kabsorbList].copy()
	xPiVec /= xPiVec.sum()
	xPiVec *= (piVec[kabsorbList].sum() +  piVec[ktarget])
	assert np.allclose(np.sum(xPiVec),
		piVec[ktarget] + np.sum(piVec[kabsorbList]))
	propLscoreList = list()
	for ELBOstep in range(nELBOSteps):
		xSS, Info = summarizeRestrictedLocalStep_HDPTopicModel(
			Dslice=Data,
			curModel=curModel,
			curLPslice=curLP,
			ktarget=ktarget,
			kabsorbList=kabsorbList,
			xPiVec=xPiVec,
			xObsModel=xObsModel,
			nUpdateSteps=nUpdateSteps,
			LPkwargs=LPkwargs)

		if ELBOstep < nELBOSteps - 1:
			# Update the xObsModel
			xObsModel.update_global_params(xSS)
			# TODO: update xPiVec???

		propSS = curSS.copy()
		propSS.replaceCompsWithContraction(
			replaceSS=xSS,
			replaceUIDs=[curSS.uids[k] for k in kabsorbList],
			removeUIDs=[curSS.uids[ktarget]],
			)
		assert np.allclose(propSS.getCountVec().sum(),
			curSS.getCountVec().sum(),
			atol=0.01,
			rtol=0)
		propModel.update_global_params(propSS)
		propLscore = propModel.calc_evidence(SS=propSS)
		propLscoreList.append(propLscore)
	if verbose:
		print "Deleting cluster %d" % (ktarget)
		print "Absorbing into clusters %s" % (str(kabsorbList))
		if propLscore - curLscore > 0:
			print "  ACCEPTED"
		else:
			print "  REJECTED"
		print "%.4e  cur ELBO score" % (curLscore)
		print "%.4e prop ELBO score" % (propLscore)
		print "Change in ELBO score: %.4e" % (propLscore - curLscore)
		print ""
	if doPlotELBO:
		import bnpy.viz
		from bnpy.viz.PlotUtil import pylab
		bnpy.viz.PlotUtil.ConfigPylabDefaults(pylab)
		iters = np.arange(len(propLscoreList))
		pylab.plot(iters, propLscoreList, 'b-')
		pylab.plot(iters, curLscore*np.ones_like(iters), 'k--')
		pylab.show()
	if doPlotComps:
		import bnpy.viz
		from bnpy.viz.PlotUtil import pylab
		bnpy.viz.PlotUtil.ConfigPylabDefaults(pylab)
		bnpy.viz.PlotComps.plotCompsFromHModel(
			curModel,
			vmin=0,
			vmax=.01)
		bnpy.viz.PlotComps.plotCompsFromHModel(
			propModel,
			vmin=0,
			vmax=.01)
		pylab.show()
	return (
		propModel,
		propSS,
		propLscoreList,
		curModel,
		curSS,
		curLscore)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--doPlotELBO', type=int, default=1)
	parser.add_argument('--doPlotComps', type=int, default=0)
	parser.add_argument('--ktarget', type=int, default=10)
	parser.add_argument('--kabsorbList', type=str, default='all')
	parser.add_argument('--initname', type=str, default='truelabelsandjunk')
	parser.add_argument('--K', type=int, default=10)
	parser.add_argument('--nLap', type=int, default=1)
	args = parser.parse_args()
	ktarget = args.ktarget
	kabsorbList = args.kabsorbList

	LPkwargs = dict(
		restartLP=0,
		initDocTopicCountLP='setDocProbsToEGlobalProbs',
		nCoordAscentItersLP=100,
		convThrLP=0.01)

	import bnpy
	hmodel, Info = bnpy.run('BarsK10V900', 'HDPTopicModel', 'Mult', 'memoVB',
		initname=args.initname,
		nLap=args.nLap,
		K=args.K,
		nBatch=1, nDocTotal=100, nWordsPerDoc=500,
		alpha=0.5,
		gamma=10.0,
		lam=0.1,
		**LPkwargs)
	Data = Info['Data'].getBatch(0)
	if kabsorbList == 'all':
		kabsorbList = range(hmodel.obsModel.K)
	elif kabsorbList.count(','):
		kabsorbList = [int(k) for k in kabsorbList.split(',')]
	elif kabsorbList.count('-'):
		kabsorbList = kabsorbList.split('-')
		kabsorbList = range(int(kabsorbList[0]), int(kabsorbList[1])+1)
	else:
		kabsorbList = [int(kabsorbList)]
	if ktarget in kabsorbList:
		kabsorbList.remove(ktarget)
	nIntersect = np.intersect1d(kabsorbList, range(hmodel.obsModel.K)).size
	assert nIntersect == len(kabsorbList)

	tryDeleteProposalForSpecificTarget_HDPTopicModel(
		Data,
		hmodel,
		LPkwargs=LPkwargs,
		ktarget=ktarget,
		kabsorbList=kabsorbList,
		doPlotComps=args.doPlotComps,
		doPlotELBO=args.doPlotELBO,
		nUpdateSteps=10,)
