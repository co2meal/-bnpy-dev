import numpy as np

from SAssignToExisting import assignSplitStats
from scipy.special import digamma, gammaln

try:
    import KMeansRex
    RunKMeans = KMeansRex.RunKMeans
except ImportError:
    from scipy.cluster.vq import kmeans2
    def RunKMeans(X, K, seed=0, Niter=50, **kwargs):
        np.random.seed(seed)
        Mu, Z = kmeans2(X, K, minit='points')
        return Mu, Z

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
    aName = hmodel.getAllocModelName()
    funcName = 'createSplitStats_' + aName + '_' + creationProposalName
    if funcName not in createSplitStatsMap:
        raise NotImplementedError('Unrecognized function: ' + funcName)
    
    createSplitStatsFunc = createSplitStatsMap[funcName]
    xSSslice = createSplitStatsFunc(
        Dslice, hmodel, curLPslice, curSSwhole=curSSwhole,
        **kwargs)
    return xSSslice



def createSplitStats_DPMixtureModel_kmeans(
        Dslice, hmodel, curLPslice, curSSwhole=None,
        targetUID=0, LPkwargs=dict(),
        newUIDs=None,
        lapFrac=0,
        **kwargs):
    ''' Reassign target component to new states, via kmeans.

    Returns
    -------
    xSSslice : stats for reassigned mass
        total count is equal to SS.N[ktarget]
        number of components is Kx
    '''
    xK = newUIDs.size
    ktarget = curSSwhole.uid2k(targetUID)
    Dtarget = Dslice.select_subset_by_mask(
        np.flatnonzero(curLPslice['resp'][:, ktarget] > 0.05))
    Xtarget = Dtarget.X
    Mu, Z = RunKMeans(Xtarget, xK, Niter=25, seed=lapFrac)
    Z = Z.flatten()
    resp = np.zeros((Xtarget.shape[0], xK))
    Kused = 0
    for k in range(xK):
        mask_k = Z == k
        Nk = np.sum(mask_k)
        if Nk > 5:
            resp[mask_k, Kused] = 1.0
            Kused += 1
    resp = resp[:, :Kused]
    
    LPtarget = dict(resp=resp)
    SSfake = hmodel.get_global_suff_stats(Dtarget, LPtarget)
    SSfake.setUIDs(newUIDs[:Kused])

    xSSslice = assignSplitStats(
        Dslice, hmodel, curLPslice, curSSwhole, SSfake, targetUID=targetUID)
    return xSSslice


def createSplitStats_DPMixtureModel_truelabels(
        Dslice, hmodel, curLPslice, curSSwhole=None,
        targetUID=0, LPkwargs=dict(),
        newUIDs=None,
        lapFrac=0,
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



DefaultLPkwargs = dict(
    restartLP=1,
    convThrLP=0.001,
    nCoordAscentItersLP=50,
    )

def createSplitStats_HDPTopicModel_truelabels(
        Dslice, curModel, curLPslice, curSSwhole=None,
        targetUID=0, LPkwargs=DefaultLPkwargs,
        newUIDs=None,
        lapFrac=0,
        returnPropSS=0,
        **kwargs):
    ''' Reassign target component to new states, based on true labels.

    Returns
    -------
    xSSslice : stats for reassigned mass
        total count is equal to SS.N[ktarget]
        number of components is Kx
    '''
    ktarget = curSSwhole.uid2k(targetUID)

    trueResp = Dslice.TrueParams['resp']
    Ktrue = trueResp.shape[1]
    Korig = curSSwhole.K

    scaledResp = trueResp
    scaledResp /= curLPslice['resp'][:, ktarget][:, np.newaxis]
    np.maximum(scaledResp, 1e-100, out=scaledResp)

    xLPslice = dict(resp=scaledResp)
    targetalphaEbeta = curModel.allocModel.alpha_E_beta()[ktarget]
    xalphaEbeta = targetalphaEbeta * 1.0 / (Ktrue+1) * np.ones(Ktrue)
    thetaEmptyComp = xalphaEbeta[0] * 1.0
    thetaRem = curLPslice['thetaRem'] * 1.0

    xDocTopicCount = np.zeros((Dslice.nDoc, Ktrue))
    for d in xrange(Dslice.nDoc):
        start = Dslice.doc_range[d]
        stop = Dslice.doc_range[d+1]
        xDocTopicCount[d,:] = np.sum(
            scaledResp[start:stop] * \
            Dslice.word_count[start:stop][:,np.newaxis], axis=0)
    xtheta = xDocTopicCount + xalphaEbeta

    digammaSumTheta = curLPslice['digammaSumTheta'].copy()
    xElogPi = digamma(xtheta) - digammaSumTheta[:, np.newaxis]
    ElogPiRem = digamma(thetaRem) - digammaSumTheta
    ElogPiEmptyComp = digamma(thetaEmptyComp) - digammaSumTheta

    xLPslice['digammaSumTheta'] = digammaSumTheta
    xLPslice['DocTopicCount'] = xDocTopicCount
    xLPslice['theta'] = xtheta
    xLPslice['thetaRem'] = thetaRem
    xLPslice['ElogPi'] = xElogPi
    xLPslice['ElogPiRem'] = ElogPiRem
    xLPslice['thetaEmptyComp'] = thetaEmptyComp
    xLPslice['ElogPiEmptyComp'] = ElogPiEmptyComp - \
        curLPslice['ElogPi'][:, ktarget]
    xLPslice['gammalnThetaOrigComp'] = np.sum(
        gammaln(curLPslice['theta'][:, ktarget]))
    slack = curLPslice['DocTopicCount'][:, ktarget] - \
            curLPslice['ElogPi'][:, ktarget]
    xLPslice['slackThetaOrigComp'] = np.sum(
        slack * curLPslice['ElogPi'][:, ktarget])

    xSSslice = curModel.get_global_suff_stats(
        Dslice, xLPslice, doPrecompEntropy=1, doTrackTruncationGrowth=1)
    xSSslice.setUIDs(newUIDs[:Ktrue])

    if returnPropSS:
        propLPslice = dict()
        propLPslice['resp'] = np.hstack([curLPslice['resp'], xLPslice['resp']])
        propLPslice['resp'][:, ktarget] = 1e-100
        curalphaEbeta = curModel.allocModel.alpha_E_beta().copy()
        propalphaEbetaRem = curModel.allocModel.alpha_E_beta_rem() * 1.0
        propalphaEbeta = np.hstack([curalphaEbeta, xalphaEbeta])
        propalphaEbeta[ktarget] = xalphaEbeta[0] * 1.0
        assert np.allclose(np.sum(propalphaEbeta) + propalphaEbetaRem,
                           curModel.allocModel.alpha)
        propLPslice = curModel.allocModel.initLPFromResp(
            Dslice, propLPslice,
            alphaEbeta=propalphaEbeta,
            alphaEbetaRem=propalphaEbetaRem)
        # Verify computations
        assert np.allclose(xDocTopicCount,
                           propLPslice['DocTopicCount'][:, Korig:])
        assert np.allclose(xtheta,
                           propLPslice['theta'][:, Korig:])
        propSSslice = curModel.get_global_suff_stats(
            Dslice, propLPslice, doPrecompEntropy=1, doTrackTruncationGrowth=1)

        return xSSslice, propSSslice
    return xSSslice