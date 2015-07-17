import numpy as np
from scipy.special import digamma, gammaln

def assignSplitStats(
        Dslice, hmodel, curLPslice, curSSwhole, propXSS,
        **kwargs):
    assignSplitStatsMap = dict([
        (k,v) for (k,v) in globals().items()
        if str(k).count('assignSplitStats')])
    aName = hmodel.getAllocModelName()
    funcName = 'assignSplitStats_' + aName
    if funcName not in assignSplitStatsMap:
        raise NotImplementedError('Unrecognized function: ' + funcName)
    
    assignSplitStatsFunc = assignSplitStatsMap[funcName]
    xSSslice = assignSplitStatsFunc(
        Dslice, hmodel, curLPslice, curSSwhole, propXSS,
        **kwargs)
    return xSSslice

def assignSplitStats_DPMixtureModel(
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



def assignSplitStats_HDPTopicModel(
        Dslice, curModel, curLPslice, curSSwhole, propXSS,
        targetUID=0,
        returnPropSS=0,
        **kwargs):
    ''' Reassign target comp. using an existing set of proposal states.

    Returns
    -------
    xSSslice : stats for reassigned mass
        total count is equal to SS.N[ktarget]
        number of components is Kx
    '''
    Korig = curSSwhole.K
    Kfresh = propXSS.K
    ktarget = curSSwhole.uid2k(targetUID)

    tmpModel = curModel.copy()
    tmpModel.obsModel.update_global_params(propXSS)

    xLPslice = tmpModel.obsModel.calc_local_params(Dslice)
    xDocTopicCount = np.zeros((Dslice.nDoc, Kfresh))
    xtheta = np.zeros((Dslice.nDoc, Kfresh))
    thetaRem = curModel.allocModel.alpha_E_beta_rem()

    targetalphaEbeta = curModel.allocModel.alpha_E_beta()[ktarget]
    xalphaEbeta = targetalphaEbeta * 1.0 / (Kfresh+1) * np.ones(Kfresh)
    thetaEmptyComp = xalphaEbeta[0] * 1.0

    # From-scratch strategy
    for d in range(Dslice.nDoc):
        start = Dslice.doc_range[d]
        stop = Dslice.doc_range[d+1]

        wc_d = Dslice.word_count[start:stop]

        xLik_d = xLPslice['E_log_soft_ev'][start:stop, :]
        np.exp(xLik_d, out=xLik_d)
       
        targetsumResp_d = curLPslice['resp'][start:stop, ktarget] * wc_d
        xsumResp_d = np.zeros_like(targetsumResp_d)
        
        xDocTopicCount_d = np.zeros(Kfresh)
        xDocTopicProb_d = np.zeros_like(xDocTopicCount_d)

        for riter in range(10):
            np.add(xDocTopicCount_d, xalphaEbeta, out=xDocTopicProb_d)
            digamma(xDocTopicProb_d, out=xDocTopicProb_d)
            xDocTopicProb_d -= xDocTopicProb_d.max()
            np.exp(xDocTopicProb_d, out=xDocTopicProb_d)
            
            # Update sumResp for all tokens in document
            np.dot(xLik_d, xDocTopicProb_d, out=xsumResp_d)

            # Update DocTopicCount_d: 1D array, shape K
            #     sum(DocTopicCount_d) equals Nd[ktarget]
            np.dot(targetsumResp_d / xsumResp_d, xLik_d, out=xDocTopicCount_d)
            xDocTopicCount_d *= xDocTopicProb_d

            DocTopicCount_dnew = np.sum(xDocTopicCount_d)
            assert np.allclose(
                curLPslice['DocTopicCount'][d, ktarget],
                DocTopicCount_dnew,
                rtol=0, atol=1e-6)

        # Create proposal resp for relevant atoms in this doc only
        xResp_d = xLik_d
        xResp_d *= xDocTopicProb_d[np.newaxis, :]
        xResp_d /= xsumResp_d[:, np.newaxis]
        xResp_d *= curLPslice['resp'][start:stop, ktarget][:, np.newaxis]
        np.maximum(xResp_d, 1e-100, out=xResp_d)
        xDocTopicCount[d, :] = xDocTopicCount_d
        xtheta[d, :] = xDocTopicCount_d + xalphaEbeta

    xLPslice['resp'] = xLPslice['E_log_soft_ev'] # modified in-place
    del xLPslice['E_log_soft_ev']

    digammaSumTheta = curLPslice['digammaSumTheta'].copy()
    xLPslice['digammaSumTheta'] = digammaSumTheta
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
    xLPslice['ElogPiEmptyComp'] = ElogPiEmptyComp
    xLPslice['gammalnThetaOrigComp'] = np.sum(
        gammaln(curLPslice['theta'][:, ktarget]))
    slack = curLPslice['DocTopicCount'][:, ktarget] - \
            curLPslice['ElogPi'][:, ktarget]
    xLPslice['slackThetaOrigComp'] = np.sum(
        slack * curLPslice['ElogPi'][:, ktarget])
    print thetaEmptyComp, '<<< empty theta'
    xSSslice = tmpModel.get_global_suff_stats(
        Dslice, xLPslice, doPrecompEntropy=1, doTrackTruncationGrowth=1)
    xSSslice.setUIDs(propXSS.uids)

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
