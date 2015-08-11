import numpy as np
from scipy.special import digamma, gammaln

def assignSplitStats(
        Dslice, hmodel, curLPslice, propXSS,
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
        Dslice, hmodel, curLPslice, propXSS,
        **kwargs)

    return xSSslice

def assignSplitStats_DPMixtureModel(
        Dslice, hmodel, curLPslice, propXSS,
        targetUID=0,
        ktarget=None,
        curSSwhole=None,
        **kwargs):
    ''' Reassign target comp. using an existing set of proposal states.

    Returns
    -------
    xSSslice : stats for reassigned mass
        total count is equal to SS.N[ktarget]
        number of components is Kx
    '''
    if ktarget is None:
        ktarget = curSSwhole.uid2k(targetUID)

    tmpModel = hmodel.copy()
    tmpModel.update_global_params(propXSS)

    xLPslice = tmpModel.calc_local_params(Dslice)
    xLPslice['resp'] *= curLPslice['resp'][:, ktarget][:, np.newaxis]

    xSSslice = tmpModel.get_global_suff_stats(
        Dslice, xLPslice, doPrecompEntropy=1)
    xSSslice.setUIDs(propXSS.uids)
    return xSSslice



def assignSplitStats_HDPTopicModel(
        Dslice, curModel, curLPslice, propXSS,
        targetUID=0,
        ktarget=None,
        curSSwhole=None,
        returnPropSS=0,
        LPkwargs=None,
        batchPos=None,
        verbose=False,
        doSortBigToSmall=False,
        mUIDPairs=list(),
        **kwargs):
    ''' Reassign target comp. using an existing set of proposal states.

    Args
    ----
    curModel : bnpy HModel
        must be exact model used to create the local parameters
    curLPslice : dict of local parameters
        obtained exactly from curModel

    Returns
    -------
    xSSslice : stats for reassigned mass
        total count is equal to SS.N[ktarget]
        number of components is Kx
    '''
    if ktarget is None:
        ktarget = curSSwhole.uid2k(targetUID)

    Korig = curModel.obsModel.K
    Kfresh = propXSS.K
    Nfresh_active = propXSS.getCountVec()
    Kfresh_active = np.flatnonzero(Nfresh_active > 1e-50)[-1] + 1

    thetaRem = curModel.allocModel.alpha_E_beta_rem()
    assert np.allclose(thetaRem, curLPslice['thetaRem'])

    # Use propXSS stats to define the likelihood for each token
    # xLPslice['E_log_soft_ev'] is 2D array, Natom x Kfresh 
    tmpModel = curModel.copy()
    tmpModel.obsModel.update_global_params(propXSS)
    xLPslice = tmpModel.obsModel.calc_local_params(Dslice)

    # Initialize alphaEbeta for expansion components
    xalphaEbeta, thetaEmptyComp = _calc_expansion_alphaEbeta(
        curModel, ktarget, Kfresh)
    xalphaEbeta_active = xalphaEbeta[:Kfresh_active]

    # Initialize DocTopicCount and Theta
    xDocTopicCount = np.zeros((Dslice.nDoc, Kfresh))
    xtheta = np.tile(xalphaEbeta, (Dslice.nDoc, 1))

    # Visit each doc and compute token assignments
    for d in range(Dslice.nDoc):
        start = Dslice.doc_range[d]
        stop = Dslice.doc_range[d+1]

        wc_d = Dslice.word_count[start:stop]
        xLik_d = xLPslice['E_log_soft_ev'][start:stop, :Kfresh_active]
        np.exp(xLik_d, out=xLik_d)
       
        targetsumResp_d = curLPslice['resp'][start:stop, ktarget] * wc_d
        xsumResp_d = np.zeros_like(targetsumResp_d)
        
        xDocTopicCount_d = np.zeros(Kfresh_active)
        xDocTopicProb_d = np.zeros_like(xDocTopicCount_d)
        prevxDocTopicCount_d = 500*np.ones(Kfresh_active)
        for riter in range(LPkwargs['nCoordAscentItersLP']):
            np.add(xDocTopicCount_d, xalphaEbeta_active, out=xDocTopicProb_d)
            digamma(xDocTopicProb_d, out=xDocTopicProb_d)
            xDocTopicProb_d -= xDocTopicProb_d.max()
            np.exp(xDocTopicProb_d, out=xDocTopicProb_d)
            
            # Update sumResp for all tokens in document
            np.dot(xLik_d, xDocTopicProb_d, out=xsumResp_d)

            # Update DocTopicCount_d: 1D array, shape K
            #     sum(DocTopicCount_d) equals Nd[ktarget]
            np.dot(targetsumResp_d / xsumResp_d, xLik_d, out=xDocTopicCount_d)
            xDocTopicCount_d *= xDocTopicProb_d

            if riter % 5 == 0:
                maxDiff_d = np.max(np.abs(
                    prevxDocTopicCount_d - xDocTopicCount_d))
                if maxDiff_d < LPkwargs['convThrLP']:
                    break
            prevxDocTopicCount_d[:] = xDocTopicCount_d

        if verbose:
            print '---'
            weightedtrueDocTopicCount_d = np.dot(
                targetsumResp_d,
                Dslice.TrueParams['resp'][start:stop])
            print ' '.join(
                ['%6.1f' % x for x in weightedtrueDocTopicCount_d])
            print ' '

        # Create proposal resp for relevant atoms in this doc only
        xResp_d = xLik_d
        xResp_d *= xDocTopicProb_d[np.newaxis, :]
        xResp_d /= xsumResp_d[:, np.newaxis]
        xResp_d *= curLPslice['resp'][start:stop, ktarget][:, np.newaxis]
        np.maximum(xResp_d, 1e-100, out=xResp_d)
        # Fill in values in appropriate row of xDocTopicCount and xtheta
        xDocTopicCount[d, :Kfresh_active] = xDocTopicCount_d
        xtheta[d, :Kfresh_active] += xDocTopicCount_d

    # E_log_soft_ev field really contains resp at this point,
    # so just rename this field as 'resp'
    xLPslice['resp'] = xLPslice['E_log_soft_ev']
    xLPslice['resp'][:, Kfresh_active:] = 1e-100
    # Insert DocTopicCount and theta into the LP dict we're building
    xLPslice['DocTopicCount'] = xDocTopicCount
    xLPslice['theta'] = xtheta
    xLPslice['thetaRem'] = thetaRem
    assert np.allclose(xLPslice['resp'].sum(axis=1),
                       curLPslice['resp'][:, ktarget])
    assert np.allclose(xLPslice['DocTopicCount'].sum(axis=1),
                       curLPslice['DocTopicCount'][:, ktarget])
    assert np.allclose(thetaEmptyComp + xtheta.sum(axis=1),
                       curLPslice['theta'][:, ktarget])

    # Compute quantities related to log prob (topic | doc)
    # and fill these into the LP dict
    digammaSumTheta = curLPslice['digammaSumTheta'].copy()
    xLPslice['digammaSumTheta'] = digammaSumTheta
    xElogPi = digamma(xtheta) - digammaSumTheta[:, np.newaxis]
    ElogPiRem = digamma(thetaRem) - digammaSumTheta
    xLPslice['ElogPi'] = xElogPi
    xLPslice['ElogPiRem'] = ElogPiRem

    # Compute quantities related to leaving ktarget empty,
    # as we expand and transfer mass to other comps
    ElogPiEmptyComp = digamma(thetaEmptyComp) - digammaSumTheta
    xLPslice['thetaEmptyComp'] = thetaEmptyComp
    xLPslice['ElogPiEmptyComp'] = ElogPiEmptyComp
    xLPslice['ElogPiOrigComp'] = curLPslice['ElogPi'][:, ktarget]
    xLPslice['gammalnThetaOrigComp'] = np.sum(
        gammaln(curLPslice['theta'][:, ktarget]))
    slack = curLPslice['DocTopicCount'][:, ktarget] - \
            curLPslice['theta'][:, ktarget]
    xLPslice['slackThetaOrigComp'] = np.sum(
        slack * curLPslice['ElogPi'][:, ktarget])

    # Compute sufficent stats for expanded local parameters
    xSSslice = tmpModel.get_global_suff_stats(
        Dslice, xLPslice,
        trackDocUsage=1, doPrecompEntropy=1, doTrackTruncationGrowth=1)

    if batchPos == 0 and doSortBigToSmall:
        order = np.argsort(-1 * xSSslice.getCountVec())
        xSSslice.reorderComps(order)
    else:
        order = None
    xSSslice.setUIDs(propXSS.uids)

    if mUIDPairs is not None and len(mUIDPairs) > 0:
        xSSslice = curModel.allocModel.calcMergeTermsFromSeparateLP(
            Dslice, xLPslice, xSSslice, 
            curLPslice, curSSwhole.uids,
            mUIDPairs)

    if returnPropSS:
        return _verify_HDPTopicModel_and_return_xSSslice_and_propSSslice(
            Dslice, curModel, curLPslice, xLPslice, xSSslice, 
            ktarget=ktarget, order=order, **kwargs)
    return xSSslice

def _calc_expansion_alphaEbeta(curModel, ktarget=0, Kfresh=0):
    ''' Calculate values of alphaEbeta for expansion of Kfresh new components

    Leaves some fraction of mass leftover for the displaced component.

    Returns
    -------
    xalphaEbeta_vec : 1D array, size K
        sum plus xalphaEbeta_empty will be equal to alphaEbeta[ktarget]
    xalphaEbeta_empty : scalar
    '''
    target_alphaEbeta = curModel.allocModel.alpha_E_beta()[ktarget]
    xalphaEbeta_vec = target_alphaEbeta * 1.0 / (Kfresh + 1) * np.ones(Kfresh)
    xalphaEbeta_empty = target_alphaEbeta * 1.0 / (Kfresh + 1)
    return xalphaEbeta_vec, xalphaEbeta_empty

def _verify_HDPTopicModel_and_return_xSSslice_and_propSSslice(
            Dslice, curModel, curLPslice, xLPslice, xSSslice, 
            ktarget=None,
            order=None,
            **kwargs):
        '''

        Returns
        -------
        xSSslice
        propSSslice : optional
        '''
        Kfresh = xLPslice['resp'].shape[1]
        Korig = curLPslice['resp'].shape[1]
        xalphaEbeta = _calc_expansion_alphaEbeta(curModel, ktarget, Kfresh)

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
        assert np.allclose(xLPslice['DocTopicCount'],
                           propLPslice['DocTopicCount'][:, Korig:])
        assert np.allclose(xLPslice['theta'],
                           propLPslice['theta'][:, Korig:])
        propSSslice = curModel.get_global_suff_stats(
            Dslice, propLPslice, doPrecompEntropy=1, doTrackTruncationGrowth=1)
        if order is not None:
            order = np.hstack([np.arange(Korig), order+Korig])
            propSSslice.reorderComps(order)
            propSSslice.setUIDs(np.arange(propSSslice.K))
        assert np.allclose(propSSslice.getELBOTerm('gammalnTheta')[ktarget],
                           Dslice.nDoc * gammaln(propalphaEbeta[ktarget]))
        slackThetaEmpty = np.sum(
            (0 - xalphaEbeta[0]) * propLPslice['ElogPi'][:, ktarget])
        slackThetaOrig = np.sum(
            (curLPslice['DocTopicCount'][:, ktarget] - \
            curLPslice['theta'][:, ktarget]) * \
            curLPslice['ElogPi'][:, ktarget])

        assert np.allclose(xLPslice['slackThetaOrigComp'], slackThetaOrig)
        assert np.allclose(propSSslice.getELBOTerm('slackTheta')[ktarget],
                           slackThetaEmpty)
        assert np.allclose(xSSslice.getELBOTerm('slackThetaEmptyComp'),
                           slackThetaEmpty - slackThetaOrig)

        return xSSslice, propSSslice
