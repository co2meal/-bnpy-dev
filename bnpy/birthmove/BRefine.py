import numpy as np
from scipy.special import digamma, gammaln

import BLogger

def assignSplitStats(
        Dslice, curModel, curLPslice, propXSS,
        **kwargs):
    assignSplitStatsMap = dict([
        (k,v) for (k,v) in globals().items()
        if str(k).count('assignSplitStats')])
    aName = curModel.getAllocModelName()
    funcName = 'assignSplitStats_' + aName
    if funcName not in assignSplitStatsMap:
        raise NotImplementedError('Unrecognized function: ' + funcName)
    
    assignSplitStatsFunc = assignSplitStatsMap[funcName]
    xSSslice = assignSplitStatsFunc(
        Dslice, curModel, curLPslice, propXSS,
        **kwargs)

    return xSSslice

def assignSplitStats_DPMixtureModel(
        Dslice, curModel, curLPslice, propXSS,
        targetUID=0,
        ktarget=None,
        curSSwhole=None,
        mUIDPairs=None,
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

    tmpModel = curModel.copy()
    tmpModel.update_global_params(propXSS)

    xLPslice = tmpModel.calc_local_params(Dslice)
    xLPslice['resp'] *= curLPslice['resp'][:, ktarget][:, np.newaxis]

    xSSslice = tmpModel.get_global_suff_stats(
        Dslice, xLPslice, doPrecompEntropy=1)
    xSSslice.setUIDs(propXSS.uids)

    if mUIDPairs is not None and len(mUIDPairs) > 0:
        Mdict = curModel.allocModel.calcMergeTermsFromSeparateLP(
            Data=Dslice, LPa=curLPslice, SSa=curSSwhole,
            LPb=xLPslice, SSb=xSSslice, 
            mUIDPairs=mUIDPairs)
        xSSslice.setMergeUIDPairs(mUIDPairs)
        for key, arr in Mdict.items():
            xSSslice.setMergeTerm(key, arr, dims='M')

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
        keepTargetCompAsEmpty=True,
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
        curModel, ktarget, Kfresh, keepTargetCompAsEmpty)
    xalphaEbeta_active = xalphaEbeta[:Kfresh_active]

    # Initialize DocTopicCount and Theta
    xDocTopicCount = np.zeros((Dslice.nDoc, Kfresh))
    xtheta = np.tile(xalphaEbeta, (Dslice.nDoc, 1))
    xLPslice['resp'] = xLPslice['E_log_soft_ev']

    # Visit each doc and compute token assignments
    for d in range(Dslice.nDoc):
        start = Dslice.doc_range[d]
        stop = Dslice.doc_range[d+1]

        mask_d = np.flatnonzero(
            curLPslice['resp'][start:stop, ktarget] > 0.01)
        lumpmask_d = np.setdiff1d(np.arange(stop-start), mask_d)
        if hasattr(Dslice, 'word_count'):
            wc_d = Dslice.word_count[start + mask_d]
            wc_lump_d = Dslice.word_count[start + lumpmask_d]
        else:
            wc_d = 1.0
            wc_lump_d = 1.0

        xLik_d = xLPslice['E_log_soft_ev'][
            start + mask_d, :Kfresh_active].copy()
        xLik_d -= np.max(xLik_d, axis=1)[:,np.newaxis]
        np.exp(xLik_d, out=xLik_d)

        lumpMass_d = np.sum(
            curLPslice['resp'][start + lumpmask_d, ktarget] * \
            wc_lump_d)
        targetsumResp_d = curLPslice['resp'][start + mask_d, ktarget] * wc_d

        # Allocate memory for this document
        xsumResp_d = np.zeros_like(targetsumResp_d)        
        xDocTopicCount_d = np.zeros(Kfresh_active)

        # Begin loop
        if mask_d.size > 0:

            xDocTopicProb_d = np.zeros_like(xDocTopicCount_d)
            prevxDocTopicCount_d = 500 * np.ones(Kfresh_active)

            for riter in range(LPkwargs['nCoordAscentItersLP']):
                np.add(xDocTopicCount_d, xalphaEbeta_active, 
                       out=xDocTopicProb_d)
                digamma(xDocTopicProb_d, out=xDocTopicProb_d)
                xDocTopicProb_d -= xDocTopicProb_d.max()
                np.exp(xDocTopicProb_d, out=xDocTopicProb_d)
                
                # Update sumResp for active tokens in document
                np.dot(xLik_d, xDocTopicProb_d, out=xsumResp_d)

                # Update DocTopicCount_d: 1D array, shape K
                #     sum(DocTopicCount_d) equals Nd[ktarget]
                np.dot(targetsumResp_d / xsumResp_d, xLik_d, 
                       out=xDocTopicCount_d)
                xDocTopicCount_d *= xDocTopicProb_d

                if riter % 5 == 0:
                    maxDiff_d = np.max(np.abs(
                        prevxDocTopicCount_d - xDocTopicCount_d))
                    if maxDiff_d < LPkwargs['convThrLP']:
                        break
                prevxDocTopicCount_d[:] = xDocTopicCount_d

            # Make proposal resp for relevant atoms in current doc d
            if np.any(np.isnan(xDocTopicCount_d)):
                # Edge case! Common only when deleting... 
                # Recover from numerical issues in coord ascent
                # by falling back to likelihood only to make resp
                xResp_d = xLik_d
                xResp_d /= xResp_d.sum(axis=1)[:,np.newaxis]

                np.dot(targetsumResp_d, xResp_d, out=xDocTopicCount_d)
            else:
                # Common case: Use valid result of coord ascent
                xResp_d = xLik_d
                xResp_d *= xDocTopicProb_d[np.newaxis, :]
                xResp_d /= xsumResp_d[:, np.newaxis]

            # Here, sum of each row of xResp_d is equal to 1.0
            # Need to make sum of each row equal mass on target cluster
            xResp_d *= curLPslice['resp'][
                start + mask_d, ktarget][:, np.newaxis]
            np.maximum(xResp_d, 1e-100, out=xResp_d)
            assert np.allclose(
                xResp_d.sum(axis=1),
                curLPslice['resp'][start+mask_d, ktarget])
            xLPslice['resp'][start+mask_d, :Kfresh_active] = xResp_d

        if lumpmask_d.size > 0:
            kmax = xDocTopicCount_d.argmax()
            xLPslice['resp'][start+lumpmask_d, :Kfresh_active] = 1e-100
            xLPslice['resp'][start+lumpmask_d, kmax] = \
                curLPslice['resp'][start + lumpmask_d, ktarget]
            xDocTopicCount_d[kmax] += lumpMass_d

            assert np.allclose(
                xLPslice['resp'][
                    start+lumpmask_d, :Kfresh_active].sum(axis=1),
                curLPslice['resp'][start+lumpmask_d, ktarget])

        # Fill in values in appropriate row of xDocTopicCount and xtheta
        xDocTopicCount[d, :Kfresh_active] = xDocTopicCount_d
        xtheta[d, :Kfresh_active] += xDocTopicCount_d

        assert np.allclose(xDocTopicCount[d,:].sum(),
                           curLPslice['DocTopicCount'][d, ktarget])

    # Force all entries beyond the active K to very small.
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
    if keepTargetCompAsEmpty:
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
        Mdict = curModel.allocModel.calcMergeTermsFromSeparateLP(
            Data=Dslice, LPa=curLPslice, SSa=curSSwhole,
            LPb=xLPslice, SSb=xSSslice, 
            mUIDPairs=mUIDPairs)
        xSSslice.setMergeUIDPairs(mUIDPairs)
        for key, arr in Mdict.items():
            xSSslice.setMergeTerm(key, arr, dims='M')

    if returnPropSS:
        return _verify_HDPTopicModel_and_return_xSSslice_and_propSSslice(
            Dslice, curModel, curLPslice, xLPslice, xSSslice, 
            ktarget=ktarget, order=order, **kwargs)
    return xSSslice

def _calc_expansion_alphaEbeta(
        curModel,
        ktarget=0, Kfresh=0,
        keepTargetCompAsEmpty=0):
    ''' Calculate values of alphaEbeta for expansion of Kfresh new components

    Leaves some fraction of mass leftover for the displaced component.

    Returns
    -------
    xalphaEbeta_vec : 1D array, size K
        sum plus xalphaEbeta_empty will be equal to alphaEbeta[ktarget]
    xalphaEbeta_empty : scalar
    '''
    target_aEbeta = curModel.allocModel.alpha_E_beta()[ktarget]
    if keepTargetCompAsEmpty:
        xalphaEbeta_vec = target_aEbeta * 1.0 / (Kfresh + 1) * np.ones(Kfresh)
        xalphaEbeta_empty = target_aEbeta * 1.0 / (Kfresh + 1)
    else:
        xalphaEbeta_vec = target_aEbeta * 1.0 / (Kfresh) * np.ones(Kfresh)
        xalphaEbeta_empty = 0
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
