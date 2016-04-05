import numpy as np

from scipy.special import digamma, gammaln
from bnpy.util import NumericUtil
from bnpy.allocmodel import make_xPiVec_and_emptyPi


def summarizeRestrictedLocalStep_HDPTopicModel(
        Dslice=None, 
        curModel=None,
        curLPslice=None,
        curSSwhole=None,
        targetUID=None,
        ktarget=None,
        kabsorbList=None,
        xUIDs=None,
        xObsModel=None,
        xInitSS=None,
        doBuildOnInit=False,
        xPiVec=None,
        emptyPi=0.0,
        nUpdateSteps=5,
        **kwargs):
    ''' Perform restricted local step and summarize it.

    Returns
    -------
    xSSslice : SuffStatBag
    Info : dict with other information
    '''
    # Translate specififed unique-IDs (UID) into current order IDs
    if targetUID is not None:
        ktarget = curSSwhole.uid2k(targetUID)
    if xUIDs is not None:
        kabsorbList = list()
        for uid in xUIDs:
            kabsorbList.append(curSSwhole.uid2k(uid))
        kabsorbList.sort()

    # Create probabilities for each of the Kfresh new clusters
    # by subdividing the target comp's original probabilities
    if xPiVec is None:
        piVec = curModel.allocModel.get_active_comp_probs()
        xPiVec = piVec[kabsorbList].copy()
        xPiVec /= xPiVec.sum()
        xPiVec *= (piVec[kabsorbList].sum() +  piVec[ktarget])
        assert np.allclose(np.sum(xPiVec),
            piVec[ktarget] + np.sum(piVec[kabsorbList]))

    xalphaPi = curModel.allocModel.alpha * xPiVec
    thetaEmptyComp = curModel.allocModel.alpha * emptyPi
    # Create expansion observation model, if necessary
    if xObsModel is None:
        assert xInitSS is not None
        xObsModel = curModel.obsModel.copy()
        xObsModel.update_global_params(xInitSS)
        assert xObsModel.K == len(kabsorbList)
    # Perform restricted inference!
    # xLPslice contains local params for all Kfresh expansion clusters
    xLPslice = restrictedLocalStep_HDPTopicModel(
        Dslice=Dslice,
        curLPslice=curLPslice,
        ktarget=ktarget,
        kabsorbList=kabsorbList,
        xObsModel=xObsModel,
        xalphaPi=xalphaPi,
        thetaEmptyComp=thetaEmptyComp,
        nUpdateSteps=nUpdateSteps,
        doBuildOnInit=doBuildOnInit,
        xInitSS=xInitSS,
        **kwargs)
    if emptyPi > 0:
        assert "HrespOrigComp" in xLPslice

    # Summarize this expanded local parameter pack
    xSSslice = curModel.get_global_suff_stats(
        Dslice, xLPslice,
        trackDocUsage=1, doPrecompEntropy=1, doTrackTruncationGrowth=1)
    if xUIDs is not None:
        xSSslice.setUIDs(xUIDs)
    assert xSSslice.hasELBOTerm("Hresp")
    if emptyPi > 0:
        assert xSSslice.hasELBOTerm("HrespEmptyComp")

    # Prepare dict of info for debugging/inspection
    Info = dict()
    Info['Kfresh'] = xPiVec.size
    Info['xLPslice'] = xLPslice
    Info['xPiVec'] = xPiVec
    Info['emptyPi'] = emptyPi
    return xSSslice, Info

def restrictedLocalStep_HDPTopicModel(
        Dslice=None, 
        curLPslice=None,
        ktarget=0,
        kabsorbList=None,
        xObsModel=None,
        xalphaPi=None,
        nUpdateSteps=3,
        doBuildOnInit=False,
        convThr=0.5,
        thetaEmptyComp=None,
        **kwargs):
    '''

    Returns
    -------
    xLPslice : dict with updated fields
        Fields with learned values
        * resp : N x Kfresh
        * DocTopicCount : nDoc x Kfresh
        * theta : nDoc x Kfresh
        * ElogPi : nDoc x Kfresh

        Fields copied directly from curLPslice
        * digammaSumTheta : 1D array, size nDoc
        * thetaRem : scalar
        * ElogPiRem : scalar
    '''
    if doBuildOnInit:
        xWholeSS = xInitSS.copy()

    Kfresh = xObsModel.K
    assert Kfresh == xalphaPi.size

    # Initialize values
    xLPslice = dict()
    xLPslice['resp'] = \
        curLPslice['resp'][:, kabsorbList].copy()
    xLPslice['DocTopicCount'] = \
        curLPslice['DocTopicCount'][:, kabsorbList].copy()
    xLPslice['theta'] = \
        xLPslice['DocTopicCount'] + xalphaPi[np.newaxis,:]

    xLPslice['_nIters'] = -1 * np.ones(Dslice.nDoc)
    xLPslice['_maxDiff'] = -1 * np.ones(Dslice.nDoc)

    for step in range(nUpdateSteps):
        # Compute conditional likelihoods for every data atom
        xLPslice = xObsModel.calc_local_params(Dslice, xLPslice)
        assert 'E_log_soft_ev' in xLPslice
        assert 'obsModelName' in xLPslice

        # Fill in these fields, one doc at a time
        for d in xrange(Dslice.nDoc):
            xLPslice = restrictedLocalStepForSingleDoc_HDPTopicModel(
                d=d,
                Dslice=Dslice,
                curLPslice=curLPslice,
                xLPslice=xLPslice,
                ktarget=ktarget,
                kabsorbList=kabsorbList,
                xalphaPi=xalphaPi,
                **kwargs)

        isLastStep = step == nUpdateSteps - 1
        if not isLastStep:
            xSS = xObsModel.calcSummaryStats(Dslice, None, xLPslice)
            # Increment
            if doBuildOnInit:
                xSS.setUIDs(xWholeSS.uids)
                xWholeSS += xSS
            else:
                xWholeSS = xSS
            # Global step
            xObsModel.update_global_params(xWholeSS)
            # Decrement stats
            if doBuildOnInit:
                xWholeSS -= xSS
            # Assess early stopping
            if step > 0:
                thr = np.sum(np.abs(prevCountVec - xSS.getCountVec()))
                if thr < convThr:
                    break
            prevCountVec = xSS.getCountVec()

    # Compute other LP quantities related to log prob (topic | doc)
    # and fill these into the expanded LP dict
    digammaSumTheta = curLPslice['digammaSumTheta'].copy()
    xLPslice['digammaSumTheta'] = digammaSumTheta
    xLPslice['ElogPi'] = \
        digamma(xLPslice['theta']) - digammaSumTheta[:, np.newaxis]
    xLPslice['thetaRem'] = curLPslice['thetaRem'].copy()
    xLPslice['ElogPiRem'] = curLPslice['ElogPiRem'].copy()

    # Compute quantities related to leaving ktarget almost empty,
    # as we expand and transfer mass to other comps
    if thetaEmptyComp > 0:
        ElogPiEmptyComp = digamma(thetaEmptyComp) - digammaSumTheta
        xLPslice['thetaEmptyComp'] = thetaEmptyComp
        xLPslice['ElogPiEmptyComp'] = ElogPiEmptyComp

        # Compute quantities related to OrigComp, the original target cluster.
        # These need to be tracked and turned into relevant summaries
        # so that they can be used to created a valid proposal state "propSS"
        xLPslice['ElogPiOrigComp'] = curLPslice['ElogPi'][:, ktarget]
        xLPslice['gammalnThetaOrigComp'] = \
            np.sum(gammaln(curLPslice['theta'][:, ktarget]))
        slack = curLPslice['DocTopicCount'][:, ktarget] - \
            curLPslice['theta'][:, ktarget]
        xLPslice['slackThetaOrigComp'] = np.sum(
            slack * curLPslice['ElogPi'][:, ktarget])

        if hasattr(Dslice, 'word_count') and \
                xLPslice['resp'].shape[0] == Dslice.word_count.size:
            xLPslice['HrespOrigComp'] = -1 * NumericUtil.calcRlogRdotv(
                curLPslice['resp'][:, ktarget], Dslice.word_count)
        else:
            xLPslice['HrespOrigComp'] = -1 * NumericUtil.calcRlogR(
                curLPslice['resp'][:, ktarget])
    return xLPslice

def restrictedLocalStepForSingleDoc_HDPTopicModel(
        d=0,
        Dslice=None,
        curLPslice=None,
        ktarget=0,
        kabsorbList=None,
        xalphaPi=None,
        xLPslice=None,
        LPkwargs=dict(),
        **kwargs):
    ''' Perform restricted local step on one document.

    Returns
    -------
    xLPslice : dict with updated entries related to document d
        * resp
        * DocTopicCount
        * theta
    '''
    # Verify we have likelihoods
    assert 'E_log_soft_ev' in xLPslice
    assert 'obsModelName' in xLPslice
    obsModelName = xLPslice['obsModelName']
    # Verify prior
    Kfresh = xalphaPi.size
    assert xLPslice['E_log_soft_ev'].shape[1] == Kfresh

    if hasattr(Dslice, 'word_count') and obsModelName.count('Bern'):
        raise ValueError("TODO")
    start = Dslice.doc_range[d]
    stop = Dslice.doc_range[d+1]

    constrained_sumTheta_d = curLPslice['theta'][d,ktarget] + \
        np.sum(curLPslice['theta'][d, kabsorbList])

    # Establish the total mass we must reallocate
    constrained_sumResp_d = curLPslice['resp'][start:stop,ktarget] + \
        np.sum(curLPslice['resp'][start:stop, kabsorbList], axis=1)
    #mask_d = np.flatnonzero(constrained_sumResp_d > 1e-5)
    mask_d = np.arange(stop-start)

    if mask_d.size == 0:
        return xLPslice
    # Compute the conditional likelihood matrix for the target atoms
    # xCLik_d will always have an entry equal to one.
    if mask_d.size > 0:
        xCLik_d = xLPslice['E_log_soft_ev'][start + mask_d].copy()
        xCLik_d -= np.max(xCLik_d, axis=1)[:,np.newaxis]
        #???Protect against underflow
        #np.maximum(xCLik_d, -300, out=xCLik_d)
        np.exp(xCLik_d, out=xCLik_d)

    constrained_sumResp_d = constrained_sumResp_d[mask_d].copy()
    if hasattr(Dslice, 'word_count') and obsModelName.count('Mult'):
        wc_d = Dslice.word_count[start + mask_d]
        wc_d *= constrained_sumResp_d
    else:
        wc_d = constrained_sumResp_d

    # Allocate temporary memory for this document
    xsumResp_d = np.zeros(xCLik_d.shape[0])      
    xDocTopicCount_d = np.zeros(Kfresh)

    # Initialize
    xDocTopicProb_d = xalphaPi.copy()
    xDocTopicCount_d = xLPslice['DocTopicCount'][d, :].copy()

    prevxDocTopicCount_d = -1 * np.ones(Kfresh)
    maxDiff_d = -1
    for riter in range(LPkwargs['nCoordAscentItersLP']):
        np.add(xDocTopicCount_d, xalphaPi, 
            out=xDocTopicProb_d)
        digamma(xDocTopicProb_d, out=xDocTopicProb_d)
        # Protect against underflow
        np.maximum(xDocTopicProb_d, -300, out=xDocTopicProb_d)
        np.exp(xDocTopicProb_d, out=xDocTopicProb_d)
        assert np.min(xDocTopicProb_d) > 0.0

        # Update sumResp for active tokens in document
        np.dot(xCLik_d, xDocTopicProb_d, out=xsumResp_d)
        # Update DocTopicCount_d: 1D array, shape K
        np.dot(wc_d / xsumResp_d, xCLik_d, 
               out=xDocTopicCount_d)
        xDocTopicCount_d *= xDocTopicProb_d

        if riter % 5 == 0:
            maxDiff_d = np.max(np.abs(
                prevxDocTopicCount_d - xDocTopicCount_d))
            if maxDiff_d < LPkwargs['convThrLP']:
                break
        prevxDocTopicCount_d[:] = xDocTopicCount_d

    assert np.all(np.isfinite(xDocTopicCount_d))
    xResp_d = xCLik_d
    xResp_d *= xDocTopicProb_d[np.newaxis, :]
    xResp_d /= xsumResp_d[:, np.newaxis]
    # Here, sum of each row of xResp_d is equal to 1.0

    # Need to make sum of each row equal mass on target cluster
    xResp_d *= constrained_sumResp_d[:,np.newaxis]
    np.maximum(xResp_d, 1e-100, out=xResp_d)

    # Pack up into final LP dict
    xLPslice['resp'][start+mask_d] = xResp_d
    xLPslice['DocTopicCount'][d, :] = xDocTopicCount_d
    xLPslice['theta'][d, :] = xalphaPi + xDocTopicCount_d
    # Final verifcation that output meets required constraints
    respOK = np.allclose(
        xLPslice['resp'][start:stop].sum(axis=1),
        curLPslice['resp'][start:stop, ktarget] +
        curLPslice['resp'][start:stop, kabsorbList].sum(axis=1),
        atol=1e-5,
        rtol=0)
    assert respOK
    thetaOK = np.allclose(
        xLPslice['theta'][d, :].sum(),
        constrained_sumTheta_d,
        atol=0.01,
        rtol=0)
    assert thetaOK
    # That's all folks
    return xLPslice

