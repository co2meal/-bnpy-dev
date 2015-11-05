
def makeSummaryForRestrictedLocalStep_HDPTopicModel(
        Dslice=None, 
        curModel=None,
        curLPslice=None,
        ktarget=0,
        xUIDs=None,
        mUIDPairs=None,
        xObsModel=None,
        xInitSS=None,
        emptyPiFrac=0.01,
        **kwargs):
    ''' Perform restricted local step and summarize it.

    Returns
    -------
    xSSslice : SuffStatBag
    Info : dict with other information
    '''
    Kfresh = len(xUIDs)
    # Verify provided summary states used to initialize clusters, if any.
    if xInitSS is not None:
        assert xInitSS.K == Kfresh
        xInitSS.setUIDs(xUIDs)
    # Create temporary observation model for each of Kfresh new clusters
    # If it doesn't exist already
    if xObsModel is None:
        xObsModel = curModel.obsModel.copy()
    if xInitSS is not None:      
        xObsModel.update_global_step(xInitSS)
    assert xObsModel.K == Kfresh

    # Create temporary probabilities for each new cluster
    target_alphaPi = curModel.allocModel.alpha_E_beta_active()[ktarget]
    xalphaPi = (1-emptyPiFrac) / Kfresh * target_alphaPi * np.ones(Kfresh)
    emptyalphaPi = emptyPiFrac * target_alphaPi
    assert np.allclose(np.sum(xalphaPi) + emptyalphaPi, target_alphaPi)

    # Perform restricted inference!
    xLPslice = restrictedLocalStep_HDPTopicModel(
        Dslice=Dslice,
        curLPslice=curLPslice,
        ktarget=ktarget,
        xObsModel=xObsModel,
        xalphaPi=xalphaPi,
        thetaEmptyComp=emptyalphaPi)

    # Summarize this expanded local parameter pack
    xSSslice = curModel.get_global_suff_stats(
        Dslice, xLPslice,
        trackDocUsage=1, doPrecompEntropy=1, doTrackTruncationGrowth=1)
    xSSslice.setUIDs(xUIDs)

    # If desired, add merge terms into the expanded summaries,
    if mUIDPairs is not None and len(mUIDPairs) > 0:
        Mdict = curModel.allocModel.calcMergeTermsFromSeparateLP(
            Data=Dslice, 
            LPa=curLPslice, SSa=curSSwhole,
            LPb=xLPslice, SSb=xSSslice, 
            mUIDPairs=mUIDPairs)
        xSSslice.setMergeUIDPairs(mUIDPairs)
        for key, arr in Mdict.items():
            xSSslice.setMergeTerm(key, arr, dims='M')

    return xSSslice, Info


def restrictedLocalStep_HDPTopicModel(
        Dslice=None, 
        curLPslice=None,
        ktarget=0,
        xObsModel=None,
        xalphaPi=None,
        thetaEmptyComp=None,
        ):
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

        * thetaEmptyComp
        * ElogPiEmptyComp
    '''
    Kfresh = xObsModel.K
    assert Kfresh == xPi.size

    # Compute conditional likelihoods for every data atom
    xLPslice = xObsModel.calc_local_params(Dslice)
    assert 'E_log_soft_ev' in xLPslice

    # Initialize DocTopicCount and theta
    xLPslice['resp'] = xLPslice['E_log_soft_ev']
    xLPslice['DocTopicCount'] = np.zeros((Dslice.nDoc, Kfresh))
    xLPslice['theta'] = np.zeros((Dslice.nDoc, Kfresh))

    # Fill in these fields, one doc at a time
    for d in xrange(Dslice.nDoc):
        xLPslice = restrictedLocalStepForSingleDoc_HDPTopicModel(
            d=d, Dslice=Dslice,
            curLPslice=curLPslice,
            xLPslice=xLPslice,
            ktarget=ktarget,
            Kfresh=Kfresh,
            xalphaPi_d=xalphaPi,
            )

    # Compute other LP quantities related to log prob (topic | doc)
    # and fill these into the expanded LP dict
    digammaSumTheta = curLPslice['digammaSumTheta'].copy()
    xLPslice['digammaSumTheta'] = digammaSumTheta
    xLPslice['ElogPi'] = \
        digamma(xLPslice['theta']) - digammaSumTheta[:, np.newaxis]
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
    return xLPslice

def restrictedLocalStepForSingleDoc_HDPTopicModel(
        d=0, Dslice=None, curLPslice=None,
        ktarget=0,
        Kfresh=None,
        xalphaPi=None,
        xLPslice=None,
        **kwargs):
    ''' Perform restricted local step on one document.

    Returns
    -------
    xLPslice : dict with updated entries related to document d
        * resp
        * DocTopicCount
        * theta
    '''
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
    # Determine total mass assigned to each target atom,
    # We will learn how to redistribute this mass.
    targetsumResp_d = curLPslice['resp'][start + mask_d, ktarget] * wc_d
    # Compute total mass that will be dealt with as lump sum,
    # because it belongs to atoms that are too small to worry about.
    lumpMass_d = np.sum(
        curLPslice['resp'][start + lumpmask_d, ktarget] * wc_lump_d)
    # Compute the conditional likelihood matrix for the target atoms
    # xCLik_d will always have an entry equal to one.
    assert 'E_log_soft_ev' in xLPslice
    xClik_d = xLPslice['E_log_soft_ev'][start + mask_d]
    xClik_d -= np.max(xCLik_d, axis=1)[:,np.newaxis]
    np.exp(xCLik_d, out=xCLik_d)
    # Allocate temporary memory for this document
    xsumResp_d = np.zeros_like(targetsumResp_d)        
    xDocTopicCount_d = np.zeros(Kfresh)
    # Run coordinate ascent that alternatively updates
    # doc-topic counts and resp for document d
    if mask_d.size > 0:
        xDocTopicProb_d = np.zeros_like(xDocTopicCount_d)
        prevxDocTopicCount_d = 500 * np.ones(Kfresh)
        for riter in range(LPkwargs['nCoordAscentItersLP']):
            # xalphaEbeta_active_d potentially includes counts
            # for absorbing states from curLPslice_d
            np.add(xDocTopicCount_d, xalphaPi, 
                out=xDocTopicProb_d)
            digamma(xDocTopicProb_d, out=xDocTopicProb_d)
            xDocTopicProb_d -= xDocTopicProb_d.max()
            np.exp(xDocTopicProb_d, out=xDocTopicProb_d)

            # Update sumResp for active tokens in document
            np.dot(xCLik_d, xDocTopicProb_d, out=xsumResp_d)

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
            xResp_d = xCLik_d
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
        xLPslice['resp'][start+mask_d] = xResp_d

    if lumpmask_d.size > 0:
        kmax = (xDocTopicCount_d + xalphaPi).argmax()
        xLPslice['resp'][start+lumpmask_d, :] = 1e-100
        xLPslice['resp'][start+lumpmask_d, kmax] = \
            curLPslice['resp'][start + lumpmask_d, ktarget]
        xDocTopicCount_d[kmax] += lumpMass_d

    # Fill in values in appropriate row of xDocTopicCount and xtheta
    xLPslice['DocTopicCount'][d, :] = xDocTopicCount_d
    xLPslice['theta'][d, :] = xalphaPi + xDocTopicCount_d
    assert np.allclose(xDocTopicCount[d,:].sum(),
                       curLPslice['DocTopicCount'][d, ktarget])
    assert np.allclose(
            xLPslice['resp'][start:stop, :].sum(axis=1),
            curLPslice['resp'][start:stop, ktarget])

    return xLPslice





def makeExpansionLPForBirthProposal_HDPTopicModel_FromZ(
        Dslice=None,
        curModel=None,
        Z=None,
        curLPslice=None,
        ktarget=None,
        remFrac=0.001,
        ):
    ''' Create expanded local parameters from hard assignments of target subset.

    Returns
    -------
    xLP : dict with fields
        resp : N x Knew, each row sums to sumResp[n]
        DocTopicCount : D x Knew, each row sums to targetsumDTC[d]
        theta : D x Knew
        ElogPi : D x Knew
    '''
    thetaRem = curModel.allocModel.alpha_E_beta_rem()
    assert np.allclose(thetaRem, curLPslice['thetaRem'])

    assert Z.ndim == 1
    assert Z.size == target_sumResp.shape[0]
    assert Z.min() >= 0
    Knew = Z.max() + 1

    # Initialize alphaEbeta for expanded clusters
    target_alphaEbeta = curModel.allocModel.alpha_E_beta()[ktarget]
    rem_alphaEbeta = remFrac * target_alphaEbeta
    x_alphaEbeta = (1-remFrac) * target_alphaEbeta / Knew * np.ones(Knew)


    for d in xrange(Dslice.nDoc):
        start_d = Dslice.doc_range[d]
        stop_d = Dslice.doc_range[d+1]

        target_sumDTC_d = curLPslice['DocTopicCount'][d, ktarget]
        target_sumResp_d = curLPslice['resp'][start_d:stop_d, ktarget]
        target_digammaSumTheta_d = curLPslice['digammaSumTheta'][d]

        
