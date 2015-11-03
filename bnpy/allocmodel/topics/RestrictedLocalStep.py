def createExpandedLPFromZ(
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

        
