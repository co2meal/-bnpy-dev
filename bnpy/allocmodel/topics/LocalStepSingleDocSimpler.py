import numpy as np
from scipy.special import digamma, gammaln

from bnpy.util.OptimizerForPi import estimatePiForDoc_frankwolfe

def estimateNvecForDoc_frankwolfe(wc_d, logLik_d, alphaEbeta):
    '''
    '''
    pass

def calcLocalParamsWithELBOTraceForSingleDoc(
        wc_d=None,
        logLik_d=None,
        alphaEbeta=None,
        initDocTopicProb_d=None,
        initDocTopicCount_d=None,
        convThrLP=0.001,
        nCoordAscentItersLP=100):
    '''

    Returns
    -------
    LP : dict with fields
        * resp
        * DocTopicCount
        * theta
    Ltrace : list
    '''
    K = logLik_d.shape[1]

    explogLik_d = logLik_d.copy()
    maxlogLik_d = np.max(explogLik_d, axis=1)
    explogLik_d -= maxlogLik_d[:,np.newaxis]
    np.exp(explogLik_d, out=explogLik_d)

    if initDocTopicCount_d is not None:
        DocTopicCount_d = initDocTopicCount_d.copy()
        DocTopicProb_d = DocTopicCount_d + alphaEbeta
        digamma(DocTopicProb_d, out=DocTopicProb_d)
        np.exp(DocTopicProb_d, out=DocTopicProb_d)
    elif initDocTopicProb_d is not None:
        DocTopicCount_d = np.zeros(K)
        DocTopicProb_d = initDocTopicProb_d.copy()        
    else:
        # Default initialization!
        DocTopicCount_d = np.zeros(K)
        DocTopicProb_d = alphaEbeta.copy()


    # Initialize sumResp_d
    sumResp_d = np.zeros(logLik_d.shape[0])      
    np.dot(explogLik_d, DocTopicProb_d, out=sumResp_d)

    Ltrace = np.zeros(nCoordAscentItersLP)
    prevDocTopicCount_d = DocTopicCount_d.copy()
    for riter in xrange(nCoordAscentItersLP):
        # # Update DocTopicCount_d
        np.dot(wc_d / sumResp_d, explogLik_d, 
               out=DocTopicCount_d)
        DocTopicCount_d *= DocTopicProb_d
        # # Update DocTopicProb_d
        np.add(DocTopicCount_d, alphaEbeta, 
            out=DocTopicProb_d)
        digamma(DocTopicProb_d, out=DocTopicProb_d)
        np.exp(DocTopicProb_d, out=DocTopicProb_d)
        # # Update sumResp_d
        np.dot(explogLik_d, DocTopicProb_d, out=sumResp_d)
        # # Compute ELBO
        Ltrace[riter] = calcELBOForSingleDocFromCountVec(
            DocTopicCount_d=DocTopicCount_d,
            wc_d=wc_d,
            sumResp_d=sumResp_d,
            alphaEbeta=alphaEbeta)
        # # Check for convergence
        maxDiff_d = np.max(np.abs(
            prevDocTopicCount_d - DocTopicCount_d))
        if maxDiff_d < convThrLP:
            break
        # Track previous DocTopicCount
        prevDocTopicCount_d[:] = DocTopicCount_d

    Ltrace[riter+1:] = Ltrace[riter]
    Ltrace += np.inner(wc_d, maxlogLik_d)
    return DocTopicCount_d, Ltrace

def calcELBOForSingleDocFromCountVec(
        DocTopicCount_d=None, 
        wc_d=None,
        logLik_d=None,
        sumResp_d=None,
        alphaEbeta=None,
        L_max=0.0):
    ''' Compute ELBO for single doc as function of doc-topic counts.

    Returns
    -------
    L : scalar float
        equals ELBO as function of local parameters of single document
        up to an additive constant independent of DocTopicCount_d
    '''
    theta_d = DocTopicCount_d + alphaEbeta
    logPrior_d = digamma(theta_d)
    L_theta = np.sum(gammaln(theta_d)) - np.inner(DocTopicCount_d, logPrior_d)
    explogPrior_d = np.exp(logPrior_d)
    if sumResp_d is None:
        maxlogLik_d = np.max(logLik_d, axis=1)
        explogLik_d = logLik_d - maxlogLik_d[:,np.newaxis]
        np.exp(explogLik_d, out=explogLik_d)
        sumResp_d = np.dot(explogLik_d, explogPrior_d)
        L_max = np.inner(wc_d, maxlogLik_d)
    L_resp = np.inner(wc_d, np.log(sumResp_d))
    return L_theta + L_resp + L_max



def calcLossFuncForInterpolatedDocTopicCount(
        DTC_A, DTC_B, wc_d, logLik_d=None, alphaEbeta=None,
        nGrid=100):        
    wgrid = np.linspace(0, 1.0, nGrid)
    fgrid = np.zeros(nGrid)
    for ii in range(nGrid):
        DTC_ii = wgrid[ii] * DTC_B + (1.0 - wgrid[ii]) * DTC_A
        fgrid[ii] = calcELBOForSingleDocFromCountVec(
            DTC_ii,
            wc_d,
            logLik_d=logLik_d,
            alphaEbeta=alphaEbeta,
            )
    return fgrid, wgrid

def isMonotonicIncreasing(Ltrace):
    return np.diff(Ltrace).min() > -1e-8

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--initname", type=str, default="truelabels")
    args = parser.parse_args()

    import bnpy
    import BarsK10V900

    pylab = bnpy.viz.PlotUtil.pylab

    Data = BarsK10V900.get_data(nDocTotal=50, nWordsPerDoc=500)

    nCoordAscentItersLP = 100
    convThrLP = .00001
    hmodel, Info = bnpy.run(
        Data, 'HDPTopicModel', 'Mult', 'VB',
        initname=args.initname,
        K=args.K,
        gamma=10.0,
        alpha=0.5,
        nLap=1,
        initDocTopicCount_d='setDocProbsToEGlobalProbs',
        restartLP=1,
        nCoordAscentItersLP=nCoordAscentItersLP,
        convThrLP=convThrLP)

    pylab.figure(figsize=(10,4));
    alphaEbeta = hmodel.allocModel.alpha_E_beta()
    alpha = hmodel.allocModel.alpha
    obsLP = hmodel.obsModel.calc_local_params(Data, None)
    topics = hmodel.obsModel.getTopics()

    for d in xrange(Data.nDoc):
        start = Data.doc_range[d]
        stop = Data.doc_range[d+1]
        logLik_d = obsLP['E_log_soft_ev'][start:stop].copy()
        wids_d = Data.word_id[start:stop].copy()
        wc_d = Data.word_count[start:stop].copy()

        pylab.clf();
        ax = pylab.subplot(1, 2, 1);

        bestL = -np.inf
        worstL = +np.inf 
        PRNG = np.random.RandomState(101 * d + 1)
        for randiter in range(50):
            randProb_d = PRNG.rand(alphaEbeta.size)
            randProb_d /= randProb_d.sum()
            #randProb_d = 1.0 + PRNG.dirichlet(alphaEbeta)
            randDTC_d, randLtrace = calcLocalParamsWithELBOTraceForSingleDoc(
                initDocTopicProb_d=randProb_d,
                logLik_d=logLik_d,
                wc_d=wc_d,
                alphaEbeta=alphaEbeta,
                convThrLP=convThrLP,
                nCoordAscentItersLP=nCoordAscentItersLP)
            pylab.plot(randLtrace, 'r--')
            if randLtrace[-1] > bestL:
                bestL = randLtrace[-1]
                bestDTC_d = randDTC_d
            if randLtrace[-1] < worstL:
                worstL = randLtrace[-1]
                worstDTC_d = randDTC_d
            assert isMonotonicIncreasing(randLtrace)

        pi_fw, _, _ = estimatePiForDoc_frankwolfe(
            wids_d, wc_d, topics, alpha)
        fwDTC_d, fwLtrace = calcLocalParamsWithELBOTraceForSingleDoc(
            initDocTopicProb_d=pi_fw,
            logLik_d=logLik_d,
            wc_d=wc_d,
            alphaEbeta=alphaEbeta,
            convThrLP=convThrLP,
            nCoordAscentItersLP=nCoordAscentItersLP)
        pylab.plot(fwLtrace, 'b-', linewidth=2);

        DTC_d, Ltrace = calcLocalParamsWithELBOTraceForSingleDoc(
            initDocTopicCount_d=pi_fw * np.sum(wc_d),
            logLik_d=logLik_d,
            wc_d=wc_d,
            alphaEbeta=alphaEbeta,
            convThrLP=convThrLP,
            nCoordAscentItersLP=nCoordAscentItersLP)


        # From prior
        DTC_d, Ltrace = calcLocalParamsWithELBOTraceForSingleDoc(
            logLik_d=logLik_d,
            wc_d=wc_d,
            alphaEbeta=alphaEbeta,
            convThrLP=convThrLP,
            nCoordAscentItersLP=nCoordAscentItersLP)
        pylab.plot(Ltrace, 'k-', linewidth=2);
        pylab.show(block=False)
        assert isMonotonicIncreasing(Ltrace)
        print "initProb_d = prior"
        print ' '.join(['%6.1f' % (x) for x in DTC_d])
        print "initProb_d = frankwolfe"
        print ' '.join(['%6.1f' % (x) for x in fwDTC_d])

        print "BEST"
        print ' '.join(['%6.1f' % (x) for x in bestDTC_d])
        print "WORST"
        print ' '.join(['%6.1f' % (x) for x in worstDTC_d])

        explogLik_d = np.exp(logLik_d)
        fgrid, wgrid = calcLossFuncForInterpolatedDocTopicCount(
            worstDTC_d, bestDTC_d,
            wc_d,
            logLik_d,
            alphaEbeta)
        pylab.subplot(1,2,2, sharey=ax);
        pylab.plot(wgrid, fgrid, 'b.-')

        ymax = np.maximum(bestL, Ltrace[-1])
        ymin = np.minimum(worstL, fgrid.min())
        yspan = np.maximum(ymax - ymin, 10)
        pylab.ylim(
            ymin=ymin - 0.25 * yspan,
            ymax=ymax + 0.1 * yspan)
        pylab.draw();
        pylab.show(block=False)

        keypress = raw_input("Press any key for next plot >>>")
        if keypress.count("embed"):
            from IPython import embed; embed()
        elif keypress.count("exit"):
            break
        if (d + 1) % 25 == 0:
            print "%3d/%d docs done" % (d+1, Data.nDoc)