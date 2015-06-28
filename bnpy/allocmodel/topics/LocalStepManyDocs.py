import numpy as np
import copy
from scipy.special import digamma, gammaln

from bnpy.util import NumericUtil
import LocalStepLogger

from LocalStepSingleDoc import calcLocalParams_SingleDoc
from LocalStepSingleDoc import calcLocalParams_SingleDoc_WithELBOTrace


def calcLocalParams(
        Data, LP,
        alphaEbeta=None,
        alphaEbetaRem=None,
        alpha=None,
        initDocTopicCountLP='scratch',
        cslice=(0, None),
        **kwargs):
    ''' Calculate all local parameters for provided dataset under a topic model

    Returns
    -------
    LP : dict of local params, with fields
    * DocTopicCount : 2D array, nDoc x K
    * resp : 2D array, N x K
    * model-specific fields for doc-topic probabilities
    '''
    assert isinstance(cslice, tuple)
    if len(cslice) != 2:
        cslice = (0, None)
    elif cslice[0] is None:
        cslice = (0, None)
    nDoc = calcNumDocFromSlice(Data, cslice)

    # Prepare the likelihood matrix
    # Make sure it is C-contiguous, so that matrix ops are very fast
    Lik = np.asarray(LP['E_log_soft_ev'], order='C')
    Lik -= Lik.max(axis=1)[:, np.newaxis]
    NumericUtil.inplaceExp(Lik)

    # Prepare the initial DocTopicCount matrix,
    # Useful for warm starts of the local step.
    N, K = Lik.shape
    initDocTopicCount = None
    if 'DocTopicCount' in LP:
        if LP['DocTopicCount'].shape == (nDoc, K):
            initDocTopicCount = LP['DocTopicCount'].copy()

    sumRespTilde = np.zeros(N)
    DocTopicCount = np.zeros((nDoc, K))
    DocTopicProb = np.zeros((nDoc, K))

    if alphaEbeta is None:
        assert alpha is not None
        alphaEbeta = alpha * np.ones(K)
    else:
        alphaEbeta = alphaEbeta[:K]

    slice_start = Data.doc_range[cslice[0]]
    AggInfo = dict()
    for d in xrange(nDoc):
        start = Data.doc_range[cslice[0] + d]
        stop = Data.doc_range[cslice[0] + d + 1]

        lstart = start - slice_start
        lstop = stop - slice_start
        Lik_d = Lik[lstart:lstop].copy()  # Local copy
        if hasattr(Data, 'word_count'):
            wc_d = Data.word_count[start:stop].copy()
        else:
            wc_d = 1.0
        if initDocTopicCountLP == 'memo' and initDocTopicCount is not None:
            initDTC_d = initDocTopicCount[d]
        else:
            initDTC_d = None

        DocTopicCount[d], DocTopicProb[d], sumRespTilde[lstart:lstop], Info_d \
            = calcLocalParams_SingleDoc(
                wc_d, Lik_d, alphaEbeta, alphaEbetaRem,
                DocTopicCount_d=initDTC_d,
                **kwargs)
        AggInfo = updateConvergenceInfoForDoc_d(d, Info_d, AggInfo, Data)

    LP['DocTopicCount'] = DocTopicCount
    LP = updateLPGivenDocTopicCount(LP, DocTopicCount,
                                    alphaEbeta, alphaEbetaRem)
    LP = updateLPWithResp(LP, Data, Lik, DocTopicProb, sumRespTilde, cslice)
    LP['Info'] = AggInfo
    writeLogMessageForManyDocs(Data, AggInfo, **kwargs)
    return LP


def updateLPGivenDocTopicCount(LP, DocTopicCount,
                               alphaEbeta, alphaEbetaRem=None):
    ''' Update local parameters given doc-topic counts for many docs.

    Returns for FiniteTopicModel (alphaEbetaRem is None)
    --------
    LP : dict of local params, with updated fields
        * theta : 2D array, nDoc x K
        * ElogPi : 2D array, nDoc x K

    Returns for HDPTopicModel (alphaEbetaRem is not None)
    --------
        * theta : 2D array, nDoc x K
        * ElogPi : 2D array, nDoc x K
        * thetaRem : scalar
        * ElogPiRem : scalar
    '''
    theta = DocTopicCount + alphaEbeta

    if alphaEbetaRem is None:
        # FiniteTopicModel
        digammaSumTheta = digamma(theta.sum(axis=1))
    else:
        # HDPTopicModel
        digammaSumTheta = digamma(theta.sum(axis=1) + alphaEbetaRem)
        LP['thetaRem'] = alphaEbetaRem
        LP['ElogPiRem'] = digamma(alphaEbetaRem) - digammaSumTheta
        LP['digammaSumTheta'] = digammaSumTheta  # Used for merges

    ElogPi = digamma(theta) - digammaSumTheta[:, np.newaxis]
    LP['theta'] = theta
    LP['ElogPi'] = ElogPi
    return LP


def updateLPWithResp(LP, Data, Lik, Prior, sumRespTilde, cslice=(0, None)):
    nDoc = calcNumDocFromSlice(Data, cslice)
    LP['resp'] = Lik.copy()
    slice_start = Data.doc_range[cslice[0]]
    for d in xrange(nDoc):
        start = Data.doc_range[cslice[0] + d] - slice_start
        stop = Data.doc_range[cslice[0] + d + 1] - slice_start
        LP['resp'][start:stop] *= Prior[d]
    LP['resp'] /= sumRespTilde[:, np.newaxis]
    np.maximum(LP['resp'], 1e-300, out=LP['resp'])
    # assert np.allclose(LP['resp'].sum(axis=1), 1.0)
    return LP


def updateSingleDocLPWithResp(LP_d, Lik_d, Prior_d, sumR_d):
    resp_d = Lik_d.copy()
    resp_d *= Prior_d
    resp_d /= sumR_d[:, np.newaxis]
    np.maximum(resp_d, 1e-300, out=resp_d)
    LP_d['resp'] = resp_d
    return LP_d


def calcNumDocFromSlice(Data, cslice):
    if cslice[1] is None:
        nDoc = Data.nDoc
    else:
        nDoc = cslice[1] - cslice[0]
    return int(nDoc)


def writeLogMessageForManyDocs(Data, AI,
                               sliceID=None,
                               **kwargs):
    """ Write log message summarizing convergence behavior across docs.

    Args
    ----
    Data : bnpy DataObj
    AI : dict of aggregated info for all documents.

    Post Condition
    --------------
    Message written to LocalStepLogger.
    """
    if 'lapFrac' not in kwargs:
        return
    if 'batchID' not in kwargs:
        return

    if isinstance(sliceID, int):
        sliceID = '%d' % (sliceID)
    else:
        sliceID = '0'

    perc = [0, 1, 10, 50, 90, 99, 100]
    siter = ' '.join(
        ['%d:%d' % (p, np.percentile(AI['iter'], p)) for p in perc])
    sdiff = ' '.join(
        ['%d:%.4f' % (p, np.percentile(AI['maxDiff'], p)) for p in perc])
    nConverged = np.sum(AI['maxDiff'] <= kwargs['convThrLP'])
    msg = 'lap %4.2f batch %d slice %s' % (
        kwargs['lapFrac'], kwargs['batchID'], sliceID)

    msg += ' nConverged %4d/%d' % (nConverged, AI['maxDiff'].size)
    worstDocID = np.argmax(AI['maxDiff'])
    msg += " worstDocID %4d \n" % (worstDocID)

    msg += ' iter prctiles %s\n' % (siter)
    msg += ' diff prctiles %s\n' % (sdiff)

    if 'nRestartsAccepted' in AI:
        msg += " nRestarts %4d/%4d\n" % (
            AI['nRestartsAccepted'], AI['nRestartsTried'])
    LocalStepLogger.log(msg)


def updateConvergenceInfoForDoc_d(d, Info_d, AggInfo, Data):
    """ Update convergence stats for specific doc into AggInfo.

    Returns
    -------
    AggInfo : dict, updated in place.
        * maxDiff : 1D array, nDoc
        * iter : 1D array, nDoc
    """
    if len(AggInfo.keys()) == 0:
        AggInfo['maxDiff'] = np.zeros(Data.nDoc)
        AggInfo['iter'] = np.zeros(Data.nDoc, dtype=np.int32)

    AggInfo['maxDiff'][d] = Info_d['maxDiff']
    AggInfo['iter'][d] = Info_d['iter']
    if 'ELBOtrace' in Info_d:
        AggInfo['ELBOtrace'] = Info_d['ELBOtrace']
    if 'nAccept' in Info_d:
        if 'nRestartsAccepted' not in AggInfo:
            AggInfo['nRestartsAccepted'] = 0
            AggInfo['nRestartsTried'] = 0
        AggInfo['nRestartsAccepted'] += Info_d['nAccept']
        AggInfo['nRestartsTried'] += Info_d['nTrial']
    return AggInfo
