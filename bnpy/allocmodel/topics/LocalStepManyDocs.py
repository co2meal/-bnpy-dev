import numpy as np
import copy
from scipy.special import digamma, gammaln

from bnpy.util import NumericUtil
import LocalStepLogger

from LocalStepSingleDoc import calcLocalParams_SingleDoc
from LocalStepSingleDoc import calcLocalParams_SingleDoc_WithELBOTrace


def calcLocalParamsForDataSlice(
        Data, LP, aModel,
        initDocTopicCountLP='scratch',
        **kwargs):
    ''' Calculate all local parameters for provided dataset under a topic model

    Returns
    -------
    LP : dict of local params, with fields
    * DocTopicCount : 2D array, nDoc x K
    * resp : 2D array, N x K
    * model-specific fields for doc-topic probabilities
    '''
    # Prepare the likelihood matrix
    # Make sure it is C-contiguous, so that matrix ops are very fast
    Lik = np.asarray(LP['E_log_soft_ev'], order='C')
    Lik -= Lik.max(axis=1)[:, np.newaxis]
    NumericUtil.inplaceExp(Lik)

    # Prepare the initial DocTopicCount matrix,
    # Useful for warm starts of the local step.
    K = Lik.shape[1]
    initDocTopicCount = None
    if 'DocTopicCount' in LP:
        if LP['DocTopicCount'].shape == (Data.nDoc, K):
            initDocTopicCount = LP['DocTopicCount'].copy()

    N, K = Lik.shape
    sumRespTilde = np.zeros(N)
    DocTopicCount = np.zeros((Data.nDoc, K))
    DocTopicProb = np.zeros((Data.nDoc, K))

    if str(type(aModel)).count('HDP'):
        alphaEbeta = aModel.alphaEbeta[:-1].copy()
        alphaEbetaRem = aModel.alphaEbeta[-1] * 1.0  # to float
    else:
        # FiniteTopicModel
        alphaEbeta = aModel.alpha * np.ones(K)
        alphaEbetaRem = None

    AggInfo = dict()
    for d in xrange(Data.nDoc):
        start = Data.doc_range[d]
        stop = Data.doc_range[d + 1]
        Lik_d = Lik[start:stop].copy()  # Local copy
        if hasattr(Data, 'word_count'):
            wc_d = Data.word_count[start:stop].copy()
        else:
            wc_d = 1.0
        if initDocTopicCountLP == 'memo' and initDocTopicCount is not None:
            initDTC_d = initDocTopicCount[d]
        else:
            initDTC_d = None

        DocTopicCount[d], DocTopicProb[d], sumRespTilde[start:stop], Info_d \
            = calcLocalParams_SingleDoc(
                wc_d, Lik_d, alphaEbeta, alphaEbetaRem,
                DocTopicCount_d=initDTC_d,
                **kwargs)
        AggInfo = updateConvergenceInfoForDoc_d(d, Info_d, AggInfo, Data)

    LP['DocTopicCount'] = DocTopicCount
    LP = aModel.updateLPGivenDocTopicCount(LP, DocTopicCount)
    LP = updateLPWithResp(LP, Data, Lik, DocTopicProb, sumRespTilde)
    LP['Info'] = AggInfo
    writeLogMessageForManyDocs(Data, AggInfo, **kwargs)
    return LP


def updateLPWithResp(LP, Data, Lik, Prior, sumRespTilde):
    LP['resp'] = Lik.copy()
    for d in xrange(Data.nDoc):
        start = Data.doc_range[d]
        stop = Data.doc_range[d + 1]
        LP['resp'][start:stop] *= Prior[d]
    LP['resp'] /= sumRespTilde[:, np.newaxis]
    np.maximum(LP['resp'], 1e-300, out=LP['resp'])
    return LP


def updateSingleDocLPWithResp(LP_d, Lik_d, Prior_d, sumR_d):
    resp_d = Lik_d.copy()
    resp_d *= Prior_d
    resp_d /= sumR_d[:, np.newaxis]
    np.maximum(resp_d, 1e-300, out=resp_d)
    LP_d['resp'] = resp_d
    return LP_d


def writeLogMessageForManyDocs(Data, AI, **kwargs):
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

    perc = [0, 5, 10, 50, 90, 95, 100]
    siter = ' '.join(
        ['%4d' % np.percentile(AI['iter'], p) for p in perc])
    sdiff = ['%6.4f' % np.percentile(AI['maxDiff'], p) for p in perc]
    sdiff = ' '.join(sdiff)
    nFail = np.sum(AI['maxDiff'] > kwargs['convThrLP'])
    msg = '%4.2f %3d %4d %s %s' % (kwargs['lapFrac'], kwargs['batchID'],
                                   nFail, siter, sdiff)
    worstDocID = np.argmax(AI['maxDiff'])
    msg += " %4d" % (worstDocID)
    if 'nRestartsAccepted' in AI:
        msg += " %4d/%4d" % (AI['nRestartsAccepted'],
                             AI['nRestartsTried'])
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
