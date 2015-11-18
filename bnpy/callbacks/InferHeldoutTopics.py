import argparse
import time
import os
import numpy as np
import scipy.io
import sklearn.metrics

from bnpy.allocmodel.topics.LocalStepSingleDoc import calcLocalParams_SingleDoc
from bnpy.ioutil.ModelReader import getPrefixForLapQuery, loadTopicModel
from bnpy.ioutil.DataReader import loadDataFromSavedTask, loadLPKwargsFromDisk
from bnpy.ioutil.DataReader import str2numorstr

VERSION = 0.1

def evalTopicModelOnTestDataFromTaskpath(
        taskpath='', 
        queryLap=0, seed=42,
        dataSplitName='test',
        fracHeldout=0.2,
        printFunc=None,
        **kwargs):
    ''' Evaluate trained topic model saved in specified task on test data
    '''
    # Load test dataset
    Data = loadDataFromSavedTask(taskpath, dataSplitName=dataSplitName)
    # Load saved kwargs for local step
    LPkwargs = loadLPKwargsFromDisk(taskpath)
    for key in kwargs:
        if key in LPkwargs and kwargs[key] is not None:
            LPkwargs[key] = str2val(kwargs[key])
    # Load saved model
    try:
        topics, probs, alpha = loadTopicModel(
            taskpath, queryLap=queryLap,
            returnTPA=1, normalizeTopics=1, normalizeProbs=1)
    except Exception:
        from IPython import embed; embed()
    if printFunc:
        msg = "%s heldout data. %d documents. %d total words." % (
            Data.name, Data.nDoc, Data.word_count.sum())
        printFunc(msg)
        printFunc("Using trained model from lap %7.3f with %d topics" % (
            queryLap, probs.size))
        msg = "Splitting word types in each doc" + \
            " into %3.0f%% train and %3.0f%% test, with seed %d" % (
            100*(1-fracHeldout), 100*fracHeldout, seed)
        printFunc(msg)
    logpTokensPerDoc = np.zeros(Data.nDoc)
    aucPerDoc = np.zeros(Data.nDoc)
    RprecisionPerDoc = np.zeros(Data.nDoc)
    nTokensPerDoc = np.zeros(Data.nDoc, dtype=np.int32)
    stime = time.time()
    for d in range(Data.nDoc):
        Data_d = Data.select_subset_by_mask([d], doTrackFullSize=0)
        Info_d = calcPredLikForDoc(
            Data_d, topics, probs, alpha,
            fracHeldout=fracHeldout,
            seed=seed,
            LPkwargs=LPkwargs)
        logpTokensPerDoc[d] = Info_d['sumlogProbTokens']
        nTokensPerDoc[d] = Info_d['nHeldoutToken']
        aucPerDoc[d] = Info_d['auc']
        RprecisionPerDoc[d] = Info_d['R_precision']
        if d == 0 or (d+1) % 25 == 0 or d == Data.nDoc - 1:
            meanScore = np.sum(logpTokensPerDoc[:d+1]) / \
                np.sum(nTokensPerDoc[:d+1])
            meanAUC = np.mean(aucPerDoc[:d+1])
            meanRPrec = np.mean(RprecisionPerDoc[:d+1])
            if printFunc:
                etime = time.time() - stime
                msg = "%5d/%d after %8.1f sec " + \
                    "avglogpWord %.4f avgauc %.4f avgRprec %.4f"
                msg = msg % (d+1, Data.nDoc, etime, 
                    meanScore, meanAUC, meanRPrec)
                printFunc(msg)
    meanlogpTokensPerDoc = np.sum(logpTokensPerDoc) / np.sum(nTokensPerDoc)
    # Prepare to save results.
    prefix, lap = getPrefixForLapQuery(taskpath, queryLap)
    outmatfile = os.path.join(taskpath, prefix + "PredLik.mat")
    # Collect all quantities to save into giant dict.
    SaveVars = dict(
        version=VERSION,
        outmatfile=outmatfile,
        fracHeldout=fracHeldout,
        predLLPerDoc=logpTokensPerDoc,
        avgPredLL=meanlogpTokensPerDoc,
        K=probs.size,
        aucPerDoc=aucPerDoc,
        nTokensPerDoc=nTokensPerDoc,
        avgAUCScore=np.mean(aucPerDoc),
        avgRPrecision=np.mean(RprecisionPerDoc),
        **LPkwargs)
    scipy.io.savemat(outmatfile, SaveVars, oned_as='row')
    if printFunc:
        printFunc("DONE. Results written to MAT file:\n" + outmatfile)
    return SaveVars


def calcPredLikForDoc(docData, topics, probs, alpha,
                      fracHeldout=0.2,
                      seed=42, MINSIZE=10, LPkwargs=dict(), **kwargs):
    ''' Calculate predictive likelihood for single doc under given model.

    Returns
    -------
    '''
    Info = dict()
    assert docData.nDoc == 1

    # Split document into training and heldout
    # assigning each unique vocab type to one or the other
    nUnique = docData.word_id.size
    nHeldout = int(np.ceil(fracHeldout * nUnique))
    nHeldout = np.maximum(MINSIZE, nHeldout)
    PRNG = np.random.RandomState(seed)
    shuffleIDs = PRNG.permutation(nUnique)
    heldoutIDs = shuffleIDs[:nHeldout]
    trainIDs = shuffleIDs[nHeldout:]
    if len(heldoutIDs) < MINSIZE:
        raise ValueError('Not enough unique IDs to make good test split')
    if len(trainIDs) < MINSIZE:
        raise ValueError('Not enough unique IDs to make good train split')
    ho_word_id = docData.word_id[heldoutIDs]
    ho_word_ct = docData.word_count[heldoutIDs]
    tr_word_id = docData.word_id[trainIDs]
    tr_word_ct = docData.word_count[trainIDs]
    # Run local step to get DocTopicCounts
    DocTopicCount_d, Info = inferDocTopicCountForDoc(
        tr_word_id, tr_word_ct, topics, probs, alpha, **LPkwargs)
    # Compute expected topic probs in this doc
    theta_d = DocTopicCount_d + alpha * probs
    Epi_d = theta_d / np.sum(theta_d)
    # Evaluate log prob per token metric
    probPerToken_d = np.dot(topics[:, ho_word_id].T, Epi_d)
    logProbPerToken_d = np.log(probPerToken_d)
    sumlogProbTokens_d = np.sum(logProbPerToken_d * ho_word_ct)
    nHeldoutToken_d = np.sum(ho_word_ct)
    # # Evaluate retrieval metrics

    # unseen_mask_d : 1D array, size vocab_size
    #   entry is 0 if word is seen in training half
    #   entry is 1 if word is unseen 
    unseen_mask_d = np.ones(docData.vocab_size, dtype=np.bool8)
    unseen_mask_d[tr_word_id] = 0
    probOfUnseenTypes_d = np.dot(topics[:, unseen_mask_d].T, Epi_d)
    unseen_mask_d = np.asarray(unseen_mask_d, dtype=np.int32)
    unseen_mask_d[ho_word_id] = 2
    trueLabelsOfUnseenTypes_d = unseen_mask_d[unseen_mask_d > 0]     
    trueLabelsOfUnseenTypes_d -= 1
    assert np.sum(trueLabelsOfUnseenTypes_d) == ho_word_id.size
    fpr, tpr, thr = sklearn.metrics.roc_curve(
        trueLabelsOfUnseenTypes_d, probOfUnseenTypes_d)
    auc = sklearn.metrics.auc(fpr, tpr)
    # top R precision, where R = total num positive instances
    topR = ho_word_id.size
    topRUnseenTypeIDs = np.argsort(-1 * probOfUnseenTypes_d)[:topR]
    R_precision = sklearn.metrics.precision_score(
        trueLabelsOfUnseenTypes_d[topRUnseenTypeIDs],
        np.ones(topR))

    if R_precision > 0.9:
        print R_precision
    # Useful debugging
    # >>> unseenTypeIDs = np.flatnonzero(unseen_mask_d)
    # >>> trainIm = np.zeros(900); trainIm[tr_word_id] = 1.0
    # >>> testIm = np.zeros(900); testIm[ho_word_id] = 1.0
    # >>> predictIm = np.zeros(900);
    # >>> predictIm[unseenTypeIDs[topRUnseenTypeIDs]] = 1;
    # >>> bnpy.viz.BarsViz.showTopicsAsSquareImages( np.vstack([trainIm, testIm, predictIm]) )
    Info['auc'] = auc
    Info['R_precision'] = R_precision
    Info['ho_word_ct'] = ho_word_ct
    Info['tr_word_ct'] = tr_word_ct
    Info['DocTopicCount'] = DocTopicCount_d
    Info['nHeldoutToken'] = nHeldoutToken_d
    Info['sumlogProbTokens'] = sumlogProbTokens_d
    return Info

def inferDocTopicCountForDoc(
        word_id, word_ct, topics, probs, alpha, 
        **LPkwargs):
    K = probs.size
    K2, W = topics.shape
    assert K == K2
    # topics : 2D array, vocab_size x K
    # Each col is non-negative and sums to one.
    topics = topics.T.copy()
    assert np.allclose(np.sum(topics, axis=0), 1.0)
    # Lik_d : 2D array, size N x K
    # Each row is non-negative
    Lik_d = np.asarray(topics[word_id, :].copy(), dtype=np.float64)
    # alphaEbeta : 1D array, size K
    alphaEbeta = np.asarray(alpha * probs, dtype=np.float64)

    DocTopicCount_d, _, _, Info = calcLocalParams_SingleDoc(
        word_ct, Lik_d, alphaEbeta,
        alphaEbetaRem=None,
        **LPkwargs)
    assert np.allclose(DocTopicCount_d.sum(), word_ct.sum())
    return DocTopicCount_d, Info

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('taskpath', type=str)
    parser.add_argument('--queryLap', type=float, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--restartLP', type=int, default=None)
    parser.add_argument('--fracHeldout', type=float, default=0.2)
    args = parser.parse_args()

    evalTopicModelOnTestDataFromTaskpath(**args.__dict__)
