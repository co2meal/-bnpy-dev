import argparse
import time
import os
import numpy as np
import scipy.io
import sklearn.metrics
import bnpy
import glob

from scipy.special import digamma
from scipy.misc import logsumexp
from bnpy.allocmodel.topics.LocalStepSingleDoc import calcLocalParams_SingleDoc
from bnpy.ioutil.ModelReader import \
    getPrefixForLapQuery, loadTopicModel, loadModelForLap
from bnpy.ioutil.DataReader import \
    loadDataFromSavedTask, loadLPKwargsFromDisk, loadDataKwargsFromDisk
from bnpy.ioutil.DataReader import str2numorstr

VERSION = 0.1

def evalTopicModelOnTestDataFromTaskpath(
        taskpath='', 
        queryLap=0,
        seed=42,
        dataSplitName='test',
        fracHeldout=0.2,
        printFunc=None,
        **kwargs):
    ''' Evaluate trained topic model saved in specified task on test data
    '''
    # Load saved kwargs for local step
    LPkwargs = loadLPKwargsFromDisk(taskpath)
    for key in kwargs:
        if key in LPkwargs and kwargs[key] is not None:
            LPkwargs[key] = str2val(kwargs[key])
    # Force to be 0, which gives better performance
    # (due to mismatch in objectives)
    if 'restartLP' in LPkwargs:
        LPkwargs['restartLP'] = 0

    # Load test dataset
    Data = loadDataFromSavedTask(taskpath, dataSplitName=dataSplitName)
    DataKwargs = loadDataKwargsFromDisk(taskpath)

    # Check if info is stored in topic-model form
    topicFileList = glob.glob(os.path.join(taskpath, 'Lap*TopicModel.mat'))
    if len(topicFileList) > 0:
        topics, probs, alpha = loadTopicModel(
            taskpath, queryLap=queryLap,
            returnTPA=1, normalizeTopics=1, normalizeProbs=1)
        K = probs.size
    else:
        hmodel, foundLap = loadModelForLap(taskpath, queryLap)
        if hasattr(Data, 'word_count'):
            # Convert to topics 2D array (K x V)
            topics = hmodel.obsModel.getTopics()
            probs = hmodel.allocModel.get_active_comp_probs()
        assert np.allclose(foundLap, queryLap)
        if hasattr(hmodel.allocModel, 'alpha'):
            alpha = hmodel.allocModel.alpha
        elif 'alpha' in DataKwargs:
            alpha = float(DataKwargs['alpha'])
        else:
            alpha = 0.5
        K = hmodel.allocModel.K
    # Prepare debugging statements
    if printFunc: 
        startmsg = "Heldout Metrics at lap %.3f" % (queryLap) 
        filler = '=' * (80 - len(startmsg))
        printFunc(startmsg + ' ' + filler)
        if hasattr(Data, 'word_count'):
            nAtom = Data.word_count.sum()
        else:
            nAtom = Data.nObs
        msg = "%s heldout data. %d documents. %d total words." % (
            Data.name, Data.nDoc, nAtom)
        printFunc(msg)
        printFunc("Using trained model from lap %7.3f with %d topics" % (
            queryLap, K))
        printFunc("Using alpha=%.3f for heldout inference." % (alpha))
        printFunc("Local step params:")
        for key in ['nCoordAscentItersLP', 'convThrLP', 'restartLP']:
            printFunc("    %s: %s" % (key, str(LPkwargs[key])))
        msg = "Splitting each doc" + \
            " into %3.0f%% train and %3.0f%% test, with seed %d" % (
            100*(1-fracHeldout), 100*fracHeldout, seed)
        printFunc(msg)

    # Preallocate storage for metrics
    logpTokensPerDoc = np.zeros(Data.nDoc)
    nTokensPerDoc = np.zeros(Data.nDoc, dtype=np.int32)
    if hasattr(Data, 'word_count'):
        aucPerDoc = np.zeros(Data.nDoc)
        RprecisionPerDoc = np.zeros(Data.nDoc)
    stime = time.time()
    for d in range(Data.nDoc):
        Data_d = Data.select_subset_by_mask([d], doTrackFullSize=0)
        if hasattr(Data, 'word_count'):
            Info_d = calcPredLikForDoc(
                Data_d, topics, probs, alpha,
                fracHeldout=fracHeldout,
                seed=seed + d,
                LPkwargs=LPkwargs)
            logpTokensPerDoc[d] = Info_d['sumlogProbTokens']
            nTokensPerDoc[d] = Info_d['nHeldoutToken']
            aucPerDoc[d] = Info_d['auc']
            RprecisionPerDoc[d] = Info_d['R_precision']
            avgAUCscore = np.mean(aucPerDoc[:d+1])
            avgRscore = np.mean(RprecisionPerDoc[:d+1])
            scoreMsg = "avgLik %.4f avgAUC %.4f avgRPrec %.4f" % (
                np.sum(logpTokensPerDoc[:d+1]) / np.sum(nTokensPerDoc[:d+1]),
                avgAUCscore, avgRscore)
            SVars = dict(
                avgRPrecScore=avgRscore,
                avgAUCScore=avgAUCscore,
                avgAUCScorePerDoc=aucPerDoc,
                avgRPrecScorePerDoc=RprecisionPerDoc)
        else:
            Info_d = calcPredLikForDocFromHModel(
                Data_d, hmodel,
                alpha=alpha,
                fracHeldout=fracHeldout,
                seed=seed + d,
                LPkwargs=LPkwargs)
            logpTokensPerDoc[d] = Info_d['sumlogProbTokens']
            nTokensPerDoc[d] = Info_d['nHeldoutToken']
            scoreMsg = "avgLik %.4f" % (
                np.sum(logpTokensPerDoc[:d+1]) / np.sum(nTokensPerDoc[:d+1]),
                )
            SVars = dict()

        if d == 0 or (d+1) % 25 == 0 or d == Data.nDoc - 1:
            if printFunc:
                etime = time.time() - stime
                msg = "%5d/%d after %8.1f sec " % (d+1, Data.nDoc, etime) 
                printFunc(msg + scoreMsg)
    # Aggregate results
    meanlogpTokensPerDoc = np.sum(logpTokensPerDoc) / np.sum(nTokensPerDoc)
    # Compute heldout Lscore
    if not hasattr(Data, 'word_count'):
        if hasattr(hmodel.allocModel, 'gamma'):
            gamma = hmodel.allocModel.gamma
        else:
            gamma = hmodel.allocModel.gamma0
        aParams = dict(gamma=gamma, alpha=alpha)
        oParams = hmodel.obsModel.get_prior_dict()
        del oParams['inferType']

        # Create DP mixture model from current hmodel
        DPmodel = bnpy.HModel.CreateEntireModel('VB', 'DPMixtureModel',
            hmodel.getObsModelName(),
            aParams, oParams,
            Data)
        DPmodel.set_global_params(hmodel=hmodel)
        LP = DPmodel.calc_local_params(Data, **LPkwargs)
        SS = DPmodel.get_global_suff_stats(Data, LP, doPrecompEntropy=1)
        dpLscore = DPmodel.calc_evidence(SS=SS)

        # Create HDP topic model from current hmodel
        HDPmodel = bnpy.HModel.CreateEntireModel('VB', 'HDPTopicModel',
            hmodel.getObsModelName(),
            aParams, oParams,
            Data)
        HDPmodel.set_global_params(hmodel=hmodel)
        LP = HDPmodel.calc_local_params(Data, **LPkwargs)
        SS = HDPmodel.get_global_suff_stats(Data, LP, doPrecompEntropy=1)
        hdpLscore = HDPmodel.calc_evidence(SS=SS)

        SVars['dpLscore'] = dpLscore
        SVars['hdpLscore'] = hdpLscore
        printFunc("~~~ dpL=%.6e\n~~~hdpL=%.6e" % (dpLscore, hdpLscore))
    # Prepare to save results.
    prefix, lap = getPrefixForLapQuery(taskpath, queryLap)
    outmatfile = os.path.join(taskpath, prefix + "PredLik.mat")
    # Collect all quantities to save into giant dict.
    SaveVars = dict(
        version=VERSION,
        outmatfile=outmatfile,
        fracHeldout=fracHeldout,
        predLLPerDoc=logpTokensPerDoc,
        avgPredLL=np.sum(logpTokensPerDoc) / np.sum(nTokensPerDoc),
        K=K,
        nTokensPerDoc=nTokensPerDoc,
        **LPkwargs)
    SaveVars.update(SVars)
    scipy.io.savemat(outmatfile, SaveVars, oned_as='row')
    SVars['avgLikScore'] = SaveVars['avgPredLL']
    SVars['lapTrain'] = queryLap
    SVars['K'] = K
    for key in SVars:
        if key.endswith('PerDoc'):
            continue
        outtxtfile = os.path.join(taskpath, 'predlik-%s.txt' % (key))
        with open(outtxtfile, 'a') as f:
            f.write("%.6e\n" % (SVars[key]))
    if printFunc:
        printFunc("DONE with heldout inference at lap %.3f" % queryLap)
        printFunc("Wrote per-doc results in MAT file:" + 
            outmatfile.split(os.path.sep)[-1])
        printFunc("      Aggregate results in txt files: predlik-__.txt")


    # Write the summary message
    if printFunc:
        etime = time.time() - stime
        curLapStr = '%7.3f' % (queryLap)
        nLapStr = '%d' % (kwargs['learnAlg'].algParams['nLap'])
        logmsg = '  %s/%s heldout metrics   | K %4d | %s'
        logmsg = logmsg % (curLapStr, nLapStr, K, scoreMsg) 
        printFunc(logmsg, 'info')

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
    PRNG = np.random.RandomState(int(seed))
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



def calcPredLikForDocFromHModel(
        docData, hmodel,
        fracHeldout=0.2,
        seed=42,
        MINSIZE=10,
        LPkwargs=dict(),
        alpha=None,
        **kwargs):
    ''' Calculate predictive likelihood for single doc under given model.

    Returns
    -------
    '''
    Info = dict()
    assert docData.nDoc == 1

    # Split document into training and heldout
    # assigning each unique vocab type to one or the other
    if hasattr(docData, 'word_id'):
        N = docData.word_id.size
    else:
        N = docData.nObs
    nHeldout = int(np.ceil(fracHeldout * N))
    nHeldout = np.maximum(MINSIZE, nHeldout)
    PRNG = np.random.RandomState(int(seed))
    shuffleIDs = PRNG.permutation(N)
    heldoutIDs = shuffleIDs[:nHeldout]
    trainIDs = shuffleIDs[nHeldout:]
    if len(heldoutIDs) < MINSIZE:
        raise ValueError('Not enough unique IDs to make good test split')
    if len(trainIDs) < MINSIZE:
        raise ValueError('Not enough unique IDs to make good train split')

    hoData = docData.select_subset_by_mask(atomMask=heldoutIDs)
    trData = docData.select_subset_by_mask(atomMask=trainIDs)

    # Run local step to get DocTopicCounts
    DocTopicCount_d, Info = inferDocTopicCountForDocFromHModel(
        trData, hmodel, **LPkwargs)
    probs = hmodel.allocModel.get_active_comp_probs()
    # Compute expected topic probs in this doc
    theta_d = DocTopicCount_d + alpha * probs
    E_log_pi_d = digamma(theta_d) - digamma(np.sum(theta_d))
    # Evaluate log prob per token metric
    LP = hmodel.obsModel.calc_local_params(hoData)
    logProbArr_d = LP['E_log_soft_ev']
    logProbArr_d += E_log_pi_d[np.newaxis, :]
    logProbPerToken_d = logsumexp(logProbArr_d, axis=1)
    # Pack up and ship
    Info['DocTopicCount'] = DocTopicCount_d
    Info['nHeldoutToken'] = len(heldoutIDs)
    Info['sumlogProbTokens'] = np.sum(logProbPerToken_d)
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

def inferDocTopicCountForDocFromHModel(
        docData, hmodel, alpha=0.5, **LPkwargs):
    # Lik_d : 2D array, size N x K
    # Each row is non-negative
    LP = hmodel.obsModel.calc_local_params(docData)
    Lik_d = LP['E_log_soft_ev']
    Lik_d -= Lik_d.max(axis=1)[:,np.newaxis]
    np.exp(Lik_d, out=Lik_d)

    # alphaEbeta : 1D array, size K
    alphaEbeta = alpha * hmodel.allocModel.get_active_comp_probs()
    DocTopicCount_d, _, _, Info = calcLocalParams_SingleDoc(
        1.0, Lik_d, alphaEbeta,
        alphaEbetaRem=None,
        **LPkwargs)
    assert np.allclose(DocTopicCount_d.sum(), Lik_d.shape[0])
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
