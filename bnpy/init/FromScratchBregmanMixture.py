import numpy as np
import bnpy.data
from bnpy.util.OptimizerForPi import \
    estimatePiForDoc_frankwolfe, \
    estimatePiForDoc_graddescent, \
    pi2str
from FromTruth import \
    convertLPFromHardToSoft, \
    convertLPFromTokensToDocs, \
    convertLPFromDocsToTokens, \
    convertLPFromDocsToTypes
from FromScratchBregman import makeDataSubsetByThresholdResp

def init_global_params(hmodel, Data, **kwargs):
    ''' Initialize parameters of observation model.
    
    Post Condition
    --------------
    hmodel internal parameters updated to reflect sufficient statistics.
    '''
    if kwargs['initname'].lower().count('priormean'):
        kwargs['init_setOneToPriorMean'] = 1

    if kwargs['initname'].count('+'):
        kwargs['init_NiterForBregmanKMeans'] = \
            int(kwargs['initname'].split('+')[1])
        if 'logFunc' not in kwargs:
            def logFunc(msg):
                print msg
            kwargs['logFunc'] = logFunc
    # Obtain initial suff statistics
    SS, Info = initSS_BregmanMixture(
        Data, hmodel, includeAllocSummary=True, **kwargs)
    # Execute global step from these stats
    hmodel.obsModel.update_global_params(SS)
    Info['targetSS'] = SS
    if kwargs['init_NiterForBregmanKMeans'] > 0:
        hmodel.allocModel.update_global_params(SS)
    else:
        hmodel.allocModel.init_global_params(Data, **kwargs)    
    return Info


def initSS_BregmanMixture(
        Dslice=None, 
        curModel=None, 
        curLPslice=None,
        K=5, 
        ktarget=None, 
        seed=0,
        includeAllocSummary=False,
        NiterForBregmanKMeans=1,
        logFunc=None,
        setOneToPriorMean=0,
        **kwargs):
    ''' Create observation model statistics via Breg. distance sampling.

    Args
    ------
    Data

    Returns
    -------
    xSS : SuffStatBag
    DebugInfo : dict
        contains info about provenance of this initialization.
    '''
    # Reformat any keyword argument to drop 
    # prefix of 'b_' or 'init_'
    for key, val in kwargs.items():
        if key.startswith('b_'):
            newkey = key[2:]
            kwargs[newkey] = val
            del kwargs[key]
        elif key.startswith('init_'):
            newkey = key[5:]
            kwargs[newkey] = val
            del kwargs[key]
    if 'setOneToPriorMean' in kwargs:
        setOneToPriorMean = kwargs['setOneToPriorMean']
    Niter = np.maximum(NiterForBregmanKMeans, 0)

    if logFunc:
        logFunc("Preparing target dataset for Bregman mixture analysis...")
    DebugInfo, targetData, targetX, targetW, chosenRespIDs = \
        makeDataSubsetByThresholdResp(
            Dslice,
            curModel,
            curLPslice,
            ktarget,
            K=K,
            **kwargs)
    if logFunc:
        logFunc(DebugInfo['targetAssemblyMsg'])
    if targetData is None:
        assert 'errorMsg' in DebugInfo
        return None, DebugInfo
    K = np.minimum(K, targetX.shape[0])
    if logFunc:
        msg = "Running Bregman mixture with K=%d for %d iters" % (
            K, Niter)
        if setOneToPriorMean:
            msg += ", with initial prior mean cluster"
        logFunc(msg)

    # Perform plusplus initialization + Kmeans clustering
    targetZ, Mu, minDiv, sumDataTerm, Lscores = initKMeans_BregmanMixture(
        targetData, K, curModel.obsModel,
        seed=seed)
    # Convert labels in Z to compactly use all ints from 0, 1, ... Kused
    # Then translate these into a proper 'resp' 2D array,
    # where resp[n,k] = w[k] if z[n] = k, and 0 otherwise
    xtargetLP, targetZ = convertLPFromHardToSoft(
        dict(Z=targetZ), targetData, initGarbageState=0, returnZ=1)
    if isinstance(Dslice, bnpy.data.WordsData):
        if curModel.obsModel.DataAtomType.count('word'):
            if curModel.getObsModelName().count('Bern'):
                xtargetLP = convertLPFromDocsToTypes(xtargetLP, targetData)
            else:
                xtargetLP = convertLPFromDocsToTokens(xtargetLP, targetData)
    # Summarize the local parameters
    if includeAllocSummary and Niter > 0:
        if hasattr(curModel.allocModel, 'initLPFromResp'):
            xtargetLP = curModel.allocModel.initLPFromResp(
                targetData, xtargetLP)
        xSS = curModel.get_global_suff_stats(
            targetData, xtargetLP)
    else:
        xSS = curModel.obsModel.get_global_suff_stats(
            targetData, None, xtargetLP)
        if setOneToPriorMean:
            neworder = np.hstack([xSS.K, np.arange(xSS.K)])
            xSS.insertEmptyComps(1)
            xSS.reorderComps(neworder)
        else:
            assert np.allclose(np.unique(targetZ), np.arange(xSS.K))
        assert np.allclose(len(Mu), xSS.K)
    # Reorder the components from big to small
    oldids_bigtosmall = np.argsort(-1 * xSS.getCountVec())
    xSS.reorderComps(oldids_bigtosmall)
    # Be sure to account for the sorting that just happened.
    # By fixing up the cluster means Mu and assignments Z
    Mu = [Mu[k] for k in oldids_bigtosmall] 
    neworder = np.arange(xSS.K)    
    old2newID=dict(zip(oldids_bigtosmall, neworder))
    targetZnew = -1 * np.ones_like(targetZ)
    for oldk in xrange(xSS.K):
        old_mask = targetZ == oldk
        targetZnew[old_mask] = old2newID[oldk]
    assert np.all(targetZnew >= 0)
    assert np.allclose(len(Mu), xSS.K)
    if logFunc:
        logFunc('Bregman k-means DONE. Delivered %d non-empty clusters' % (
            xSS.K))
    # Package up algorithm final state and Lscore trace
    DebugInfo.update(dict(
        targetZ=targetZnew,
        targetData=targetData,
        Mu=Mu,
        Lscores=Lscores))
    return xSS, DebugInfo

def initKMeans_BregmanMixture(Data, K, obsModel, seed=0,
        optim_method='frankwolfe'):
    '''

    Returns
    -------
    Z : 1D array

    '''
    PRNG = np.random.RandomState(int(seed))
    X = Data.getDocTypeCountMatrix()
    V = Data.vocab_size
    # Select first cluster mean as uniform distribution
    Mu0 = obsModel.calcSmoothedMu(np.zeros(V))
    # Initialize list to hold all Mu values
    Mu = [None for k in range(K)]
    Mu[0] = Mu0
    chosenZ = np.zeros(K, dtype=np.int32)
    chosenZ[0] = -1
    # Compute minDiv
    minDiv, DivDataVec = obsModel.calcSmoothedBregDiv(
        X=X, Mu=Mu0,
        returnDivDataVec=True,
        return1D=True,
        smoothFrac=1.0)
    Pi = np.ones((X.shape[0], K))
    scoreVsK = list()
    for k in xrange(1, K):
        sum_minDiv = np.sum(minDiv)        
        scoreVsK.append(sum_minDiv)
        if sum_minDiv == 0.0:
            # Duplicate rows corner case
            # Some rows of X may be exact copies, 
            # leading to all minDiv being zero if chosen covers all copies
            chosenZ = chosenZ[:k]
            for emptyk in reversed(range(k, K)):
                # Remove remaining entries in the Mu list,
                # so its total size is now k, not K
                Mu.pop(emptyk)
            assert len(Mu) == chosenZ.size
            break
        elif sum_minDiv < 0 or not np.isfinite(sum_minDiv):
            raise ValueError("sum_minDiv not valid: %f" % (sum_minDiv))
        pvec = minDiv / np.sum(sum_minDiv)
        chosenZ[k] = PRNG.choice(X.shape[0], p=pvec)
        Mu[k] = obsModel.calcSmoothedMu(X[chosenZ[k]])

        # Compute next value of pi
        Pi, minDiv = estimatePiAndDiv_ManyDocs(Data, obsModel, Mu, Pi, k+1,
            minDiv=minDiv,
            DivDataVec=DivDataVec,
            optim_method=optim_method)

    scoreVsK.append(np.sum(minDiv))
    #assert np.all(np.diff(scoreVsK) >= -1e-6)
    print scoreVsK

    Z = -1 * np.ones(Data.nDoc)
    if chosenZ[0] == -1:
        Z[chosenZ[1:]] = np.arange(chosenZ.size - 1)
    else:
        Z[chosenZ] = np.arange(chosenZ.size)
    # Without full pass through dataset, many items not assigned
    # which we indicated with Z value of -1
    # Should ignore this when counting states
    uniqueZ = np.unique(Z)
    uniqueZ = uniqueZ[uniqueZ >= 0]
    assert len(Mu) == uniqueZ.size + 1 # prior
    return Z, Mu, minDiv, np.sum(DivDataVec), scoreVsK


def estimatePiAndDiv_ManyDocs(Data, obsModel, Mu, Pi, k, alpha=0.0,
        optim_method='frankwolfe',
        DivDataVec=None,
        minDiv=None):
    '''
    '''
    if minDiv is None:
        minDiv = np.zeros(Data.nDoc)
    if isinstance(Mu, list):
        topics = np.vstack(Mu[:k])    
    else:
        topics = Mu[:k]
    for d in range(Data.nDoc):
        start_d = Data.doc_range[d]
        stop_d = Data.doc_range[d+1]
        wids_d = Data.word_id[start_d:stop_d]
        wcts_d = Data.word_count[start_d:stop_d]
        # Todo: smart initialization of pi??
        piInit = Pi[d, :k].copy()
        piInit[-1] = 0.1
        piInit[:-1] *= 0.9
        assert np.allclose(piInit.sum(), 1.0)
        if optim_method == 'frankwolfe':
            Pi[d, :k], minDiv[d], _ = estimatePiForDoc_frankwolfe(
                ids_d=wids_d, 
                cts_d=wcts_d,
                topics=topics,
                alpha=alpha)
        else:
            Pi[d, :k], minDiv[d], _ = estimatePi2(
                ids_d=wids_d, 
                cts_d=wcts_d,
                topics=topics,
                alpha=alpha,
                scale=1.0,
                piInit=None)
                #piInit=piInit)
        if d == 0:
            print pi2str(Pi[d,:k])
    minDiv -= np.dot(np.log(np.dot(Pi[:, :k], topics)), obsModel.Prior.lam)
    if DivDataVec is not None:
        minDiv += DivDataVec
    assert np.min(minDiv) > -1e-6
    np.maximum(minDiv, 0, out=minDiv)
    return Pi, minDiv

if __name__ == '__main__':
    import CleanBarsK10
    Data = CleanBarsK10.get_data(nDocTotal=100, nWordsPerDoc=500)
    K = 3

    #import nips
    #Data = nips.get_data()
    hmodel, Info = bnpy.run(Data, 'DPMixtureModel', 'Mult', 'memoVB', 
        initname='bregmankmeans+0',
        K=K,
        nLap=0)

    obsModel = hmodel.obsModel.copy()
    bestMu = None
    bestScore = np.inf
    nTrial = 1
    for trial in range(nTrial):
        chosenZ, Mu, minDiv, sumDataTerm, scoreVsK = initMu_BregmanMixture(
            Data, K, obsModel, seed=trial)
        score = np.sum(minDiv)
        print "init %d/%d : sum(minDiv) %8.2f" % (trial, nTrial, np.sum(minDiv))
        if score < bestScore:
            bestScore = score
            bestMu = Mu
            print "*** New best"
