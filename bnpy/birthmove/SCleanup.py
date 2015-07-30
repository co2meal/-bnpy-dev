import numpy as np
import bnpy.viz
import os

def cleanupDeleteSmallClusters(xSSslice, minAtomCountThr):
    ''' Remove all clusters with size less than specified amount.

    Returns
    -------
    xSSslice : SuffStatBag
        May have fewer components than K.
        Will not exactly represent dataset Dslice afterwards (if delete occurs).
    '''
    CountVec = xSSslice.getCountVec()
    badids = np.flatnonzero(CountVec < minAtomCountThr)
    for k in reversed(badids):
        if xSSslice.K == 1:
            break
        xSSslice.removeComp(k)
    return xSSslice

def cleanupMergeClusters(
        xSSslice, curModel,
        obsSS=None,
        vocabList=None,
        b_mergeLam=5,
        b_debugOutputDir=None,
        **kwargs):
    ''' Merge all possible pairs of clusters that improve the Ldata objective.

    Returns
    -------
    xSSslice : SuffStatBag
        May have fewer components than K.
    '''
    xSSslice.removeELBOandMergeTerms()

    # Discard all fields unrelated to observation model
    reqFields = set()
    for key in obsSS._Fields._FieldDims.keys():
        reqFields.add(key)
    for key in xSSslice._Fields._FieldDims.keys():
        if key not in reqFields:
            xSSslice.removeField(key)

    # For merges, we crank up value of the topic-word prior hyperparameter,
    # to prioritize only care big differences in word counts across many terms
    tmpModel = curModel.copy()
    tmpModel.obsModel.Prior.lam[:] = b_mergeLam 
    tmpModel.obsModel.update_global_params(xSSslice)
    GainLdata = tmpModel.obsModel.calcHardMergeGap_AllPairs(xSSslice)
    triuIDs = np.triu_indices(xSSslice.K, 1)
    posLocs = np.flatnonzero(GainLdata[triuIDs] > 0)
    mergeID = 0
    for loc in reversed(posLocs):
        kA = triuIDs[0][loc]
        kB = triuIDs[1][loc]
        if xSSslice.K > kB:
            mergeID += 1
            if b_debugOutputDir:
                savefilename = os.path.join(
                    b_debugOutputDir, 'MergeComps_%d.png' % (mergeID))
                # Show side-by-side topics
                bnpy.viz.PlotComps.plotCompsFromHModel(
                    tmpModel,
                    compListToPlot=[kA, kB],
                    vocabList=vocabList,
                    xlabels=[str(xSSslice.uids[kA]), str(xSSslice.uids[kB])],
                    )
                bnpy.viz.PlotUtil.pylab.savefig(
                    savefilename, pad_inches=0, bbox_inches='tight')
            xSSslice.mergeComps(kA, kB)
    return xSSslice
