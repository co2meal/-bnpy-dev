import numpy as np
import bnpy.viz
import os

from bnpy.viz.PlotComps import plotAndSaveCompsFromSS

def cleanupDeleteSmallClusters(xSSslice, minAtomCountThr):
    ''' Remove all clusters with size less than specified amount.

    Returns
    -------
    xSSslice : SuffStatBag
        May have fewer components than K.
        Will not exactly represent data Dslice afterwards (if delete occurs).
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
        b_mergeLam=None,
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

    # For merges, we can crank up value of the topic-word prior hyperparameter,
    # to prioritize only care big differences in word counts across many terms
    tmpModel = curModel.copy()
    if b_mergeLam is not None:
        tmpModel.obsModel.Prior.lam[:] = b_mergeLam

    mergeID = 0
    for trial in range(3):
        print 'Merge! Wave %d' % (trial)
        tmpModel.obsModel.update_global_params(xSSslice)
        GainLdata = tmpModel.obsModel.calcHardMergeGap_AllPairs(xSSslice)
        triuIDs = np.triu_indices(xSSslice.K, 1)
        posLocs = np.flatnonzero(GainLdata[triuIDs] > 0)
        if posLocs.size == 0:
            # No merges to accept. Stop!
            print 'No more merges to accept. Done.'
            break

        # Rank the positive pairs from largest to smallest
        sortIDs = np.argsort(-1 * GainLdata[triuIDs][posLocs])
        posLocs = posLocs[sortIDs]

        usedUIDs = set()
        uidpairsToAccept = list()
        origidsToAccept = list()
        for loc in posLocs:
            kA = triuIDs[0][loc]
            kB = triuIDs[1][loc]
            uidA = xSSslice.uids[triuIDs[0][loc]]
            uidB = xSSslice.uids[triuIDs[1][loc]]
            if uidA in usedUIDs or uidB in usedUIDs:
                continue
            usedUIDs.add(uidA)
            usedUIDs.add(uidB)
            uidpairsToAccept.append((uidA, uidB))
            origidsToAccept.append((kA, kB))

        for posID, (uidA, uidB) in enumerate(uidpairsToAccept):
            mergeID += 1
            kA, kB = origidsToAccept[posID]
            print 'Merge uids %d and %d: +%.3f' % (
                uidA, uidB, GainLdata[kA,kB])

            xSSslice.mergeComps(uidA=uidA, uidB=uidB)
            if b_debugOutputDir:
                savefilename = os.path.join(
                    b_debugOutputDir, 'MergeComps_%d.png' % (mergeID))
                # Show side-by-side topics
                bnpy.viz.PlotComps.plotCompsFromHModel(
                    tmpModel,
                    compListToPlot=[kA, kB],
                    vocabList=vocabList,
                    xlabels=[str(uidA), str(uidB)],
                    )
                bnpy.viz.PlotUtil.pylab.savefig(
                    savefilename, pad_inches=0, bbox_inches='tight')

    if mergeID > 0 and b_debugOutputDir:
        tmpModel.obsModel.update_global_params(xSSslice)
        plotAndSaveCompsFromSS(
            tmpModel, xSSslice, b_debugOutputDir, 'NewComps_AfterMerge.png',
            vocabList=vocabList,
            )

    return xSSslice

'''
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
'''
