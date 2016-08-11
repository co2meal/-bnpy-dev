from __future__ import print_function

import numpy as np
import itertools
import Symbols as S
import bnpy
from scipy.sparse import csr_matrix
from bnpy.init.FromExistingBregman import runKMeans_BregmanDiv_existing
from bnpy.mergemove.MPlanner import selectCandidateMergePairs
from bnpy.viz.PlotUtil import pylab
from bnpy.allocmodel.topics.LocalStepManyDocs import updateLPGivenDocTopicCount

if __name__ == '__main__':
    Npersymbol = 1000
    Ndoc = 10
    Kfresh = 10

    # Create training set
    # Each document will have exactly Npersymbol/Ndoc examples of each cluster
    # For exactly Npersymbol total examples of each cluster across the corpus
    Xlist = list()
    Zlist = list()
    doc_range = [0]
    PRNG = np.random.RandomState(0)
    for d in xrange(Ndoc):
        N_doc = 0
        for k, patch_name in enumerate(
                ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']):
            N_d_k =  Npersymbol // Ndoc
            X_ND = S.generate_patches_for_symbol(patch_name, N_d_k)
            Xlist.append(X_ND)
            Zlist.append(k * np.ones(N_d_k, dtype=np.int32))
            N_doc += N_d_k
        doc_range.append(N_doc + doc_range[-1])
    X = np.vstack(Xlist)
    TrainData = bnpy.data.GroupXData(
        X,
        doc_range=doc_range,
        TrueZ=np.hstack(Zlist))
    TrainData.name = 'SimpleSymbols'

    # Train simple HDP model on this set
    trainedModel, RInfo = bnpy.run(
        TrainData, 'HDPTopicModel', 'ZeroMeanGauss', 'memoVB',
        initname='truelabels',
        nLap=50, nBatch=1,
        moves='merge', m_startLap=5,
        ECovMat='eye', sF=0.01,
        gamma=10.0,
        alpha=0.5)
    Korig = trainedModel.obsModel.K

    # Obtain local params and suff stats for this trained model
    trainLP = trainedModel.calc_local_params(TrainData)
    trainSS = trainedModel.get_global_suff_stats(
        TrainData, trainLP, doPrecompEntropy=1, doTrackTruncationGrowth=1)

    # Create test set, with some novel clusters and some old ones
    Xlist = list()
    for patch_name in ['A', 'B', 'C', 'D',
                       'slash', 'horiz_half', 'vert_half', 'cross']:
        X_ND = S.generate_patches_for_symbol(patch_name, Npersymbol)
        Xlist.append(X_ND)
    X = np.vstack(Xlist)
    TestData = bnpy.data.GroupXData(X, doc_range=[0, len(X)])
    TestData.name = 'SimpleSymbols'

    # Run FromExistingBregman procedure on test set
    print("Expanding model!")
    print("Creating %d new clusters via Bregman k-means++" % (Kfresh))
    print("Then assigning all %d test items to closest cluster")
    print("using union of %d existing and %d new clusters" % (Korig, Kfresh))
    Z, Mu, Lscores = runKMeans_BregmanDiv_existing(
        TestData.X, Kfresh, trainedModel.obsModel,
        assert_monotonic=False,
        Niter=5, logFunc=print)
    Kall = Z.max() + 1
    Kfresh = Kall - Korig
    testLP = dict(
        nnzPerRow=1,
        spR=csr_matrix(
            (np.ones(Z.size), Z, np.arange(0, Z.size+1, 1)),
            shape=(TestData.nObs, Kall))
        )
    DocTopicCount = np.asarray(np.bincount(Z, minlength=Kall).reshape((1, Kall)), dtype=np.float64)
    testLP['DocTopicCount'] = DocTopicCount

    alphaPi0 = np.hstack([
        trainedModel.allocModel.alpha_E_beta(),
        trainedModel.allocModel.E_beta_rem() / (Kfresh + 1) * np.ones(Kfresh)
    ])
    alphaPi0Rem = trainedModel.allocModel.E_beta_rem() / (Kfresh + 1)
    testLP = updateLPGivenDocTopicCount(
        testLP,
        DocTopicCount,
        alphaPi0,
        alphaPi0Rem)
    testSS = trainedModel.get_global_suff_stats(
        TestData,
        testLP,
        doPrecompEntropy=1,
        doTrackTruncationGrowth=1)

    print()
    print("Refining model!")
    print("Performing several full VB iterations")
    print("Merging any new clusters when VB objective approves")

    # Create a combined model for the train AND test set
    trainSS.insertEmptyComps(testSS.K - trainSS.K)
    combinedSS = trainSS + testSS
    combinedModel = trainedModel.copy()
    combinedModel.update_global_params(combinedSS)

    # Refine this combined model via several coord ascent passes thru TestData
    testLP = dict(DocTopicCount=DocTopicCount)
    for aiter in range(10):
        testLP = combinedModel.calc_local_params(TestData,
                        testLP, initDocTopicCountLP='memo')
        testSS = combinedModel.get_global_suff_stats(
            TestData, testLP, doPrecompEntropy=1, doTrackTruncationGrowth=1)

        print("VB refinement iter %d" % aiter)
        print("   orig counts: ",
              ' '.join(['%9.2f' % x for x in testSS.N[:Korig]]))
        print("  fresh counts: ",
              ' '.join(['%9.2f' % x for x in testSS.N[Korig:]]))

        testSS.setUIDs(trainSS.uids)
        combinedSS = trainSS + testSS
        combinedModel.update_global_params(combinedSS)

        if aiter > 2:
            cur_ELBO = combinedModel.calc_evidence(SS=combinedSS)

            # Create list of all unique pairs of "fresh" uids
            m_UIDPairs = [pair for pair in itertools.combinations(
                trainSS.uids[Korig:], 2)]
            acceptedUIDs = set()

            # Try merging each possible pair of uids
            for ii, (uidA, uidB) in enumerate(m_UIDPairs):
                if uidA in acceptedUIDs or uidB in acceptedUIDs:
                    print('pair %2d %2d skipped since one already accepted' % (
                        uidA, uidB))
                    continue

                kA = trainSS.uid2k(uidA)
                kB = trainSS.uid2k(uidB)

                # Remove empty training comp
                prop_trainSS = trainSS.copy()
                prop_trainSS.removeComp(uid=uidB)

                # Update proposed test comp
                prop_testLP = \
                    combinedModel.allocModel.applyHardMergePairToLP(
                        testLP, kA, kB)
                prop_testSS = combinedModel.get_global_suff_stats(
                    TestData, prop_testLP,
                    doPrecompEntropy=1,
                    doTrackTruncationGrowth=1)
                prop_testSS.setUIDs(prop_trainSS.uids)

                prop_combinedSS = prop_trainSS + prop_testSS

                # Create proposed model
                prop_combinedModel = combinedModel.copy()
                prop_combinedModel.update_global_params(prop_combinedSS)

                # Now, compare the ELBO
                prop_ELBO = prop_combinedModel.calc_evidence(
                    SS=prop_combinedSS)
                if prop_ELBO > cur_ELBO:
                    # ACCEPT!
                    combinedModel = prop_combinedModel
                    cur_ELBO = prop_ELBO
                    trainSS = prop_trainSS
                    testSS = prop_testSS
                    testLP = prop_testLP
                    acceptedUIDs.add(uidA)
                    acceptedUIDs.add(uidB)
                    print('pair %2d %2d ACCEPTED!' % (uidA, uidB))
                else:
                    print('pair %2d %2d rejected' % (uidA, uidB))

    print()
    print("Plotting final combined model!")
    print("Each plot shows 25 samples of image patches from that cluster")

    PRNG = np.random.RandomState(0)
    for k in xrange(trainSS.K):
        Sigma_k = combinedModel.obsModel.get_covar_mat_for_comp(k)
        X_k = PRNG.multivariate_normal(
            np.zeros(64),
            Sigma_k,
            size=25)
        figH, axList = pylab.subplots(nrows=5, ncols=5)
        figH.canvas.set_window_title(
            'Cluster %d: Sample patches' % (k))

        ii = 0
        for r in range(5):
            for c in range(5):
                axList[r, c].imshow(
                    X_k[ii].reshape((8, 8)),
                    interpolation='nearest',
                    cmap='gray_r',
                    vmin=-0.1,
                    vmax=0.1)
                ii += 1
                axList[r, c].set_xticks([])
                axList[r, c].set_yticks([])

pylab.show()
