import numpy as np
from scipy.special import gammaln, digamma

import bnpy

'''
BRAINSTORM

* option A: reconfigure all words of given type
pro: big changes
con: cant accept any other moves for entire lap

* option B: reconfig r vals only for words of type assigned to target topic
pro: can change any assignments to non-involved topics
con: 
'''



def evaluateReconfigWordMoveCandidate_LP(
        Data, curModel, 
        curLP=None,
        propLP=None,
        targetCompID=0,
        destCompIDs=[1],
        targetWordIDs=None,
        **kwargs):
    propModel = curModel.copy()
    curModel = curModel.copy()
    Korig = curModel.allocModel.K


    # Evaluate current model
    curSS = curModel.get_global_suff_stats(
        Data, curLP, 
        doPrecompEntropy=1)
    curModel.update_global_params(curSS)
    curELBO = curModel.calc_evidence(SS=curSS)
    print ' current ELBO: %.5f' % (curELBO)

    # Visualize proposed expansion
    propSS = curModel.get_global_suff_stats(Data, propLP, doPrecompEntropy=1)
    propModel.update_global_params(propSS)

    mPairIDs = [(targetCompID, Korig)]
    for ctr, kk in enumerate(destCompIDs):
        mPairIDs.append((kk, Korig+ctr+1))
    print 'Candidate merge pairs: '
    print mPairIDs
    
    # Create full expansion (including merge terms)
    propSS = propModel.get_global_suff_stats(
        Data, propLP, 
        doPrecompEntropy=1, doPrecompMergeEntropy=1, mPairIDs=mPairIDs)
    propModel.update_global_params(propSS)
    propELBO = propModel.calc_evidence(SS=propSS)
    print 'expanded ELBO: %.5f' % (propELBO)
    
    # Create final refined model after merging
    finalModel, finalSS, finalELBO, Info = \
        bnpy.mergemove.MergeMove.run_many_merge_moves(
            propModel, propSS, propELBO, mPairIDs)

    print 'Accepted merge pairs: '
    print Info['AcceptedPairOrigIDs']

    finalELBO = finalModel.calc_evidence(SS=finalSS)
    print '   final ELBO: %.5f' % (finalELBO)
    
    return finalModel, dict(
        SS=finalSS,
        ELBO=finalELBO,
        MergeInfo=Info
        )


def makeReconfigWordMoveCandidate_LP(
        Data, curLP, curModel, 
        targetWordIDs=[0,1,2,3],
        targetCompID=5,
        destCompIDs=[0],
        deleteStrategy='truelabels',
        minResp=0.001,
        **curLPkwargs):
    '''

    Returns
    -------
    propcurLP : dict of local params
        Replaces targetCompID with K "new" states,
        each one tracking exactly one existing state.
    '''

    curResp = curLP['resp']
    maxRespValBelowThr = curResp[curResp < minResp].max()
    assert maxRespValBelowThr < 1e-90

    Natom, Korig = curResp.shape
    destCompIDs = np.insert(destCompIDs, 0, targetCompID)
    Kprop = Korig + len(destCompIDs)

    propResp = 1e-100 * np.ones((Natom, Kprop))
    propResp[:, :Korig] = curResp
    if deleteStrategy.count('truelabels'):
        # Identify atoms that match both target state and one-of target wordids
        relAtoms_byWords = np.zeros(Data.word_id.size, dtype=np.int32)
        for v in targetWordIDs:
            relAtoms_v = Data.word_id == v
            relAtoms_byWords = np.logical_or(relAtoms_byWords, relAtoms_v)
        relAtoms_byComp = curLP['resp'][:, targetCompID] > minResp
        relAtoms_twords = np.logical_and(relAtoms_byWords, relAtoms_byComp)
        relAtoms_nottwords = np.logical_and(
            relAtoms_byComp, 
            np.logical_not(relAtoms_byWords))
        
        # Keep non-target-word atoms assigned as they are
        propResp[relAtoms_nottwords, targetCompID] = \
            curResp[relAtoms_nottwords, targetCompID]
        propResp[relAtoms_twords, targetCompID] = 1e-100

        # Re-assign target-word atoms based on ground-truth labels
        reltrueResp = Data.TrueParams['resp'][relAtoms_twords].copy()
        reltrueResp = reltrueResp[:, destCompIDs]
        reltrueResp[reltrueResp < minResp] = 1e-100
        reltrueResp /= reltrueResp.sum(axis=1)[:,np.newaxis]
        reltrueResp *= curResp[relAtoms_twords, targetCompID][:, np.newaxis]
        assert np.allclose(curResp[relAtoms_twords, targetCompID],
                           reltrueResp.sum(axis=1))

        propResp[relAtoms_twords, Korig:] = reltrueResp

        assert np.allclose(1.0, propResp.sum(axis=1))
        propLP = curModel.allocModel.initLPFromResp(
            Data, dict(resp=propResp))
        return propLP
    
    Lik = curLP['E_log_soft_ev'][:, remCompIDs].copy()

    # From-scratch strategy
    for d in relDocIDs:
        mask_d = np.arange(Data.doc_range[d],Data.doc_range[d+1])
        relAtomIDs_d = mask_d[
            curLP['resp'][mask_d, targetCompID] > minResp]
        fixedDocTopicCount_d = curLP['DocTopicCount'][d, remCompIDs]
        relLik_d = Lik[relAtomIDs_d, :]
        relwc_d = Data.word_count[relAtomIDs_d]
        
        targetsumResp_d = curLP['resp'][relAtomIDs_d, targetCompID] * relwc_d
        sumResp_d = np.zeros_like(targetsumResp_d)
        
        DocTopicCount_d = np.zeros_like(fixedDocTopicCount_d)
        DocTopicProb_d = np.zeros_like(DocTopicCount_d)
        sumalphaEbeta = curModel.allocModel.alpha_E_beta()[targetCompID]
        alphaEbeta = sumalphaEbeta * 1.0 / (Korig-1.0) * np.ones(Korig-1)
        for riter in range(10):
            np.add(DocTopicCount_d, alphaEbeta, out=DocTopicProb_d)
            digamma(DocTopicProb_d, out=DocTopicProb_d)
            DocTopicProb_d -= DocTopicProb_d.max()
            np.exp(DocTopicProb_d, out=DocTopicProb_d)
            
            # Update sumResp for all tokens in document
            np.dot(relLik_d, DocTopicProb_d, out=sumResp_d)

            # Update DocTopicCount_d: 1D array, shape K
            #     sum(DocTopicCount_d) equals Nd[targetCompID]
            np.dot(targetsumResp_d / sumResp_d, relLik_d, out=DocTopicCount_d)
            DocTopicCount_d *= DocTopicProb_d
            DocTopicCount_d += fixedDocTopicCount_d

        DocTopicCount_dj = curLP['DocTopicCount'][d, targetCompID]
        DocTopicCount_dnew = np.sum(DocTopicCount_d) - \
            fixedDocTopicCount_d.sum()
        assert np.allclose(DocTopicCount_dj, DocTopicCount_dnew,
                           rtol=0, atol=1e-6)

        # Create proposal resp for relevant atoms in this doc only
        propResp_d = relLik_d.copy()
        propResp_d *= DocTopicProb_d[np.newaxis, :]
        propResp_d /= sumResp_d[:, np.newaxis]
        propResp_d *= curLP['resp'][relAtomIDs_d, targetCompID][:,np.newaxis]

        for n in range(propResp_d.shape[0]):
            size_n = curLP['resp'][relAtomIDs_d[n], targetCompID]
            sizeOrder_n = np.argsort(propResp_d[n,:])
            for k, compID in enumerate(sizeOrder_n):
                if propResp_d[n, compID] > minResp:
                    break
                propResp_d[n, compID] = 1e-100
                biggerCompIDs = sizeOrder_n[k+1:]
                propResp_d[n, biggerCompIDs] /= \
                    propResp_d[n,biggerCompIDs].sum()
                propResp_d[n, biggerCompIDs] *= size_n

        # Fill in huge resp matrix with specific values
        propResp[relAtomIDs_d, Korig-1:] = propResp_d
        assert np.allclose(propResp.sum(axis=1), 1.0, rtol=0, atol=1e-8)

    propcurLP = curModel.allocModel.initLPFromResp(Data, dict(resp=propResp))
    return propcurLP

def makeLPWithMinNonzeroValFromLP(Data, hmodel, LP, minResp=0.001):
    ''' Create sparse-ified local parameters, where all resp > threshold

    Returns
    -------
    sparseLP : dict of local parameters
    '''
    respS = LP['resp'].copy()
    Natom, Korig = respS.shape
    for n in xrange(Natom):
        sizeOrder_n = np.argsort(respS[n,:])
        for posLoc, compID in enumerate(sizeOrder_n):
            if respS[n, compID] > minResp:
                break
            respS[n, compID] = 1e-100
            biggerCompIDs = sizeOrder_n[posLoc+1:]
            respS[n, biggerCompIDs] /= respS[n, biggerCompIDs].sum()
    sparseLP = hmodel.allocModel.initLPFromResp(Data, dict(resp=respS))
    sparseLP['E_log_soft_ev'] = LP['E_log_soft_ev'].copy()
    return sparseLP