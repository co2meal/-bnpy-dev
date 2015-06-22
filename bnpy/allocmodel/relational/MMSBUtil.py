'''
MMSBUtil.py

TODO : Add reference for the assortative O(K) tricks
'''
import numpy as np
from scipy.cluster.vq import kmeans2

from bnpy.util import gammaln, digamma, EPS


def calc_full_resp_sparse(Data, ElogPi, logSoftEv, epsilonEv, K):
    '''
    Computes the full unnormalized N x N x K x K resp (i.e. phi) matrix, which
    isn't normally necessary for the assortative-MMSB.

    This is only here for testing and error checking.
    '''
    print '********* COMPUTING FULL RESPONSIBILITY MATRIX **********'

    fullResp = ElogPi[Data.nodes, np.newaxis, :, np.newaxis] + \
        ElogPi[np.newaxis, :, np.newaxis, :] + np.log(epsilonEv[0])
    fullResp[Data.respInds[:, 0], Data.respInds[:, 1]] += \
        (np.log(epsilonEv[1]) - np.log(epsilonEv[0]))
    diag = np.diag_indices(K)

    fullResp[:, :, diag[0], diag[1]
             ] += (logSoftEv[0, :] - np.log(epsilonEv[0]))

    tmp = fullResp[Data.respInds[:, 0], Data.respInds[:, 1]]
    tmp[:, diag[0], diag[1]] +=  \
        (logSoftEv[1, :] - np.log(epsilonEv[1]))
    tmp[:, diag[0], diag[1]] -=  \
        (logSoftEv[0, :] - np.log(epsilonEv[0]))
    fullResp[Data.respInds[:, 0], Data.respInds[:, 1]] = tmp
    np.exp(fullResp, out=fullResp)

    fullResp[np.arange(Data.nNodes), Data.nodes] = 0
    if Data.heldOut is not None:
        fullResp[Data.heldOut[0], Data.heldOut[1]] = 0.0

    return fullResp


def calc_full_resp_nonsparse(Data, ELogPi, logSoftEv, K, epsilon, M):
    print '********* COMPUTING FULL RESPONSIBILITY MATRIX **********'
    print '********** NOT WORKING W/ HELD-OUT DATA *****************'

    mask = Data.assortativeMask
    fullResp = np.zeros((Data.nNodes, Data.nNodesTotal, K, K))
    diag = np.diag_indices(K)
    fullResp[:, :] = (mask * np.log((1 - epsilon) / (2 * M)) +
                      (1 - mask) * np.log(epsilon / (2 * M)))[:, :, np.newaxis, np.newaxis]

    fullResp[:, :, diag[0], diag[1]] = logSoftEv
    fullResp += ELogPi[Data.nodes, np.newaxis, :, np.newaxis] + \
        ELogPi[np.newaxis, :, np.newaxis, :]
    np.exp(fullResp, out=fullResp)

    fullResp[np.arange(Data.nNodes), Data.nodes, :, :] = 0.0

    return fullResp


def assortative_elbo_entropy(LP, SS, Data,
                             N, K,
                             ELogPi, expELogPi,
                             epsilon, M=None):

    # return np.sum(-LP['fullResp'] * np.log(LP['fullResp']+EPS))

    if Data.isSparse:
        H = _entropyTermsWithF_sparse(LP, SS, Data,
                                      N, K,
                                      ELogPi, expELogPi,
                                      epsilon)
    else:
        H = _entropyTermsWithF_nonsparse(LP, SS, Data,
                                         N, K,
                                         ELogPi, expELogPi,
                                         epsilon, M)

    # H -= \sum_ij log Z_ij
    arange = np.arange(Data.nNodes)
    LP['normConst'][arange, Data.nodes] = 1.0
    if Data.heldOut is not None:
        LP['normConst'][Data.heldOut[0], Data.heldOut[1]] = 1.0
    H -= np.sum(np.log(LP['normConst']))
    Zterm = -np.sum(np.log(LP['normConst']))
    LP['normConst'][arange, Data.nodes] = np.inf
    if Data.heldOut is not None:
        LP['normConst'][Data.heldOut[0], Data.heldOut[1]] = np.inf

    # H += complicated_1
    H += np.sum(ELogPi * SS.sumSource)
    H += np.sum(ELogPi * SS.sumReceiver)

    #assert np.allclose(-H, np.sum(-LP['fullResp']*np.log(LP['fullResp']+EPS)))
    return -H


def _entropyTermsWithF_nonsparse(LP, SS, Data,
                                 N, K,
                                 ELogPi, expELogPi,
                                 epsilon, M):
    # TODO : MAKE THIS WORK WITH HELD-OUT DATA

    mask = Data.raveledMask
    logSoftEv = LP['E_log_soft_ev']

    # \sum_ij log f(\epsilon, x_{ij})
    sumMask = np.sum(mask) - Data.nNodes
    H = sumMask * np.log((1 - epsilon) / (2 * M)) + \
        (Data.nNodes * Data.nNodesTotal - Data.nNodes - sumMask) * \
        np.log((epsilon) / (2 * M))

    # H += \sum_ij \sum_k \phi_{ijkk} (log f(w_k,x_ij) - log f(\epsilon,x_ij))
    scratch = np.ones((Data.nNodes, Data.nNodesTotal, K))

    diag = np.diag_indices(Data.nNodes)

    # scratch = LP['resp'] * (logSoftEv - (mask*np.log((1-epsilon)/(2*M)) +
    #                                (1-mask)*np.log(epsilon/(2*M)))[:,np.newaxis])

    mask2 = Data.assortativeMask
    square = LP['evSquare']
    scratch2 = LP['respSquare'] * (square - (mask2 * np.log((1 - epsilon) / (2 * M)) +
                                             (1 - mask2) * np.log(epsilon / (2 * M)))[:, :, np.newaxis])

    H += np.sum(scratch2)
    return H


def _entropyTermsWithF_sparse(LP, SS, Data,
                              N, K,
                              ELogPi, expELogPi,
                              epsilon):

    logSoftEv = LP['E_log_soft_ev']
    # return np.sum(-LP['fullResp'] * np.log(LP['fullResp']+EPS))

    # \sum_ij log f(\epsilon, x_{ij})
    if Data.heldOut is not None:
        heldOutPresent = Data.respIndsSet.intersection(Data.heldOutSet)
        heldOut1s = len(heldOutPresent)
        heldOut0s = len(Data.heldOutSet) - heldOut1s
    else:
        heldOut1s = 0
        heldOut0s = 0

    H = (Data.nNodes * Data.nNodesTotal - Data.nNodes - len(Data.edgeSet) - heldOut0s) * np.log(1 - epsilon) +\
        (len(Data.edgeSet) - heldOut1s) * np.log(epsilon)

    # H += \sum_ij \sum_k \phi_{ijkk} (log f(w_k,x_ij) - log f(\epsilon,x_ij))
    scratch = np.ones((Data.nNodes, Data.nNodesTotal, K))

    scratch = LP[
        'resp'] * (logSoftEv[0, :] - np.log(1 - epsilon))[np.newaxis, np.newaxis, :]
    scratch[Data.respInds[:, 0], Data.respInds[:, 1]] /= \
        (logSoftEv[0, :] - np.log(1 - epsilon))[np.newaxis, :]
    scratch[Data.respInds[:, 0], Data.respInds[:, 1]] *= \
        (logSoftEv[1, :] - np.log(epsilon))[np.newaxis, :]
    H += np.sum(scratch)
    # print 'FIRST H = ', H
    return H


def assortative_elbo_extraObsModel_sparse(Data, LP, epsilon):
    '''
    Returns portions of E_q[log p(y | s,r,phi)] not computed by the obsModel.

    Specifically, \sum_{i,j} \sum_{\ell \neq m}^K resp_{ij\ell m}*log(f(eps,x_ij))
    while the obsmodel does \sum_{i,j} \sum_k^K resp_{ijkk} * log f(phi_k,x_ij)
    '''
    extra = (1 - np.sum(LP['resp'], axis=2)) * np.log(1 - epsilon)
    extra[Data.respInds[:, 0], Data.respInds[:, 1]] = \
        (1 - np.sum(LP['resp'][Data.respInds[:, 0], Data.respInds[:, 1]], axis=1)) * \
        np.log(epsilon)
    extra[np.arange(Data.nNodes), Data.nodes] = 0.0

    if Data.heldOut is not None:
        extra[Data.heldOut[0], Data.heldOut[1]] = 0.0

    return np.sum(extra)


def assortative_elbo_extraObsModel_nonsparse(Data, LP, epsilon, M):
    mask = Data.assortativeMask
    extra = mask * np.log((1 - epsilon) / (2 * M)) + \
        (1 - mask) * np.log(epsilon / (2 * M))
    extra[np.arange(Data.nNodes), Data.nodes] = 0.0
    extra = extra.reshape(Data.nNodes * Data.nNodesTotal)
    extra *= (1 - np.sum(LP['resp'], axis=1))

    # TODO : WORK WITH HELD-OUT DATA

    return np.sum(extra)
