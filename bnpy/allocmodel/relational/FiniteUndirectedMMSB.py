'''
FiniteUndirectedMMSB.py
'''

import numpy as np
from scipy.cluster.vq import kmeans2

from bnpy.allocmodel import AllocModel
from bnpy.suffstats import SuffStatBag
from bnpy.util import gammaln, digamma, EPS
from bnpy.allocmodel.topics.HDPTopicModel import c_Beta, c_Dir
from FiniteMMSB import FiniteMMSB


class FiniteUndirectedMMSB(FiniteMMSB):

    def __init__(self, inferType, priorDict=dict()):
        super(FiniteUndirectedMMSB, self).__init__(inferType, priorDict)

    def getSSDims(self):
        ''' Determine dimensions of statistics for obsmodel.

        Overrides default ('K'), to be E x K(K+1)/2
        '''
        return ('KupperTriangular',)

    def initLPFromTruth(self, Data):
        LP = super(FiniteUndirectedMMSB, self).initLPFromTruth(Data)
        K = np.max(Data.TrueParams['Z']) + 1
        N = Data.nNodes
        Z = Data.TrueParams['Z']
        upper = np.triu_indices(K)
        diag = np.diag_indices(K)

        Count1 = LP['Count1']
        Count0 = LP['Count0']
        Count1 /= 2
        Count0 /= 2
        Count1 += Count1.T
        Count0 += Count0.T
        Count1[diag] /= 2
        Count0[diag] /= 2

        LP['Count1'] = Count1[upper]
        LP['Count0'] = Count0[upper]
        return LP

    def calc_local_params(self, Data, LP, **kwargs):
        if not Data.isSparse:
            raise NotImplementedError('Not implemented for non-sparse data')

        N = Data.nNodes
        K = self.K
        D = Data.dim
        ElogPi = digamma(self.theta) - \
            digamma(np.sum(self.theta, axis=1))[:, np.newaxis]

        upper = np.triu_indices(K)
        diag = np.diag_indices(K)
        logSoftEv = np.zeros((2, K, K))

        if Data.isSparse:
            logSoftEvFlat = LP['E_log_soft_ev']
            assert logSoftEvFlat.shape == (2, K * (K + 1) / 2)

            for i in xrange(2):
                logSoftEv[i, upper[0], upper[1]] = logSoftEvFlat[i]
                logSoftEv[i] += logSoftEv[i].T
                logSoftEv[i, diag[0], diag[1]] /= 2

            resp = ElogPi[Data.nodes, np.newaxis, :, np.newaxis] + \
                ElogPi[np.newaxis, :, np.newaxis, :] + logSoftEv[0, :, :]
            resp[Data.respInds[:, 0], Data.respInds[:, 1]] += \
                (logSoftEv[1, :, :] - logSoftEv[0, :, :])

            np.exp(resp, out=resp)
            resp /= np.sum(resp, axis=(2, 3))[:, :, np.newaxis, np.newaxis]
            resp[np.diag_indices(Data.nNodes)[0], Data.nodes, :, :] = 0

            Count1 = np.sum(
                resp[
                    Data.respInds[
                        :, 0], Data.respInds[
                        :, 1]], axis=0)
            Count0 = np.sum(resp, axis=(0, 1)) - Count1
            LP['resp'] = resp.reshape((Data.nNodes * Data.nNodesTotal, K, K))
            LP['squareResp'] = resp

        Count1 /= 2
        Count0 /= 2
        Count1 += Count1.T
        Count0 += Count0.T
        Count1[diag] /= 2
        Count0[diag] /= 2

        LP['Count1'] = Count1[upper]
        LP['Count0'] = Count0[upper]
        return LP

    def get_global_suff_stats(self, Data, LP, doPrecompEntropy=None, **kwargs):
        N = Data.nNodesTotal
        K = LP['resp'].shape[-1]
        SS = SuffStatBag(K=K,
                         D=Data.dim,
                         nNodes=N,
                         KupperTriangular=K * (K + 1) / 2)

        sumSource = np.zeros((N, K))
        sumSource[Data.nodes, :] = np.sum(LP['squareResp'], axis=(1, 3))
        SS.setField('sumSource', sumSource, dims=('nNodes', 'K'))
        SS.setField('N', np.sum(LP['resp'], axis=0) / 2, dims=('K', 'K'))

        if doPrecompEntropy is not None:
            entropy = self.elbo_entropy(LP)
            SS.setELBOTerm('Elogqz', entropy, dims=None)

        return SS

    def forceSSInBounds(self, SS):
        ''' Force certain fields in bounds, to avoid numerical issues.

        Returns
        -------
        Nothing.  SS is updated in-place.
        '''
        np.maximum(SS.sumSource, 0, out=SS.sumSource)

    def update_global_params_VB(self, SS, **kwargs):
        self.theta = self.alpha + SS.sumSource

    def calc_evidence(self, Data, SS, LP, **kwargs):
        N = float(SS.sumSource.shape[0])
        alloc = self.elbo_alloc_no_slack(Data, LP, SS)
        if SS.hasELBOTerm('Elogqz'):
            entropy = SS.getELBOTerm('Elogqz')
        else:
            entropy = self.elbo_entropy(LP, Data)

        entropy *= ((N * (N - 1) / 2) / (N**2 - N))
        return alloc + entropy
