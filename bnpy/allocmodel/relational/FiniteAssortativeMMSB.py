'''
FiniteAssortativeMMSB.py

Assortative mixed membership stochastic blockmodel.
'''
import numpy as np
from scipy.cluster.vq import kmeans2

from bnpy.allocmodel import AllocModel
from bnpy.suffstats import SuffStatBag
from bnpy.util import gammaln, digamma, EPS

from FiniteMMSB import FiniteMMSB
import MMSBUtil


class FiniteAssortativeMMSB(FiniteMMSB):

    """ Assortative version of FiniteMMSB. Finite number of components K.

    Attributes
    -------
    * inferType : string {'EM', 'VB', 'moVB', 'soVB'}
        indicates which updates to perform for local/global steps
    * K : int
        number of components
    * alpha : float
        scalar symmetric Dirichlet prior on mixture weights

    Attributes for VB
    ---------
    * theta : 1D array, size K
        Estimated parameters for Dirichlet posterior over mix weights
        theta[k] > 0 for all k
    """

    def __init__(self, inferType, priorDict=dict()):
        super(FiniteaMMSB, self).__init__(inferType, priorDict)
        self.expELogPi = None
        self.sumExpELogPi = None

    def set_prior(self, alpha=0.1, epsilon=0.05):
        self.alpha = alpha
        self.epsilon = epsilon

    def getSSDims(self):
        ''' Get dimensions of interactions between components.
        '''
        return ('K',)

    def calc_local_params(self, Data, LP, **kwargs):
        if self.inferType.count('EM') > 0:
            pass
        N = Data.nNodes
        K = self.K
        D = Data.dim
        respI = Data.respInds
        ElogPi = digamma(self.theta) - \
            digamma(np.sum(self.theta, axis=1))[:, np.newaxis]
        self._update_Epi_stats(N=Data.nNodesTotal, K=self.K)

        if Data.isSparse:
            logSoftEv = LP['E_log_soft_ev']
            expLogSoftEv = np.exp(logSoftEv)
            epsilonEv = np.zeros(2)
            epsilonEv[0] = 1 - self.epsilon
            epsilonEv[1] = self.epsilon

            assert logSoftEv.shape == (2, K)
            resp = ElogPi[Data.nodes, np.newaxis, :] + \
                ElogPi[np.newaxis, :, :] + \
                logSoftEv[0, :]

            resp[respI[:, 0], respI[:, 1]] += \
                (logSoftEv[1, :] - logSoftEv[0, :])
            np.exp(resp, out=resp)

            # Calculate the normalization constant for each resp_{ij}
            # sum_k \tilde\pi_{ik} \tilde\pi_{jk} (f(w_k,x_{ij}) -
            # f(\eps,x_{ij}))
            normConst = np.einsum(
                'ik,jk,k->ij', self.expELogPi, self.expELogPi,
                expLogSoftEv[0, :] - epsilonEv[0])
            normConst[respI[:, 0], respI[:, 1]] = np.einsum(
                'ik,ik,k->i', self.expELogPi[respI[:, 0]],
                self.expELogPi[respI[:, 1]],
                expLogSoftEv[1, :] - epsilonEv[1])

            normConst += self.sumExpELogPi[np.newaxis, :] * \
                self.sumExpELogPi[:, np.newaxis] * \
                epsilonEv[0]
            normConst[respI[:, 0], respI[:, 1]] -= \
                self.sumExpELogPi[respI[:, 0]] * \
                self.sumExpELogPi[respI[:, 1]] * \
                epsilonEv[0]
            normConst[respI[:, 0], respI[:, 1]] += \
                self.sumExpELogPi[respI[:, 0]] * \
                self.sumExpELogPi[respI[:, 1]] * \
                epsilonEv[1]
            resp /= normConst[:, :, np.newaxis]

            # Do the right thing for entries resp_{ii}
            diag = np.diag_indices(N)
            normConst[diag] = np.inf
            resp[diag] = 0

            # Using sparse data, so the obsmodel needs us to comp. counts here
            LP['Count1'] = np.sum(
                resp[Data.respInds[:, 0],
                     Data.respInds[:, 1]],
                axis=0)
            LP['Count0'] = np.sum(resp, axis=(0, 1)) - LP['Count1']

            LP['normConst'] = normConst
            LP['resp'] = resp
            LP['fullResp'] = MMSBUtil.calc_full_resp(Data, ElogPi, logSoftEv,
                                                     epsilonEv, self.K)
            LP['fullResp'] /= normConst[:, :, np.newaxis, np.newaxis]

            return LP

    def _update_Epi_stats(self, N, K):
        if self.expELogPi is None or self.expELogPi.shape != (N, K):
            self.expELogPi = np.zeros((N, K))
        if self.sumExpELogPi is None or self.sumExpELogPi.shape != (N):
            self.sumExpELogPi = np.zeros(N)

        ElogPi = digamma(self.theta) - \
            digamma(np.sum(self.theta, axis=1))[:, np.newaxis]
        np.exp(ElogPi, out=self.expELogPi)
        np.sum(self.expELogPi, axis=1, out=self.sumExpELogPi)

    def get_global_suff_stats(self, Data, LP, doPrecompEntropy=None, **kwargs):
        nNodes = Data.nNodes
        K = LP['resp'].shape[-1]
        self.K = K

        # Calculate sumSource[i,l] = \sum_j s_{ijl} = \sum_j \sum_m \phi_{ijlm}
        scratch = np.ones((nNodes, nNodes, K))
        scratch *= (1 - self.epsilon) / LP['normConst'][:, :, np.newaxis]

        scratch *= (self.sumExpELogPi[:,
                                      np.newaxis] - self.expELogPi)[np.newaxis,
                                                                    :,
                                                                    :]
        scratch[Data.respInds[:, 0], Data.respInds[:, 1]] *= \
            (self.epsilon / (1 - self.epsilon))

        sumSource = np.sum(scratch, axis=1) * self.expELogPi
        sumSource += np.sum(LP['resp'], axis=1)

        # Calculate sumReceiver[i,l]
        # = \sum_j r_{jil} = \sum_j \sum_m \phi_{jiml}
        scratch.fill(1)
        scratch *= (1 - self.epsilon) / LP['normConst'].T[:, :, np.newaxis]

        scratch *= (self.sumExpELogPi[:,
                                      np.newaxis] - self.expELogPi)[np.newaxis,
                                                                    :,
                                                                    :]
        scratch[Data.respInds[:, 1], Data.respInds[:, 0]] *= \
            (self.epsilon / (1 - self.epsilon))

        sumReceiver = np.sum(scratch, axis=1) * self.expELogPi
        sumReceiver += np.sum(LP['resp'], axis=0)

        assert np.allclose(sumReceiver, np.sum(LP['fullResp'], axis=(0, 2)))
        assert np.allclose(sumSource, np.sum(LP['fullResp'], axis=(1, 3)))

        SS = SuffStatBag(K=K, D=Data.dim, nNodes=Data.nNodes)
        SS.setField('sumSource', sumSource, dims=('nNodes', 'K'))
        SS.setField('sumReceiver', sumReceiver, dims=('nNodes', 'K'))
        # TODO : should this also compute the K x K Npair?
        SS.setField('N', np.sum(LP['resp'], axis=(0, 1)), dims=('K'))
        return SS

    def update_global_params_VB(self, SS, **kwargs):
        self.theta = self.alpha + SS.sumSource + SS.sumReceiver

    def calc_evidence(self, Data, SS, LP, **kwargs):
        alloc = self.elbo_alloc_no_slack(Data, LP, SS)
        if SS.hasELBOTerm('Elogqz'):
            entropy = SS.getELBOTerm('Elogqz')
        else:
            entropy = MMSBUtil.assortative_elbo_entropy(
                LP, Data, self.epsilon,
                Data.nNodesTotal, self.K,
                self.expELogPi,
                self.sumExpELogPi)
        extraObsModelTerm = MMSBUtil.assortative_elbo_extraObsModel(
            Data, LP, self.epsilon)

        return alloc + entropy + extraObsModelTerm

    def elbo_extraObsModel(self, Data, LP):
        extra = (1 - np.sum(LP['resp'], axis=2)) * np.log(1 - self.epsilon)
        extra[Data.respInds[:, 0], Data.respInds[:, 1]] *= \
            (np.log(self.epsilon) / np.log(1 - self.epsilon))
        extra[np.diag_indices(Data.nNodes)] = 0
        assert extra.shape == (Data.nNodes, Data.nNodes)
        oldExtra = np.sum(extra)

        extra = np.sum(LP['resp'], axis=2) * np.log(1 - self.epsilon)
        extra[Data.respInds[:, 0], Data.respInds[:, 1]] *= \
            (np.log(self.epsilon) / np.log(1 - self.epsilon))
        extra[np.diag_indices(Data.nNodes)] = 0
        newExtra = np.sum(1 - extra)
        return oldExtra

    def elbo_entropy(self, LP, Data):
        oldH = -np.sum(LP['fullResp'] * np.log(LP['fullResp'] + EPS))
        logSoftEv = LP['E_log_soft_ev']

        # \sum_ij log f(\epsilon, x_{ij})
        H = (Data.nNodes**2 - Data.nNodes - len(Data.edgeSet)) * \
            np.log(1 - self.epsilon) + \
            len(Data.edgeSet) * np.log(self.epsilon)

        # H += \sum_ij \sum_k \phi_{ijkk}
        #      (log f(w_k,x_ij) - log f(\epsilon,x_ij))
        # scratch = np.ones((Data.nNodes, Data.nNodes, self.K))
        scratch = LP['resp'] * (
            logSoftEv[0, :] - np.log(1 - self.epsilon))
        scratch = scratch[np.newaxis, np.newaxis, :]
        scratch[Data.respInds[:, 0], Data.respInds[:, 1]] /= \
            (logSoftEv[0, :] - np.log(1 - self.epsilon))[np.newaxis, :]
        scratch[Data.respInds[:, 0], Data.respInds[:, 1]] *= \
            (logSoftEv[1, :] - np.log(self.epsilon))[np.newaxis, :]
        term1 = np.sum(scratch)
        H += np.sum(scratch)

        # H -= \sum_ij log Z_ij
        diag = np.diag_indices(Data.nNodes)
        LP['normConst'][diag] = 1.0
        H -= np.sum(np.log(LP['normConst']))
        Zterm = -np.sum(np.log(LP['normConst']))
        LP['normConst'][diag] = np.inf

        # H += complicated_1
        scratch.fill(1)
        scratch *= (1 - self.epsilon) / LP['normConst'][:, :, np.newaxis]
        scratch[Data.respInds[:, 0], Data.respInds[:, 1]] *= \
            (self.epsilon / (1 - self.epsilon))
        scratch *= (self.sumExpELogPi[:,
                                      np.newaxis] - self.expELogPi)[np.newaxis,
                                                                    :,
                                                                    :]
        scratch *= self.expELogPi[:, np.newaxis, :]
        scratch += LP['resp']
        preS1 = scratch.copy()
        scratch *= np.log(self.expELogPi)[:, np.newaxis, :]
        s1 = scratch.copy()
        assert np.allclose(preS1, np.sum(LP['fullResp'], axis=3))
        H += np.sum(scratch)

        scratch.fill(1)
        scratch *= (1 - self.epsilon) / LP['normConst'][:, :, np.newaxis]
        scratch[Data.respInds[:, 0], Data.respInds[:, 1]] *= \
            (self.epsilon / (1 - self.epsilon))
        scratch *= (self.sumExpELogPi[:,
                                      np.newaxis] - self.expELogPi)[:,
                                                                    np.newaxis,
                                                                    :]
        scratch *= self.expELogPi[np.newaxis, :, :]
        scratch += LP['resp']
        preS2 = scratch.copy()
        scratch *= np.log(self.expELogPi)[np.newaxis, :, :]
        s2 = scratch.copy()
        assert np.allclose(preS2, np.sum(LP['fullResp'], axis=2))
        H += np.sum(scratch)

        assert np.allclose(oldH, -H)

        return -H
