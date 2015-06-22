'''
MMSB.py

'''

import numpy as np
from scipy.cluster.vq import kmeans2

from bnpy.allocmodel import AllocModel
from bnpy.suffstats import SuffStatBag
from bnpy.util import gammaln, digamma, EPS
from bnpy.allocmodel.topics.HDPTopicModel import c_Beta, c_Dir


class FiniteMMSB(AllocModel):
    # Constructors
    #########################################################

    def __init__(self, inferType, priorDict=dict()):
        if inferType.count('EM') > 0:
            raise NotImplementedError(
                'EM not implemented for FiniteMMSB (yet)')

        self.inferType = inferType
        self.set_prior(**priorDict)
        self.K = 0

        # Variational parameter for pi
        self.theta = None

        self.estZ = [0]

    def set_prior(self, alpha=.1):
        self.alpha = alpha

    def get_active_comp_probs(self):
        print 'TODO TODO TODO'

    def getSSDims(self):
        '''Called during obsmodel.setupWithAllocModel to determine the dimensions
           of the statistics computed by the obsmodel.

           Overridden from the default of ('K',), as we need E_log_soft_ev to be
           dimension E x K x K
        '''
        return ('K', 'K',)

    def initLPFromTruth(self, Data):
        K = np.max(Data.TrueParams['Z']) + 1
        N = Data.nNodes
        Z = Data.TrueParams['Z']
        resp = np.zeros((N, N, K, K))
        for i in xrange(N):
            for j in xrange(N):
                resp[i, j, Z[i, j, 0], Z[j, i, 0]] = 1
        diag = np.diag_indices(N)
        resp[diag[0], diag[1], :, :] = 0
        squareResp = resp
        resp = np.reshape(resp, (N**2, K, K))
        LP = {'resp': resp, 'squareResp': squareResp}

        if Data.isSparse:
            LP['Count1'] = np.sum(squareResp[Data.respInds[:, 0],
                                             Data.respInds[:, 1]], axis=0)
            LP['Count0'] = np.sum(squareResp, axis=(0, 1)) - LP['Count1']

        return LP

    # Local Params
    #########################################################
    def calc_local_params(self, Data, LP, **kwargs):

        if self.inferType.count('EM') > 0:
            pass
        N = Data.nNodes
        K = self.K
        D = Data.dim
        ElogPi = digamma(self.theta) - \
            digamma(np.sum(self.theta, axis=1))[:, np.newaxis]

        if Data.isSparse:  # Sparse binary data.
            logSoftEv = LP['E_log_soft_ev']

            cpy = np.copy(logSoftEv)
            diag = np.diag_indices(K)
            logSoftEv[0, :, :] = np.log(.95)
            logSoftEv[1, :, :] = np.log(.05)
            logSoftEv[0, diag[0], diag[1]] = cpy[0, diag[0], diag[1]]
            logSoftEv[1, diag[0], diag[1]] = cpy[1, diag[0], diag[1]]

            expLogSoftEv = np.exp(logSoftEv)
            assert logSoftEv.shape == (2, K, K)
            resp = ElogPi[Data.nodes, np.newaxis, :, np.newaxis] + \
                ElogPi[np.newaxis, :, np.newaxis, :] + logSoftEv[0, :, :]
            resp[Data.respInds[:, 0], Data.respInds[:, 1]] += \
                (logSoftEv[1, :, :] - logSoftEv[0, :, :])

            np.exp(resp, out=resp)

            resp /= np.sum(resp, axis=(2, 3))[:, :, np.newaxis, np.newaxis]
            resp[np.diag_indices(Data.nNodes)[0], Data.nodes, :, :] = 0

            LP['Count1'] = np.sum(
                resp[
                    Data.respInds[
                        :, 0], Data.respInds[
                        :, 1]], axis=0)
            LP['Count0'] = np.sum(resp, axis=(0, 1)) - LP['Count1']
            LP['resp'] = resp.reshape((Data.nNodes * Data.nNodesTotal, K, K))
            LP['squareResp'] = resp

        else:
            logSoftEv = LP['E_log_soft_ev']  # E x K x K
            logSoftEv[np.where(Data.sourceID == Data.destID), :, :] = 0
            logSoftEv = np.reshape(logSoftEv, (N, N, K, K))

            resp = np.zeros((N, N, K, K))
            # resp[i,j,l,m] = ElogPi[i,l] + ElogPi[j,m] + logSoftEv[i,j,l,m]
            resp = ElogPi[:, np.newaxis, :, np.newaxis] + \
                ElogPi[np.newaxis, :, np.newaxis, :] + logSoftEv
            np.exp(resp, out=resp)
            resp /= np.sum(resp, axis=(2, 3))[:, :, np.newaxis, np.newaxis]
            resp[np.diag_indices(N)] = 0
            LP['squareResp'] = resp
            LP['resp'] = resp.reshape((N**2, K, K))

        return LP

    # Suff Stats
    #########################################################

    def get_global_suff_stats(self, Data, LP, doPrecompEntropy=None, **kwargs):

        if 'resp' in LP:
            K = LP['resp'].shape[-1]
        else:
            K = LP['sumSource'].shape[1]
        N = Data.nNodesTotal
        SS = SuffStatBag(K=K, D=Data.dim, nNodes=N)

        # sumSource[i,l] = \sum_j E_q[s_{ijl}]
        if 'sumSource' in LP:
            sumSource = LP['sumSource']
        else:
            sumSource = np.zeros((N, K))
            sumSource[Data.nodes, :] = np.sum(LP['squareResp'], axis=(1, 3))
        SS.setField('sumSource', sumSource, dims=('nNodes', 'K'))

        # sumReceiver[i,l] = \sum_j E_q[r_{jil}]
        if 'sumReceiver' in LP:
            sumReceiver = LP['sumReceiver']
        else:
            sumReceiver = np.zeros((N, K))
            sumReceiver = np.sum(LP['squareResp'], axis=(0, 2))
        SS.setField('sumReceiver', sumReceiver, dims=('nNodes', 'K'))

        if 'resp' in LP:
            Npair = np.sum(LP['squareResp'], axis=(0, 1))
        else:
            Npair = LP['Count1'] + LP['Count0']
        SS.setField('N', Npair, dims=('K', 'K'))
        SS.setField('Npair', Npair, dims=('K', 'K'))

        if doPrecompEntropy is not None:
            entropy = self.elbo_entropy(LP, Data)
            SS.setELBOTerm('Elogqz', entropy, dims=None)

        from IPython import embed
        embed()

        return SS

    def forceSSInBounds(self, SS):
        ''' Force SS.respPairSums and firstStateResp to be >= 0.  This avoids
            numerical issues in moVB (where SS "chunks" are added and subtracted)
              such as:
                x = 10
                x += 1e-15
                x -= 10
                x -= 1e-15
              resulting in x < 0.

              Returns
              -------
              Nothing.  SS is updated in-place.
        '''
        np.maximum(SS.sumSource, 0, out=SS.sumSource)
        np.maximum(SS.sumReceiver, 0, out=SS.sumReceiver)

    # Global Params
    #########################################################
    def update_global_params_VB(self, SS, **kwargs):
        self.theta = self.alpha + SS.sumSource + SS.sumReceiver
        self.calc_estZ()

    def calc_estZ(self):
        self.estZ = np.argmax(self.theta, axis=1)

    def init_global_params(self, Data, K=0, initname=None, **kwargs):
        N = Data.nNodes
        self.K = K
        if initname == 'kmeansRelational':
            X = np.reshape(Data.X, (N, N))
            centroids, labels = kmeans2(data=X, k=K, minit='points')
            self.theta = np.ones((N, K)) * self.alpha
            for n in xrange(N):
                self.theta[n, labels[n]] += N - 1
        elif initname.count('prior') > 0:
            np.random.seed(123)
            self.theta = np.random.gamma(
                shape=5,
                scale=2,
                size=(
                    Data.nNodes,
                    K))
        else:
            if self.inferType == 'EM':
                pass
            else:
                self.theta = self.alpha + np.ones((Data.nNodes, K))

    def set_global_params(self, hmodel=None, K=None, **kwargs):
        if hmodel is not None:
            self.K = hmodel.allocModel.K
            if self.inferType == 'EM':
                raise NotImplemetedError(
                    'EM not implemented (yet) for FiniteMMSB')
            elif self.inferType.count('VB') > 0:
                self.theta = hmodel.allocModel.theta

        else:
            self.K = K
            if self.inferType == 'EM':
                raise NotImplemetedError(
                    'EM not implemented (yet) for FiniteMMSB')
            elif self.inferType.count('VB') > 0:
                self.theta = theta

    # Evidence
    #########################################################
    def calc_evidence(self, Data, SS, LP, **kwargs):
        alloc = self.elbo_alloc_no_slack(Data, LP, SS)
        if SS.hasELBOTerm('Elogqz'):
            entropy = SS.getELBOTerm('Elogqz')
        else:
            entropy = self.elbo_entropy(LP, Data)
        return alloc + entropy

    def elbo_alloc_no_slack(self, Data, LP, SS):
        N = SS.sumSource.shape[0]
        K = self.K
        p_cDir = N * (gammaln(K * self.alpha) - K * gammaln(self.alpha))
        q_cDir = np.sum(gammaln(np.sum(self.theta, axis=1))) - \
            np.sum(gammaln(self.theta))
        return p_cDir - q_cDir

    def elbo_alloc_slack(self, Data, LP, SS):
        ElogPi = digamma(self.theta) - \
            digamma(np.sum(self.theta, axis=1))[:, np.newaxis]
        term = np.sum((SS.sumSource + SS.sumReceiver) * ElogPi)
        term += np.sum((self.alpha - 1) * ElogPi)
        term -= np.sum((self.theta - 1) * ElogPi)
        return term

    def elbo_entropy(self, LP, Data):
        if LP is not None and 'entropy' in LP:
            return -np.sum(LP['entropy'])
        return -np.sum(LP['resp'] * np.log(LP['resp'] + EPS))

    # IO Utils
    # for machines
    def to_dict(self):
        return dict(theta=self.theta, estZ=self.estZ)

    def from_dict(self, myDict):
        self.inferType = myDict['inferType']
        self.K = myDict['K']
        self.theta = myDict['theta']

    def get_prior_dict(self):
        return dict(alpha=self.alpha)
