import numpy as np
from scipy.cluster.vq import kmeans2

from bnpy.allocmodel import AllocModel
from bnpy.suffstats import SuffStatBag
from bnpy.util import gammaln, digamma, EPS

from bnpy.util import StickBreakUtil
from bnpy.allocmodel.topics import OptimizerRhoOmega
from bnpy.allocmodel.topics.HDPTopicModel import c_Beta, c_Dir
import MMSBUtil


class HDPaMMSB(AllocModel):
    # Constructors
    #########################################################

    def __init__(self, inferType, priorDict=dict()):
        if inferType == 'EM':
            raise ValueError('EM is not supported for HDPaMMSB')

        self.set_prior(**priorDict)
        self.inferType = inferType
        self.K = 0

        # Variational free parameters
        self.rho = None  # rho/omega define Beta distr. on global stick lengths
        self.omega = None
        # Nx(K+1) Dirichlet params for community distributions pi
        self.theta = None

        # Cached stats. about q(pi)
        self.ELogPi = None
        self.expELogPi = None
        self.sumExpELogPi = None

        # Biggest absolute datapoint, needed for real-valued observations
        self.M = np.inf

        # Debug / testing stuff
        self.edgePrPi = []
        self.precision = []
        self.recall = []

    def set_prior(self, gamma=10, alpha=0.5, epsilon=0.05,
                  delta=.01, **kwargs):
        # HDP hyperparameters
        self.gamma = gamma
        self.alpha = alpha

        # Used for assortativity, delta only for assortative real-valued
        # likelihoods
        self.epsilon = epsilon
        self.delta = delta

    def get_active_comp_probs(self):
        ''' Return K vector of appearance probabilities for each of the K comps
        '''
        return StickBreakUtil.rho2beta_active(self.rho)

    # Local Params
    #########################################################
    def calc_local_params(self, Data, LP, MergePrepInfo=None, **kwargs):
        N = Data.nNodes
        K = self.K
        D = Data.dim

        self._update_Epi_stats(N=Data.nNodesTotal, K=self.K)

        if Data.isSparse:
            LP = self._calc_local_params_sparse(
                Data,
                LP,
                MergePrepInfo,
                **kwargs)
        else:
            LP = self._calc_local_params_nonsparse(
                Data,
                LP,
                MergePrepInfo,
                **kwargs)
        return LP

    def _calc_local_params_nonsparse(self, Data, LP, MergePrepInfo=None,
                                     **kwargs):
        '''
        Calculates the local params when using a real-valued dataset
          (e.g. when using gaussian emissions)
        Only calculates the diagonal of the resp matrix, \eta_{ijkk} and the
          normalization constants Z_{ij}
        '''
        K = self.K
        D = Data.dim
        if self.M == np.inf:
            self.M = np.max(Data.X)

        # Find where obs. are inside/outside interval [-delta, delta]
        if not hasattr(Data, 'assortativeMask'):
            Data.makeAssortativeMask(self.delta)
        mask = Data.assortativeMask

        assert(
            LP['E_log_soft_ev'].shape == (
                Data.nNodesTotal *
                Data.nNodes,
                K))
        LP['evSquare'] = \
            LP['E_log_soft_ev'].reshape((Data.nNodes, Data.nNodesTotal, K))
        logSoftEv = LP['evSquare']
        expLogSoftEv = np.exp(logSoftEv)

        resp = np.zeros((Data.nNodes, Data.nNodesTotal, K))
        resp += logSoftEv
        resp += self.ELogPi[Data.nodes, np.newaxis, :] + \
            self.ELogPi[np.newaxis, :, :]
        np.exp(resp, out=resp)

        normConst = np.einsum('ik,jk,ijk->ij',
                              self.expELogPi[Data.nodes, :],
                              self.expELogPi,
                              expLogSoftEv - (mask * (1 - self.epsilon) / (2 * self.M) +
                                              (1 - mask) * self.epsilon / (2 * self.M))[:, :, np.newaxis]
                              )
        normConst += self.sumExpELogPi[Data.nodes, np.newaxis] * \
            self.sumExpELogPi[np.newaxis, :] * \
            (mask * (1 - self.epsilon) / (2 * self.M) +
             (1 - mask) * self.epsilon / (2 * self.M))
        '''
    # Calculate the normalization constant Z_{ij} for each resp_{ij}
    ## sum_k \tilde\pi_{ik} \tilde\pi_{jk} (f(w_k,x_{ij}) - f(\eps,x_{ij}))
    normConst = np.einsum('ik,jk,k->ij',
                          self.expELogPi[Data.nodes,:],
                          self.expELogPi,
                          expLogSoftEv[0,:]-epsilonEv[0])
    normConst[respI[:,0],respI[:,1]] = \
                         np.einsum('ik,ik,k->i',self.expELogPi[respI[:,0]],
                                   self.expELogPi[respI[:,1]],
                                   expLogSoftEv[1,:]-epsilonEv[1])
    normConst[respI[:,0],respI[:,1]] = \
                         np.sum(self.expELogPi[Data.nodes[respI[:,0]]]*
                                self.expELogPi[respI[:,1]]*
                                (expLogSoftEv[1,:]-epsilonEv[1]),
                                axis=1)

    ## += \tilde\pi_{i} \tilde\pi_{j} f(\eps,x_{ij})
    normConst += \
                 self.sumExpELogPi[Data.nodes,np.newaxis] * \
                 self.sumExpELogPi[np.newaxis,:] * \
                 epsilonEv[0]
    normConst[respI[:,0],respI[:,1]] -= \
                               self.sumExpELogPi[Data.nodes[respI[:,0]]] * \
                               self.sumExpELogPi[respI[:,1]] * \
                               epsilonEv[0]
    normConst[respI[:,0],respI[:,1]] += \
                               self.sumExpELogPi[Data.nodes[respI[:,0]]] * \
                               self.sumExpELogPi[respI[:,1]] * \
                               epsilonEv[1]
    '''
        resp /= normConst[:, :, np.newaxis]
        normConst[np.arange(Data.nNodes), Data.nodes] = np.inf
        resp[np.arange(Data.nNodes), Data.nodes, :] = 0.0

        diag = np.diag_indices(K)
        LP['normConst'] = normConst
        LP['respSquare'] = resp
        LP['resp'] = np.reshape(resp, (Data.nNodes * Data.nNodesTotal, K))

        # LP['fullResp'] = MMSBUtil.calc_full_resp_nonsparse(Data, self.ELogPi,
        #                                                   logSoftEv, self.K,
        #                                                   self.epsilon, self.M)
        #LP['fullResp'] /= normConst[:,:,np.newaxis,np.newaxis]

        return LP

    def _calc_local_params_sparse(
            self, Data, LP, MergePrepInfo=None, **kwargs):
        K = self.K
        respI = Data.respInds
        logSoftEv = LP['E_log_soft_ev']
        expLogSoftEv = np.exp(logSoftEv)
        epsilonEv = np.zeros(2)
        epsilonEv[0] = 1 - self.epsilon
        epsilonEv[1] = self.epsilon
        assert logSoftEv.shape == (2, K)

        resp = self.ELogPi[Data.nodes, np.newaxis, :] + \
            self.ELogPi[np.newaxis, :, :] + logSoftEv[0, :]
        resp[respI[:, 0], respI[:, 1]] += \
            (logSoftEv[1, :] - logSoftEv[0, :])
        np.exp(resp, out=resp)

        # Calculate the normalization constant Z_{ij} for each resp_{ij}
        # sum_k \tilde\pi_{ik} \tilde\pi_{jk} (f(w_k,x_{ij}) - f(\eps,x_{ij}))
        normConst = np.einsum('ik,jk,k->ij',
                              self.expELogPi[Data.nodes, :],
                              self.expELogPi,
                              expLogSoftEv[0, :] - epsilonEv[0])
        normConst[respI[:, 0], respI[:, 1]] = \
            np.einsum('ik,ik,k->i', self.expELogPi[respI[:, 0]],
                      self.expELogPi[respI[:, 1]],
                      expLogSoftEv[1, :] - epsilonEv[1])
        normConst[respI[:, 0], respI[:, 1]] = \
            np.sum(self.expELogPi[Data.nodes[respI[:, 0]]] *
                   self.expELogPi[respI[:, 1]] *
                   (expLogSoftEv[1, :] - epsilonEv[1]),
                   axis=1)

        # += \tilde\pi_{i} \tilde\pi_{j} f(\eps,x_{ij})
        normConst += \
            self.sumExpELogPi[Data.nodes, np.newaxis] * \
            self.sumExpELogPi[np.newaxis, :] * \
            epsilonEv[0]
        normConst[respI[:, 0], respI[:, 1]] -= \
            self.sumExpELogPi[Data.nodes[respI[:, 0]]] * \
            self.sumExpELogPi[respI[:, 1]] * \
            epsilonEv[0]
        normConst[respI[:, 0], respI[:, 1]] += \
            self.sumExpELogPi[Data.nodes[respI[:, 0]]] * \
            self.sumExpELogPi[respI[:, 1]] * \
            epsilonEv[1]

        resp /= normConst[:, :, np.newaxis]

        # Do the right thing for diagonal entries of resp and Z
        diag = np.diag_indices(Data.nNodes)
        normConst[np.arange(Data.nNodes), Data.nodes] = np.inf
        resp[np.arange(Data.nNodes), Data.nodes] = 0

        LP['normConst'] = normConst
        LP['resp'] = resp

        # LP['fullResp'] = MMSBUtil.calc_full_resp(Data, self.ELogPi, logSoftEv,
        #                                         epsilonEv, self.K)
        #LP['fullResp'] /= normConst[:,:,np.newaxis,np.newaxis]

        if Data.heldOut is not None:
            LP['resp'][Data.heldOut[0], Data.heldOut[1]] = 0.0
            LP['normConst'][Data.heldOut[0], Data.heldOut[1]] = np.inf

        # Using sparse data, so the obsmodel needs us to comp. counts here
        LP['Count1'] = np.sum(
            resp[
                Data.respInds[
                    :, 0], Data.respInds[
                    :, 1]], axis=0)
        LP['Count0'] = np.sum(resp, axis=(0, 1)) - LP['Count1']

        return LP

    def _update_Epi_stats(self, N, K):
        '''
        Recomputes various statistics about pi needed for updates and ELBO
          evaluation.
        '''
        if self.expELogPi is None or self.expELogPi.shape != (N, K):
            self.expELogPi = np.zeros((N, K))
        if self.sumExpELogPi is None or self.sumExpELogPi.shape != (N):
            self.sumExpELogPi = np.zeros(N)
        if self.ELogPi is None or self.ELogPi.shape != (N, K):
            self.ELogPi = np.zeros((N, K))

        self.ELogPi = digamma(self.theta[:, 0:K]) - \
            digamma(np.sum(self.theta[:, 0:K], axis=1))[:, np.newaxis]
        np.exp(self.ELogPi, out=self.expELogPi)
        np.sum(self.expELogPi, axis=1, out=self.sumExpELogPi)

    def initLPFromTruth(self, Data):
        K = np.max(Data.TrueParams['Z']) + 1
        N = Data.nNodes
        Z = Data.TrueParams['Z']
        self.theta = np.zeros((N, K)) + self.alpha
        resp = np.zeros((N, N, K))
        squareResp = np.zeros((N, N, K, K))
        for i in xrange(N):
            for j in xrange(N):
                if Z[i, j, 0] == Z[i, j, 1]:
                    resp[i, j, Z[i, j, 0]] += 1
                squareResp[i, j, Z[i, j, 0], Z[i, j, 1]] = 1

                self.theta[i, Z[i, j, 0]] += 1
                self.theta[i, Z[j, i, 1]] += 1
        diag = np.diag_indices(N)
        resp[diag[0], diag[1], :] = 0
        squareResp[diag[0], diag[1], :, :] = 0
        squareResp = resp
        normConst = np.ones((N, N))
        normConst[diag[0], diag[1]] = np.inf
        LP = {'resp': resp, 'normConst': normConst}
        self._update_Epi_stats(N, K)

        if Data.isSparse:
            LP['Count1'] = np.sum(squareResp[Data.respInds[:, 0],
                                             Data.respInds[:, 1]], axis=0)
            LP['Count0'] = np.sum(squareResp, axis=(0, 1)) - LP['Count1']

        return LP

    # Suff Stats
    #########################################################
    def get_global_suff_stats(self, Data, LP, doPrecompEntropy=None, **kwargs):
        nNodes = Data.nNodesTotal
        inds = Data.nodes
        K = LP['resp'].shape[-1]
        self.K = K

        '''
    sumSource = np.sum(LP['fullResp'], axis=(1,3))
    sumReceiver = np.sum(LP['fullResp'], axis=(0,2))
    SS = SuffStatBag(K=K, D=Data.dim, nNodes=nNodes)
    SS.setField('sumSource', sumSource, dims=('nNodes','K'))
    SS.setField('sumReceiver', sumReceiver, dims=('nNodes','K'))
    SS.setField('N', np.sum(LP['resp'], axis=(0,)), dims=('K'))

    #print SS.N
    #print np.sum(SS.N)
    return SS
    '''

        if Data.isSparse:
            SS = self._get_global_suff_stats_sparse(Data, LP)
        else:
            SS = self._get_global_suff_stats_nonsparse(Data, LP)

        if doPrecompEntropy is not None:
            entropy = MMSBUtil.assortative_elbo_entropy(LP, SS, Data,
                                                        Data.nNodesTotal, K,
                                                        self.ELogPi, self.expELogPi,
                                                        self.epsilon, self.M)

            if Data.isSparse:
                extraObsModelTerm = \
                    MMSBUtil.assortative_elbo_extraObsModel_sparse(Data,
                                                                   LP,
                                                                   self.epsilon)
            else:
                extraObsModelTerm = \
                    MMSBUtil.assortative_elbo_extraObsModel_nonsparse(Data,
                                                                      LP,
                                                                      self.epsilon,
                                                                      self.M)

            SS.setELBOTerm('Elogqz', entropy, dims=None)
            SS.setELBOTerm('ExtraObsModel', extraObsModelTerm, dims=None)

        #assert np.allclose(SS.sumSource, np.sum(LP['fullResp'], axis=(1,3)))
        #assert np.allclose(SS.sumReceiver, np.sum(LP['fullResp'], axis=(0,2)))

        return SS

    def _get_global_suff_stats_nonsparse(self, Data, LP):
        if not hasattr(Data, 'assortativeMask'):
            Data.makeAssortativeMask(self.delta)
        mask = Data.assortativeMask

        nNodes = Data.nNodesTotal
        inds = Data.nodes
        K = LP['resp'].shape[-1]
        self.K = K

        # Calculate sumSource[i,l] = \sum_j s_{ijl} = \sum_j \sum_m resp_{ijlm}
        scratch = np.ones((Data.nNodes, Data.nNodesTotal, K))
        scratch[:, :, :] = (mask * ((1 - self.epsilon) / (2 * self.M)) +
                            (1 - mask) * (self.epsilon / (2 * self.M)))[:, :, np.newaxis]
        scratch /= LP['normConst'][:, :, np.newaxis]

        scratch *= \
            (self.sumExpELogPi[:, np.newaxis] -
             self.expELogPi)[np.newaxis, :, :]
        scratch *= self.expELogPi[inds][:, np.newaxis, :]
        sumSource = np.zeros((Data.nNodesTotal, K))
        sumSource[inds] += np.sum(scratch, axis=1)
        sumSource[inds] += np.sum(LP['respSquare'], axis=1)

        # Calculate sumReceiver[i,l] = \sum_j r_{jil} = \sum_j \sum_m
        # resp_{jiml}
        scratch[:, :, :] = (mask * ((1 - self.epsilon) / (2 * self.M)) +
                            (1 - mask) * (self.epsilon / (2 * self.M)))[:, :, np.newaxis]
        scratch /= LP['normConst'][:, :, np.newaxis]
        scratch *= \
            (self.sumExpELogPi[inds, np.newaxis] -
             self.expELogPi[inds, :])[:, np.newaxis, :]
        scratch *= self.expELogPi[np.newaxis, :]
        scratch += LP['respSquare']
        sumReceiver = np.sum(scratch, axis=0)

        SS = SuffStatBag(K=K, D=Data.dim, nNodes=nNodes)
        SS.setField('sumSource', sumSource, dims=('nNodes', 'K'))
        SS.setField('sumReceiver', sumReceiver, dims=('nNodes', 'K'))
        SS.setField('N', np.sum(LP['resp'], axis=0), dims=('K',))

        return SS

    def _get_global_suff_stats_sparse(self, Data, LP):
        nNodes = Data.nNodesTotal
        inds = Data.nodes
        K = LP['resp'].shape[-1]
        self.K = K

        # Calculate sumSource[i,l] = \sum_j s_{ijl} = \sum_j \sum_m resp_{ijlm}
        scratch = np.ones((Data.nNodes, Data.nNodesTotal, K))
        scratch *= (1 - self.epsilon) / LP['normConst'][:, :, np.newaxis]
        scratch[Data.respInds[:, 0], Data.respInds[:, 1]] *= \
            (self.epsilon / (1 - self.epsilon))
        scratch *= \
            (self.sumExpELogPi[:, np.newaxis] -
             self.expELogPi)[np.newaxis, :, :]
        scratch *= self.expELogPi[inds][:, np.newaxis, :]
        sumSource = np.zeros((Data.nNodesTotal, K))
        sumSource[inds] += np.sum(scratch, axis=1)
        sumSource[inds] += np.sum(LP['resp'], axis=1)

        # Calculate sumReceiver[i,l] = \sum_j r_{jil} = \sum_j \sum_m
        # resp_{jiml}
        scratch.fill(1)
        scratch *= (1 - self.epsilon) / LP['normConst'][:, :, np.newaxis]
        scratch[Data.respInds[:, 0], Data.respInds[:, 1]] *= \
            (self.epsilon / (1 - self.epsilon))
        scratch *= \
            (self.sumExpELogPi[inds, np.newaxis] -
             self.expELogPi[inds, :])[:, np.newaxis, :]
        scratch *= self.expELogPi[np.newaxis, :]
        scratch += LP['resp']
        sumReceiver = np.sum(scratch, axis=0)

        SS = SuffStatBag(K=K, D=Data.dim, nNodes=nNodes)
        SS.setField('sumSource', sumSource, dims=('nNodes', 'K'))
        SS.setField('sumReceiver', sumReceiver, dims=('nNodes', 'K'))

        # TODO : should this also compute the K x K Npair?
        SS.setField('N', np.sum(LP['resp'], axis=(0, 1)), dims=('K'))

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
        self.K = SS.K

        # We could repeat this cycle of updating theta, then rho/omega, but
        #   hopefully this propagates information well enough
        self.theta = self._calcTheta(SS)
        self.rho, self.omega = self.find_optimum_rhoOmega(
            N=SS.nNodes, **kwargs)
        self.theta = self._calcTheta(SS)

    def _calcTheta(self, SS):
        K = SS.K
        N = SS.nNodes
        if self.rho is None or self.rho.size != K:
            self.rho = OptimizerRhoOmega.create_initrho(K)

        Ebeta = StickBreakUtil.rho2beta(self.rho)
        theta = np.zeros((N, K + 1))
        theta += self.alpha * Ebeta
        theta[:, 0:K] += SS.sumReceiver + SS.sumSource
        return theta

    def find_optimum_rhoOmega(self, N, **kwargs):
        ''' Performs numerical optimization of rho and omega for M-step update.

            Note that the optimizer forces rho to be in [EPS, 1-EPS] for
            the sake of numerical stability
        '''
        ELogPi = digamma(self.theta) \
            - digamma(np.sum(self.theta, axis=1))[:, np.newaxis]
        sumLogPi = np.sum(ELogPi, axis=0)

        # Select initial rho, omega values for gradient descent
        if self.rho is not None and self.rho.size == self.K:
            initRho = self.rho
        else:
            initRho = None

        if self.omega is not None and self.omega.size == self.K:
            initOmega = self.omega
        else:
            initOmega = None

        # Do the optimization
        try:
            rho, omega, fofu, Info = \
                OptimizerRhoOmega.find_optimum_multiple_tries(
                    sumLogPi=sumLogPi,
                    nDoc=N,
                    gamma=self.gamma,
                    alpha=self.alpha,
                    initrho=initRho,
                    initomega=initOmega)
            self.OptimizerInfo = Info
            self.OptimizerInfo['fval'] = fofu

        except ValueError as error:
            if hasattr(self, 'rho') and self.rho.size == self.K:
                Log.error(
                    '***** Optim failed. Remain at cur val. ' +
                    str(error))
                rho = self.rho
                omega = self.omega
            else:
                Log.error('***** Optim failed. Set to prior. ' + str(error))
                omega = (self.gamma + 1) * np.ones(SS.K)
                rho = 1 / float(1 + self.gamma) * np.ones(SS.K)

        return rho, omega

    def init_global_params(self, Data, K=0, initname=None, **kwargs):
        N = Data.nNodesTotal
        self.K = K
        self.rho = OptimizerRhoOmega.create_initrho(K)
        self.omega = (1.0 + self.gamma) * np.ones(K)

        if 'seed' in kwargs:
            np.random.seed(kwargs['seed'])

        if initname.count('prior') > 0:
            self.theta = np.random.gamma(shape=5, scale=2,
                                         size=(Data.nNodesTotal, K + 1))

        # Kmeans init with sparse (binary) edges
        elif initname == 'kmeansRelational' and Data.isSparse:
            X = np.zeros((N, N))
            for e in Data.edgeSet:
                X[e[0], e[1]] = 1
            centroids, labels = kmeans2(data=X, k=K, minit='points')

            Ebeta = StickBreakUtil.rho2beta(self.rho)
            self.theta = np.ones((N, K + 1)) * self.alpha * Ebeta
            for n in xrange(N):
                self.theta[n, labels[n]] += N - 1

        # Kmeans init with nonsparse (e.g. gaussian) edges
        elif initname == 'kmeansRelational' and not Data.isSparse:

            if Data.X is None:
                X = np.zeros((N, N))
                for e in Data.edgeSet:
                    X[e[0], e[1]] = 1
                centroids, labels = kmeans2(data=X, k=K, minit='points')
            else:
                inds = np.where(Data.X > self.delta)[0]
                X = np.squeeze(Data.X[inds, :])
                centroids, labels = kmeans2(data=X, k=K, minit='random')
                X = np.ones(N**2, dtype=np.int32) * -1
                X[inds] = labels
                X = X.reshape((N, N))

            Ebeta = StickBreakUtil.rho2beta(self.rho)
            self.theta = np.ones((N, K + 1)) * self.alpha * Ebeta

            for n in xrange(N):
                binz = np.bincount(X[n, :] + 1)
                if len(binz) == 1:
                    community = -1
                else:
                    community = np.argmax(binz[1:]) - 1
                if community == -1:
                    community = np.random.randint(low=0, high=K)

                self.theta[n, community] += N - 1

    def set_global_params(self, hmodel=None,
                          rho=None, omega=None,
                          theta=None, **kwargs):
        if hmodel is not None:
            self.K = hmodel.allocModel.K
            self.rho = hmodel.allocModel.rho
            self.omega = hmodel.allocModel.omega
            self.theta = hmodel.allocModel.theta

        elif rho is not None \
                and omega is not None \
                and theta is not None:
            self.rho = rho
            self.omega = omega
            self.theta = theta
            self.K = omega.size
        else:
            raise Exception('No global parameters provided')

    # Evidence
    #########################################################
    def calc_evidence(self, Data, SS, LP, **kwargs):
        HDP = self.elbo_HDP(self.rho, self.omega, self.theta,
                            self.alpha, self.gamma, SS.nNodes)
        if SS.hasELBOTerm('Elogqz'):
            entropy = SS.getELBOTerm('Elogqz')
        else:
            entropy = MMSBUtil.assortative_elbo_entropy(LP, SS, Data,
                                                        Data.nNodesTotal, self.K,
                                                        self.ELogPi, self.expELogPi,
                                                        self.epsilon, self.M)

        if SS.hasELBOTerm('ExtraObsModel'):
            extraObsModelTerm = SS.getELBOTerm('ExtraObsModel')
        elif Data.isSparse:
            extraObsModelTerm = \
                MMSBUtil.assortative_elbo_extraObsModel_sparse(Data, LP,
                                                               self.epsilon)
        else:
            extraObsModelTerm = \
                MMSBUtil.assortative_elbo_extraObsModel_nonsparse(Data, LP,
                                                                  self.epsilon,
                                                                  self.M)

        return HDP + entropy + extraObsModelTerm

    def elbo_HDP(self, rho, omega, theta, alpha, gamma, N):
        '''
        HDP-related terms of the elbo, E_q[log p(u|\gamma) - log q(u) +
                                           lo p(pi|u,alpha) - log q(pi)]
        '''
        K = rho.size
        kvec = OptimizerRhoOmega.kvec(K)
        eta1 = rho * omega
        eta0 = (1 - rho) * omega
        digammaOmega = digamma(omega)
        ElogU = digamma(eta1) - digammaOmega
        Elog1mU = digamma(eta0) - digammaOmega

        # Normalization constants of the beta distributions over each u_k
        pU_Norm = K * c_Beta(1.0, gamma)
        qU_Norm = c_Beta(rho * omega, (1 - rho) * omega)

        # Normalization constants of the Dirichlet distributions over each pi_i
        # piPi_Norm is the term c_D(alpha*beta) that needs to be lower-bounded
        pPi_Norm = N * K * np.log(alpha) + N * np.sum(ElogU + kvec * Elog1mU)
        qPi_Norm = c_Dir(theta)

        # "meat" meaning not the normalization constants.  Note the "meat" of
        #   q(pi) and p(pi) cancel out in the alloc_slack term.
        U_Meat = np.sum((1.0 - eta1) * ElogU + (gamma - eta0) * Elog1mU)

        return pU_Norm - qU_Norm + U_Meat + pPi_Norm - qPi_Norm

    # IO Utils
    # for machines
    def to_dict(self):
        # return dict(theta=self.theta, rho=self.rho, omega=self.omega,
        #            edgePrPi=self.edgePrPi)
        return dict(theta=self.theta, rho=self.rho, omega=self.omega,
                    precision=self.precision, recall=self.recall)

    def from_dict(self, myDict):
        self.inferType = myDict['inferType']
        self.K = myDict['K']
        self.theta = myDict['theta']
        self.rho = myDict['rho']
        self.omega = myDict['omega']

    def get_prior_dict(self):
        return dict(alpha=self.alpha, gamma=self.gamma,
                    epsilon=self.epsilon, delta=self.delta, M=self.M)

    def evaluateHeldOut(self, Data, SS, obsModel):
        K = self.K
        if self.edgePrPi == []:
            self.edgePrPi = np.zeros((len(Data.heldOutSet), 4))
        wDiag = obsModel.Post.lam1 / (obsModel.Post.lam0 + obsModel.Post.lam1)
        wDiag = wDiag.squeeze()
        w = np.ones((K, K)) * self.epsilon
        w[np.diag_indices(K)] = wDiag
        held = Data.heldOutSet
        epi = self.theta[:, :K] / \
            np.sum(self.theta[:, :K], axis=1)[:, np.newaxis]
        epi = epi.squeeze()
        for ee, (i, j) in enumerate(Data.heldOutSet):
            self.edgePrPi[ee, 0] = i
            self.edgePrPi[ee, 1] = j
            self.edgePrPi[ee,
                          2] = np.sum(epi[i,
                                          np.newaxis,
                                          :,
                                          np.newaxis] * epi[np.newaxis,
                                                            j,
                                                            np.newaxis,
                                                            :] * w)
            self.edgePrPi[ee, 3] = (i, j) in Data.edgeSet

        nEvalPoints = 50
        threshs = (np.arange(nEvalPoints) + 1.0) / float(nEvalPoints)
        self.precision, self.recall = self.calcEdgePrPrecisionRecall(self.edgePrPi,
                                                                     threshs)

    def calcEdgePrPrecisionRecall(self, edgePr, threshs):
        numRight = np.zeros((2, len(threshs)))
        wtf = np.zeros((2, len(threshs)))

        numWrong = np.zeros((2, len(threshs)))
        scratch = np.zeros(edgePr.shape[0])
        edgePr[:, 3] = edgePr[:, 3].astype(int)
        for tt, thresh in enumerate(threshs):
            scratch = (edgePr[:, 2] > thresh).astype(int)
            onesRight = np.sum(
                np.logical_and(
                    scratch == edgePr[
                        :, 3], edgePr[
                        :, 3] == 1))
            zerosRight = np.sum(
                np.logical_and(
                    scratch == edgePr[
                        :, 3], edgePr[
                        :, 3] == 0))
            onesWrong = np.sum(
                np.logical_and(
                    scratch != edgePr[
                        :, 3], edgePr[
                        :, 3] == 1))
            zerosWrong = np.sum(
                np.logical_and(
                    scratch != edgePr[
                        :, 3], edgePr[
                        :, 3] == 0))

            numRight[1, tt] = onesRight
            numRight[0, tt] = zerosRight

            numOnes = np.sum(edgePr[:, 3])
            numZeros = len(edgePr[:, 3]) - numOnes
            numWrong[1, tt] = numOnes - numRight[1, tt]
            numWrong[0, tt] = numZeros - numRight[0, tt]

        recall = numRight[1] / numOnes
        precision = (numRight[1] + EPS) / (numRight[1] + numWrong[0] + EPS)
        return precision, recall
