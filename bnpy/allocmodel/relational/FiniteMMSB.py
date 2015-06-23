'''
FiniteMMSB.py
'''

import numpy as np
from scipy.cluster.vq import kmeans2

from bnpy.allocmodel import AllocModel
from bnpy.suffstats import SuffStatBag
from bnpy.util import gammaln, digamma, EPS
from bnpy.allocmodel.topics.HDPTopicUtil import c_Dir


class FiniteMMSB(AllocModel):

    """ Mixed membership stochastic block model, with K components.

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
    TODO
    """

    def __init__(self, inferType, priorDict=dict()):
        if inferType.count('EM') > 0:
            raise NotImplementedError(
                'EM not implemented for FiniteMMSB (yet)')

        self.inferType = inferType
        self.set_prior(**priorDict)
        self.K = 0

        # Variational parameter for pi
        self.theta = None

    def set_prior(self, alpha=.1):
        self.alpha = float(alpha)

    def get_active_comp_probs(self):
        print 'TODO'

    def getSSDims(self):
        ''' Get dimensions of interactions between components.

        Overrides default of ('K',), as we need E_log_soft_ev to be
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

    def calc_local_params(self, Data, LP, **kwargs):
        ''' Compute local parameters for provided dataset.

        Args
        -------
        Data : GraphData object
        LP : dict of local params, with fields
            * E_log_soft_ev : nEdges x K x K
        
        Returns
        -------
        LP : dict of local params, with fields
            * resp : nEdges x K x K
                resp[e,j,k] = prob that edge e is explained by 
                connection from state/block j to block k
        '''
        if self.inferType.count('EM') > 0:
            raise NotImplementedError("TODO")

        K = self.K
        ElogPi = digamma(self.theta) - \
            digamma(np.sum(self.theta, axis=1))[:, np.newaxis]

        # resp : nEdges x K x K
        #    resp[e(s,t),k,l] = ElogPi[s,k] + ElogPi[t,l] + likelihood
        resp = ElogPi[Data.edges[:,0], :, np.newaxis] + \
               ElogPi[Data.edges[:,1], np.newaxis, :]

        if Data.isSparse:  # Sparse binary data.
            raise NotImplementedError("TODO")

        else:
            logSoftEv = LP['E_log_soft_ev']  # E x K x K
            resp += logSoftEv

            # In-place exp and normalize
            resp -= np.max(resp, axis=(1,2))[:, np.newaxis]
            np.exp(resp, out=resp)
            resp /= resp.sum(axis=(1,2))[:, np.newaxis]
            LP['resp'] = resp
        return LP

    def get_global_suff_stats(self, Data, LP, doPrecompEntropy=0, **kwargs):
        ''' Compute sufficient stats for provided dataset and local params

        Returns
        -------
        SS : SuffStatBag with K components and fields
            * sumSource : nNodes x K
            * sumReceiver : nNodes x K
        '''
        if 'resp' in LP:
            K = LP['resp'].shape[-1]
        else:
            K = LP['sumSource'].shape[1]

        N = Data.nNodes
        SS = SuffStatBag(K=K, D=Data.dim, N=N)

        # sumSource[i,l] = \sum_j E_q[s_{ijl}]
        if 'sumSource' in LP:
            sumSource = LP['sumSource']
        else:
            srcResp = resp.sum(axis=2)
            # Method A: forloop
            sumSource = np.zeros((Data.nNodes, K))
            for i in xrange(Data.nNodes):
                mask_i = Data.edges[:,0] == i
                sumSource[i,:] = srcResp[mask_i].sum()

            # Method B: sparse matrix multiply
            srcNodeIndicMat = scipy.sparse.csc_matrix(
                (np.ones(Data.nEdges), 
                 Data.edges[:,0],
                 np.arange(Data.nNodes)),
                shape=(Data.nNodes, Data.nEdges))
            sumSource2 = srcNodeIndicMat * srcResp
            assert np.allclose(sumSource, sumSource2)
        SS.setField('sumSource', sumSource, dims=('N', 'K'))

        # sumReceiver[i,l] = \sum_j E_q[r_{jil}]
        if 'sumReceiver' in LP:
            sumReceiver = LP['sumReceiver']
        else:
            rcvResp = resp.sum(axis=1)

            # Method A: forloop
            sumReceiver = np.zeros((Data.nNodes, K))
            for i in xrange(Data.nNodes):
                mask_i = Data.edges[:,1] == i
                sumReceiver[i,:] = srcResp[mask_i].sum()

            # Method B: sparse matrix multiply
            rcvNodeIndicMat = scipy.sparse.csc_matrix(
                (np.ones(Data.nEdges), 
                 Data.edges[:,1],
                 np.arange(Data.nNodes)),
                shape=(Data.nNodes, Data.nEdges))
            sumReceiver2 = rcvNodeIndicMat * rcvResp
            assert np.allclose(sumReceiver, sumReceiver2)
        SS.setField('sumReceiver', sumReceiver, dims=('N', 'K'))

        if 'resp' in LP:
            Npair = np.sum(LP['squareResp'], axis=(0, 1))
        else:
            Npair = LP['Count1'] + LP['Count0']
        SS.setField('N', Npair, dims=('K', 'K'))

        if doPrecompEntropy:
            Hresp = self.L_entropy(LP)
            SS.setELBOTerm('Hresp', Hresp, dims=('K', 'K'))
        return SS

    def forceSSInBounds(self, SS):
        ''' Force certain fields in bounds, to avoid numerical issues.

        Returns
        -------
        Nothing.  SS is updated in-place.
        '''
        np.maximum(SS.sumSource, 0, out=SS.sumSource)
        np.maximum(SS.sumReceiver, 0, out=SS.sumReceiver)

    def update_global_params_VB(self, SS, **kwargs):
        ''' Update global parameter theta to optimize VB objective.

        Post condition
        --------------
        Attribute theta set to optimal value given suff stats.
        '''
        self.theta = self.alpha + SS.sumSource + SS.sumReceiver

    def set_global_params(self, hmodel=None, theta=None, **kwargs):
        ''' Set global parameters to specific values.

        Post condition
        --------------
        Attributes theta, K set to provided values.
        '''
        if hmodel is not None:
            self.K = hmodel.allocModel.K
            if self.inferType == 'EM':
                raise NotImplemetedError(
                    'EM not implemented (yet) for FiniteMMSB')
            elif self.inferType.count('VB') > 0:
                self.theta = hmodel.allocModel.theta
        else:
            if self.inferType == 'EM':
                raise NotImplemetedError(
                    'EM not implemented (yet) for FiniteMMSB')
            elif self.inferType.count('VB') > 0:
                self.theta = theta
                self.K = theta.shape[-1]

    def init_global_params(self, Data, K=0, **kwargs):
        ''' Initialize global parameters "from scratch" to reasonable values.

        Post condition
        --------------
        Attributes theta, K set to reasonable values.
        '''
        self.K = K
        self.theta = self.alpha * np.ones((Data.nNodes, K))


    def calc_evidence(self, Data, SS, LP, **kwargs):
        ''' Compute training objective function on provided input.

        Returns
        -------
        L : scalar float
        '''
        Lalloc = self.L_alloc_no_slack()
        Lslack = self.L_slack(SS)
        if SS.hasELBOTerm('Hresp'):
            Lentropy = SS.getELBOTerm('Hresp').sum()
        else:
            Lentropy = self.L_entropy(LP)
        if todict:
            return dict(Lentropy=Lentropy, Lalloc=Lalloc, Lslack=Lslack)
        return Lalloc + Lentropy + Lslack

    def L_alloc_no_slack(self):
        ''' Compute allocation term of objective function, without slack term

        Returns
        -------
        L : scalar float
        '''
        N = self.theta.shape[0]
        K = self.K
        prior_cDir = N * (gammaln(K * self.alpha) - K * gammaln(self.alpha))
        post_cDir = np.sum(gammaln(np.sum(self.theta, axis=1))) - \
            np.sum(gammaln(self.theta))
        return prior_cDir - post_cDir

    def L_slack(self, SS):
        ''' Compute slack term of the allocation objective function.

        Returns
        -------
        L : scalar float
        '''
        ElogPi = digamma(self.theta) - \
            digamma(np.sum(self.theta, axis=1))[:, np.newaxis]
        Q = SS.sumSource + SS.sumReceiver + self.alpha - self.theta
        Lslack = np.sum(Q * ElogPi)
        return Lslack

    def L_entropy(self, LP):
        return -1 * np.sum(LP['resp'] * np.log(LP['resp'] + EPS))

    def to_dict(self):
        return dict(theta=self.theta)

    def from_dict(self, myDict):
        self.inferType = myDict['inferType']
        self.K = myDict['K']
        self.theta = myDict['theta']

    def get_prior_dict(self):
        return dict(alpha=self.alpha)


    def calc_estZ(self):
        ''' Calculate hard assignment for each node.
        
        Returns
        -------
        Z : 1D array, size nNodes
            indicator for which cluster each node most belongs to
        '''
        return np.argmax(self.theta, axis=1)

