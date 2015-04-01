import numpy as np

from ..AllocModel import AllocModel
from bnpy.suffstats import SuffStatBag
from ...util import digamma, gammaln
from ...util import NumericUtil

from LocalStepManyDocs import calcLocalParams


class FiniteTopicModel(AllocModel):

    '''
    Bayesian topic model with a finite number of components K.

    Attributes
    -------
    inferType : string {'VB', 'moVB', 'soVB'}
        indicates which updates to perform for local/global steps
    K : int
        number of components
    alpha : float
        scalar pseudo-count
        used in Dirichlet prior on document-topic probabilities \pi_d.


    Attributes for VB
    ---------
    None. No global structure exists except scalar parameter gamma.

    Variational Local Parameters
    --------
    resp :  2D array, N x K
        q(z_n) = Categorical( resp_{n1}, ... resp_{nK} )
    theta : 2D array, nDoc x K
        q(pi_d) = Dirichlet( \theta_{d1}, ... \theta_{dK} )

    References
    -------
    Latent Dirichlet Allocation, by Blei, Ng, and Jordan
    introduces a classic topic model with Dirichlet-Mult observations.
    '''

    def __init__(self, inferType, priorDict=None):
        if inferType == 'EM':
            raise ValueError('FiniteTopicModel cannot do EM.')
        self.inferType = inferType
        self.K = 0
        if priorDict is None:
            self.set_prior()
        else:
            self.set_prior(**priorDict)

    def get_keys_for_memoized_local_params(self):
        ''' Return LP field names required for warm starts of local step
        '''
        return ['DocTopicCount']

    def get_active_comp_probs(self):
        ''' Get vector of appearance probabilities for each active comp.

        Returns
        -------
        beta : 1D array, size K
            beta[k] gives probability of comp. k under this model.
        '''
        return np.ones(self.K) / float(self.K)

    def set_prior(self, alpha=1.0, **kwargs):
        self.alpha = float(alpha)

    def to_dict(self):
        return dict()

    def from_dict(self, Dict):
        self.inferType = Dict['inferType']
        self.K = Dict['K']

    def get_prior_dict(self):
        return dict(alpha=self.alpha,
                    K=self.K,
                    inferType=self.inferType)

    def get_info_string(self):
        ''' Returns human-readable name of this object
        '''
        return 'Finite LDA model with K=%d comps. alpha=%.2f' \
            % (self.K, self.alpha)

    def calc_local_params(self, Data, LP, **kwargs):
        ''' Compute local parameters for each data item.

        Parameters
        -------
        Data : bnpy.data.DataObj subclass

        LP : dict
            Local parameters as key-value string/array pairs
            * E_log_soft_ev : 2D array, N x K
                E_log_soft_ev[n,k] = log p(data obs n | comp k)

        Returns
        -------
        LP : dict
            Local parameters, with updated fields
            * resp : 2D array, N x K
                Posterior responsibility each comp has for each item
                resp[n, k] = p(z[n] = k | x[n])
            * theta : 2D array, nDoc x K
                Defines approximate posterior on doc-topic weights.
                q(\pi_d) = Dirichlet(theta[d,0], ... theta[d, K-1])
        '''
        alphaEbeta = self.alpha * np.ones(self.K)
        LP = calcLocalParams(Data, LP, alphaEbeta, **kwargs)
        assert 'resp' in LP
        assert 'theta' in LP
        assert 'DocTopicCount' in LP
        return LP

    def initLPFromResp(self, Data, LP):
        ''' Fill in remaining local parameters given token-topic resp.

        Args
        ----
        LP : dict with fields
            * resp : 2D array, size N x K

        Returns
        -------
        LP : dict with fields
            * DocTopicCount
            * theta
            * ElogPi
        '''
        resp = LP['resp']
        K = resp.shape[1]
        DocTopicCount = np.zeros((Data.nDoc, K))
        for d in xrange(Data.nDoc):
            start = Data.doc_range[d]
            stop = Data.doc_range[d + 1]
            if hasattr(Data, 'word_count'):
                DocTopicCount[d, :] = np.dot(Data.word_count[start:stop],
                                             resp[start:stop, :])
            else:
                DocTopicCount[d, :] = np.sum(resp[start:stop, :], axis=0)

        remMass = np.minimum(0.1, 1.0 / (K * K))
        newEbeta = (1 - remMass) / K
        theta = DocTopicCount + self.alpha * newEbeta
        digammaSumTheta = digamma(theta.sum(axis=1))
        ElogPi = digamma(theta) - digammaSumTheta[:, np.newaxis]

        LP['DocTopicCount'] = DocTopicCount
        LP['theta'] = theta
        LP['ElogPi'] = ElogPi
        return LP

    def get_global_suff_stats(self, Data, LP, 
                              doPrecompEntropy=None, 
                              cslice=(0, None), **kwargs):
        ''' Calculate sufficient statistics for global updates.

        Parameters
        -------
        Data : bnpy data object
        LP : local param dict with fields
            resp : Data.nObs x K array,
                where resp[n,k] = posterior resp of comp k
        doPrecompEntropy : boolean flag
            indicates whether to precompute ELBO terms in advance
            used for memoized learning algorithms (moVB)

        Returns
        -------
        SS : SuffStatBag with K components
            Relevant fields
            * nDoc : scalar float
                Counts total documents available in provided data.

            Also has optional ELBO field when precompELBO is True
            * Hvec : 1D array, size K
                Vector of entropy contributions from each comp.
                Hvec[k] = \sum_{n=1}^N H[q(z_n)], a function of 'resp'
        '''
        resp = LP['resp']
        _, K = resp.shape

        SS = SuffStatBag(K=K, D=Data.get_dim())
        if cslice[1] is None:
            SS.setField('nDoc', Data.nDoc, dims=None)
        else:
            SS.setField('nDoc', cslice[1] - cslice[0], dims=None)
        if doPrecompEntropy:
            Hvec = self.L_entropy(Data, LP, returnVector=1)
            Lalloc = self.L_alloc(Data, LP)
            SS.setELBOTerm('Hvec', Hvec, dims='K')
            SS.setELBOTerm('L_alloc', Lalloc, dims=None)
        return SS

    def update_global_params(self, SS, rho=None, **kwargs):
        ''' Update global parameters to optimize the ELBO objective.
        '''
        self.K = SS.K

    def set_global_params(self, K=0, **kwargs):
        """ Set global parameters to provided values.
        """
        self.K = K

    def init_global_params(self, Data, K=0, **kwargs):
        """ Initialize global parameters to provided values.
        """
        self.K = K

    def calc_evidence(self, Data, SS, LP, **kwargs):
        """ Calculate ELBO objective function value for provided state.

        Returns
        -------
        L : float
            Represents sum of all terms in ELBO objective.
        """
        if SS.hasELBOTerms():
            Lentropy = SS.getELBOTerm('Hvec').sum()
            Lalloc = SS.getELBOTerm('L_alloc')
        else:
            Lentropy = self.L_entropy(Data, LP, returnVector=0)
            Lalloc = self.L_alloc(Data, LP)
        if SS.hasAmpFactor():
            Lentropy *= SS.ampF
            Lalloc *= SS.ampF
        return Lalloc + Lentropy

    def L_entropy(self, Data, LP, returnVector=1):
        ''' Calculate assignment entropy term of the ELBO objective.

        Returns
        -------
        Hvec : 1D array, size K
            Hvec[k] = \sum_{n=1}^N H[q(z_n)]
        '''
        return L_entropy(Data, LP, returnVector=returnVector)

    def L_alloc(self, Data, LP):
        ''' Calculate allocation term of the ELBO objective.

        Returns
        -------
        L_alloc : scalar float
        '''
        return L_alloc(Data=Data, LP=LP, alpha=self.alpha)

    def getSerializableArgsForLocalStep(self):
        """ Get compact dict of params for local step.

        Returns
        -------
        Info : dict
        """
        return dict(inferType=self.inferType, 
                    K=self.K, 
                    alpha=self.alpha)

    def fillSharedMemForLocalStep(self, ShMem=None):
        return dict()

    def getLocalAndSummaryFunctions(self):
        return calcLocalParams, calcSummaryStats
    

def L_alloc(Data=None, LP=None, nDoc=0, alpha=1.0, **kwargs):
    ''' Calculate allocation term of the ELBO objective.

    E[ log p(pi) + log p(z) - log q(pi)  ]

    Returns
    -------
    L_alloc : scalar float
    '''
    if Data is not None:
        nDoc = Data.nDoc
    if LP is None:
        LP = dict(**kwargs)
    K = LP['DocTopicCount'].shape[1]
    cDiff = nDoc * c_Func(alpha, K) - c_Func(LP['theta'])
    slackVec = LP['DocTopicCount'] + alpha - LP['theta']
    slackVec *= LP['ElogPi']
    return cDiff + np.sum(slackVec)


def L_entropy(Data=None, LP=None, resp=None, returnVector=0):
    """ Calculate entropy of soft assignments term in ELBO objective.

    Returns
    -------
    L_entropy : scalar float
    """
    if LP is not None:
        resp = LP['resp']
    if hasattr(Data, 'word_count'):
        Hvec = -1 * NumericUtil.calcRlogRdotv(resp, Data.word_count)
    else:
        Hvec = -1 * NumericUtil.calcRlogR(resp)
    if returnVector:
        return Hvec
    return Hvec.sum()


def c_Func(avec, K=0):
    ''' Evaluate cumulant function of the Dirichlet distribution

    Returns
    -------
    c : scalar real
    '''
    if isinstance(avec, float) or avec.ndim == 0:
        assert K > 0
        avec = avec * np.ones(K)
        return gammaln(np.sum(avec)) - np.sum(gammaln(avec))
    elif avec.ndim == 1:
        return gammaln(np.sum(avec)) - np.sum(gammaln(avec))
    else:
        return np.sum(gammaln(np.sum(avec, axis=1))) - np.sum(gammaln(avec))



def calcSummaryStats(Dslice, LP=None, alpha=None,
                     **kwargs):
    """ Calculate summary from local parameters for given data slice.

    Parameters
    -------
    Data : bnpy data object
    LP : local param dict with fields
        resp : Data.nObs x K array,
            where resp[n,k] = posterior resp of comp k
        doPrecompEntropy : boolean flag
            indicates whether to precompute ELBO terms in advance
            used for memoized learning algorithms (moVB)

    Returns
    -------
    SS : SuffStatBag with K components
        * nDoc : scalar float
            Counts total documents available in provided data.

        Also has optional ELBO field when precompELBO is True
        * Hvec : 1D array, size K
            Vector of entropy contributions from each comp.
            Hvec[k] = \sum_{n=1}^N H[q(z_n)], a function of 'resp'
    """
    resp = LP['resp']
    _, K = resp.shape

    SS = SuffStatBag(K=K, D=Dslice.vocab_size)
    SS.setField('nDoc', Dslice.nDoc, dims=None)
    if doPrecompEntropy:
        Hvec = L_entropy(Dslice, LP, returnVector=1)
        Lalloc = L_alloc(Dslice, LP, alpha=alpha)
        SS.setELBOTerm('Hvec', Hvec, dims='K')
        SS.setELBOTerm('L_alloc', Lalloc, dims=None)
    return LP
