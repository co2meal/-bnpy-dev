'''
Bayesian nonparametric mixture model with Dirichlet process prior.
'''

import numpy as np

from bnpy.allocmodel import AllocModel
from bnpy.suffstats import SuffStatBag
from bnpy.util import NumericUtil
from bnpy.util import gammaln, digamma, EPS
from bnpy.util.StickBreakUtil import beta2rho


def convertToN0(N):
    """ Convert count vector to vector of "greater than" counts.

    Parameters
    -------
    N : 1D array, size K
        each entry k represents the count of items assigned to comp k.

    Returns
    -------
    N0 : 1D array, size K
        each entry k gives the total count of items at index above k
        N0[k] = np.sum(N[k:])

    Example
    -------
    >>> convertToN0([1., 3., 7., 2])
    [12, 9, 2]
    """
    N = np.asarray(N)
    N0 = np.zeros_like(N)
    N0[:-1] = N[::-1].cumsum()[::-1][1:]
    return N0

def c_Beta(g1, g0):
    ''' Evaluate cumulant function of Beta distribution

    Parameters
    -------
    g1 : 1D array, size K
        represents ON pseudo-count parameter of the Beta
    g0 : 1D array, size K
        represents OFF pseudo-count parameter of the Beta

    Returns
    -------
    c : float
    '''
    return np.sum(gammaln(g1 + g0) - gammaln(g1) - gammaln(g0))


def c_Beta_Vec(g1, g0):
    ''' Evaluate cumulant function of Beta distribution in vectorized way

    Parameters
    -------
    g1 : 1D array, size K
        represents ON pseudo-count parameter of the Beta
    g0 : 1D array, size K
        represents OFF pseudo-count parameter of the Beta

    Returns
    -------
    cvec : 1D array, size K
    '''
    return gammaln(g1 + g0) - gammaln(g1) - gammaln(g0)


def Lalloc_no_slack(eta1, eta0, gamma1, gamma0):
    """ Evaluate partial L_alloc objective function, without slack.

    This is exact when evaluated directly after a global step.

    Returns
    -------
    Lalloc : float
    """
    K = eta1.size
    return K * c_Beta(gamma1, gamma0) - c_Beta(eta1, eta0)


def Lalloc_slack_only(N, eta1, eta0):
    """ Evaluate slack portion of L_alloc objective.

    Returns
    -------
    Lalloc : float
    """
    K = N.size
    assert K == eta1.size
    assert K == eta0.size
    N0 = convertToN0(N)
    digammaBoth = digamma(eta1 + eta0)
    ElogU = digamma(eta1) - digammaBoth
    Elog1mU = digamma(eta0) - digammaBoth
    Lslack = np.inner(N + gamma1 - eta1, ElogU) \
        + np.inner(N0 + gamma0 - eta0, Elog1mU)
    return Lslack


def Lalloc_with_slack(N, eta1, eta0, gamma1, gamma0):
    """ Evaluate complete L_alloc objective term given model parameters.

    Returns
    -------
    La : float
    """
    K = eta1.size
    return Lalloc_no_slack(eta1, eta0, gamma1, gamma0) \
        + Lalloc_slack_only(N, eta1, eta0)


def Lentropy(resp):
    """ Evaluate L_entropy objective term given model parameters.

    Returns
    -------
    Lentropy : float
    """
    return -1 * np.sum(NumericUtil.calcRlogR(resp))

def calcCachedELBOGap_SinglePair(SS, kA, kB, 
                                 delCompID=None, dtargetMinCount=None):
    """ Compute (lower bound on) gap in cacheable ELBO

    Returns
    ------
    gap : scalar
        L'_entropy - L_entropy >= gap
    """
    assert SS.hasELBOTerms()
    # Hvec : 1D array, size K
    Hvec = -1 * SS.getELBOTerm('ElogqZ')
    if delCompID is None:
        # Use bound - r log r >= 0
        gap = -1 * (Hvec[kA] + Hvec[kB])
    else:
        # Use bound - (1-r) log (1-r) >= r for small values of r 
        assert delCompID == kA or delCompID == kB
        gap1 = -1 * Hvec[delCompID] - SS.N[delCompID]
        gap2 = -1 * (Hvec[kA] + Hvec[kB])
        gap = np.maximum(gap1, gap2)
    return gap

def calcCachedELBOTerms_SinglePair(SS, kA, kB, delCompID=None):
    """ Calculate all cached ELBO terms under proposed merge.
    """
    assert SS.hasELBOTerms()
    # Hvec : 1D array, size K
    Hvec = -1 * SS.getELBOTerm('ElogqZ')
    newHvec = np.delete(Hvec, kB)
    if delCompID is None:
        newHvec[kA] = 0
    else:
        assert delCompID == kA or delCompID == kB
        if delCompID == kA:
            newHvec[kA] = Hvec[kB]
        newHvec[kA] -= SS.N[delCompID]
        newHvec[kA] = np.maximum(0, newHvec[kA])
    return dict(ElogqZ=-1*newHvec)


class DPMixtureModel(AllocModel):
    """ Nonparametric mixture model with K active components.

    Attributes
    -------
    * inferType : string {'VB', 'moVB', 'soVB'}
        indicates which updates to perform for local/global steps
    * K : int
        number of components
    * gamma1 : float
        scalar pseudo-count of ON values
        used in Beta prior on stick-breaking lengths.
    * gamma0 : float
        scalar pseudo-count of OFF values
        used in Beta prior on stick-breaking lengths.

    Attributes for VB
    ---------
    * eta1 : 1D array, size K
        Posterior ON parameters for Beta posterior factor q(u).
        eta1[k] > 0 for all k
    * eta0 : 1D array, size K
        Posterior OFF parameters for Beta posterior factor q(u).
        eta0[k] > 0 for all k

    Secondary Attributes for VB
    ---------
    * ElogU : 1D array, size K
        Expected value E[log u[k]] under current q(u[k])
    * Elog1mU : 1D array, size K
        Expected value E[log 1 - u[k]] under current q(u[k])
    """

    def __init__(self, inferType, priorDict=None, **priorKwargs):
        if inferType == 'EM':
            raise ValueError('EM not supported.')
        self.inferType = inferType
        if priorDict is not None:
            self.set_prior(**priorDict)
        else:
            self.set_prior(**priorKwargs)
        self.K = 0

    def set_prior(self, gamma1=1.0, gamma0=5.0, **kwargs):
        self.gamma1 = gamma1
        self.gamma0 = gamma0

    def set_helper_params(self):
        ''' Set dependent attributes given primary global params.

        This means storing digamma function evaluations.
        '''
        digammaBoth = digamma(self.eta0 + self.eta1)
        self.ElogU = digamma(self.eta1) - digammaBoth
        self.Elog1mU = digamma(self.eta0) - digammaBoth

        # Calculate expected mixture weights E[ log \beta_k ]
        # Using copy() allows += without modifying ElogU
        self.Elogbeta = self.ElogU.copy()
        self.Elogbeta[1:] += self.Elog1mU[:-1].cumsum()

    def get_active_comp_probs(self):
        ''' Get vector of appearance probabilities for each active comp.

            Returns
            -------
            beta : 1D array, size K
                beta[k] gives probability of comp. k under this model.
        '''
        Eu = self.eta1 / (self.eta1 + self.eta0)
        Ebeta = Eu.copy()
        Ebeta[1:] *= np.cumprod(1.0 - Eu[:-1])
        return Ebeta

    def get_keys_for_memoized_local_params(self):
        ''' Return LP field names required for warm starts of local step
        '''
        return list()

    def calc_local_params(self, Data, LP, **kwargs):
        ''' Compute local parameters for each data item and component.

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
            * resp : 2D array, size N x K array
                Posterior responsibility each comp has for each item
                resp[n, k] = p(z[n] = k | x[n])
        '''
        lpr = LP['E_log_soft_ev']
        lpr += self.Elogbeta
        # Calculate exp in numerically stable manner (first subtract the max)
        #  perform this in-place so no new allocations occur
        NumericUtil.inplaceExpAndNormalizeRows(lpr)
        LP['resp'] = lpr
        assert np.allclose(lpr.sum(axis=1), 1)
        return LP

    def selectSubsetLP(self, Data, LP, relIDs):
        ''' Make subset of provided local params for certain data items.

        Returns
        ------
        LP : dict
             New local parameter dict for subset of data, with fields
             * resp : 2D array, size Nsubset x K
        '''
        resp = LP['resp'][relIDs].copy()
        return dict(resp=resp)

    def get_global_suff_stats(self, Data, LP,
                              mergePairSelection=None,
                              doPrecompEntropy=False,
                              doPrecompMergeEntropy=False, 
                              mPairIDs=None,
                              trackDocUsage=False,
                              **kwargs):
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
        doPrecompMergeEntropy : boolean flag
            indicates whether to precompute ELBO terms in advance
            for certain merge candidates.

        Returns
        -------
        SS : SuffStatBag with K components
            Summarizes for this mixture model, with fields
            * N : 1D array, size K
                N[k] = expected number of items assigned to comp k

            Also has optional ELBO field when precompELBO is True
            * ElogqZ : 1D array, size K
                Vector of entropy contributions from each comp.
                ElogqZ[k] = \sum_{n=1}^N resp[n,k] log resp[n,k]

            Also has optional Merge field when precompMergeELBO is True
            * ElogqZ : 2D array, size K x K
                Each term is scalar entropy of merge candidate
        '''
        Nvec = np.sum(LP['resp'], axis=0)
        K = Nvec.size
        if hasattr(Data, 'dim'):
            SS = SuffStatBag(K=K, D=Data.dim)
        else:
            SS = SuffStatBag(K=K, D=Data.vocab_size)
        SS.setField('N', Nvec, dims=('K'))

        if doPrecompEntropy:
            ElogqZ_vec = self.E_logqZ(LP)
            SS.setELBOTerm('ElogqZ', ElogqZ_vec, dims=('K'))

        if doPrecompMergeEntropy:
            resp = LP['resp']
            if mPairIDs is None:
                ElogqZMat = NumericUtil.calcRlogR_allpairs(resp)
            else:
                ElogqZMat = NumericUtil.calcRlogR_specificpairs(resp, mPairIDs)
            SS.setMergeTerm('ElogqZ', ElogqZMat, dims=('K', 'K'))
        if trackDocUsage:
            ## Track num items with signif. mass assigned to each state.
            DocUsage = np.sum(LP['resp'] > 0.01, axis=0)
            SS.setSelectionTerm('DocUsageCount', DocUsage, dims='K')
        return SS

    def forceSSInBounds(self, SS):
        ''' Enforce known bounds on SS fields for numerical stability.

        Post Condition for SS fields
        --------
        N : will have no entries below zero

        Post Condition for SS ELBO fields
        --------
        ElogqZ : will have no entries above zero
        '''
        np.maximum(SS.N, 0, out=SS.N)
        if SS.hasELBOTerm('ElogqZ'):
            Hvec = SS.getELBOTerm('ElogqZ')
            Hmax = Hvec.max()
            assert Hmax < 1e-10  # should be all negative
            if Hmax > 0:  # fix numerical errors to force entropy negative
                np.minimum(Hvec, 0, out=Hvec)
        if SS.hasMergeTerm('ElogqZ'):
            Hmat = SS.getMergeTerm('ElogqZ')
            Hmax = Hmat.max()
            assert Hmax < 1e-10  # should be all negative
            if Hmax > 0:
                np.minimum(Hmat, 0, out=Hmat)

    def update_global_params_VB(self, SS, **kwargs):
        """ Update eta1, eta0 to optimize the ELBO objective.

        Post Condition for VB
        -------
        eta1 and eta0 set to valid posterior for SS.K components.
        """
        self.K = SS.K
        eta1 = self.gamma1 + SS.N
        eta0 = self.gamma0 + convertToN0(SS.N)
        self.eta1 = eta1
        self.eta0 = eta0
        self.set_helper_params()

    def update_global_params_soVB(self, SS, rho, **kwargs):
        """ Update eta1, eta0 to optimize stochastic ELBO objective.

        Post Condition for VB
        -------
        eta1 and eta0 set to valid posterior for SS.K components.
        """
        assert self.K == SS.K
        eta1 = self.gamma1 + SS.N
        eta0 = self.gamma0 * np.ones(self.K)
        eta0 = self.gamma0 + convertToN0(SS.N)
        self.eta1 = rho * eta1 + (1 - rho) * self.eta1
        self.eta0 = rho * eta0 + (1 - rho) * self.eta0
        self.set_helper_params()

    def init_global_params(self, Data, K=0, **kwargs):
        """ Initialize global parameters to reasonable default values.

        Post Condition for VB
        -------
        eta1 and eta0 set to valid K vector.
        """
        self.setParamsFromCountVec(K, np.ones(K))

    def set_global_params(self, hmodel=None, K=None,
                          beta=None,
                          eta1=None, eta0=None, **kwargs):
        """ Set global parameters to provided values.

        Post Condition for EM
        -------
        w set to valid vector with K components.

        Post Condition for VB
        -------
        theta set to define valid posterior over K components.
        """
        if hmodel is not None:
            self.setParamsFromHModel(hmodel)
        elif beta is not None:
            self.setParamsFromBeta(K, beta=beta)
        elif eta1 is not None:
            self.K = int(K)
            self.eta1 = eta1
            self.eta0 = eta0
            self.set_helper_params()
        else:
            raise ValueError("Unrecognized set_global_params args")

    def setParamsFromCountVec(self, K, N=None):
        """ Set params to reasonable values given counts for each comp.

        Parameters
        --------
        K : int
            number of components
        N : 1D array, size K. optional, default=[1 1 1 1 ... 1]
            size of each component

        Post Condition for VB
        ---------
        Attributes eta1, eta0 are set so q(beta) equals its posterior
        given count vector N.
        """
        self.K = int(K)

        if N is None:
            N = 1.0 * np.ones(K)
        assert N.ndim == 1
        assert N.size == K
        self.eta1 = self.gamma1 + N
        self.eta0 = self.gamma0 + convertToN0(N)
        self.set_helper_params()

    def setParamsFromBeta(self, K, beta=None):
        """ Set params to reasonable values given comp probabilities.

        Parameters
        --------
        K : int
            number of components
        beta : 1D array, size K. optional, default=[1 1 1 1 ... 1]
            probability of each component

        Post Condition for VB
        ---------
        Attribute eta1, eta0 is set so q(beta) has properties:
        * mean of (nearly) beta, allowing for some small remaining mass.
        * moderate variance.
        """
        if beta is None:
            beta = 1.0 / K * np.ones(K)
        assert beta.ndim == 1
        assert beta.size == K
        assert np.allclose(np.sum(beta), 1.0)
        self.K = int(K)

        # Append in small remaining/leftover mass
        betaRem = np.minimum(1.0 / (2 * K), 0.05)
        betaWithRem = np.hstack([beta * (1.0 - betaRem), betaRem])

        theta = self.K * betaWithRem
        self.eta1 = theta[:-1].copy()
        self.eta0 = theta[::-1].cumsum()[::-1][1:]
        self.set_helper_params()

    def setParamsFromHModel(self, hmodel):
        """ Set parameters exactly as in provided HModel object.

        Parameters
        ------
        hmodel : bnpy.HModel
            The model to copy parameters from.

        Post Condition
        ------
        w or theta will be set exactly equal to hmodel's allocModel.
        """
        self.K = hmodel.allocModel.K
        self.eta1 = hmodel.allocModel.eta1.copy()
        self.eta0 = hmodel.allocModel.eta0.copy()
        self.set_helper_params()

    def calc_evidence(self, Data, SS, LP=None, todict=False, **kwargs):
        """ Calculate ELBO objective function value for provided state.

        Returns
        -------
        L : float
            represents sum of all terms in objective
        """
        if SS.hasELBOTerm('ElogqZ'):
            Lentropy = -1 * np.sum(SS.getELBOTerm('ElogqZ'))
        else:
            Lentropy = -1 * np.sum(self.E_logqZ(LP))
        if SS.hasAmpFactor():
            Lentropy *= SS.ampF

        cDiff = self.ELBO_cDiff()
        slack = self.ELBO_slack(SS)
        if todict:
            return dict(Lalloc=cDiff+slack,
                        Lentropy=Lentropy)
        return cDiff + slack + Lentropy

    def calcELBOFromSS_NoCacheableTerms(self, SS):
        ''' Calculate objective value, ignoring any cached ELBO terms.

        Returns
        -------
        L : float
            represents sum of most terms in objective
        '''
        assert self.K == SS.K
        cDiff = self.ELBO_cDiff()
        slack = self.ELBO_slack(SS)
        return cDiff + slack

    def E_logqZ(self, LP):
        ''' Compute ELBO term related to entropy of soft assignments.

        Returns
        -------
        Hvec : 1D array, size K
        '''
        return NumericUtil.calcRlogR(LP['resp'])

    def ELBO_cDiff(self):
        ''' Compute difference of cumulant functions for ELBO

        Returns
        -------
        cDiff : scalar real
        '''
        cDiff = self.K * c_Beta(self.gamma1, self.gamma0) \
            - c_Beta(self.eta1, self.eta0)  # already sums over k
        return cDiff

    def ELBO_slack(self, SS):
        ''' Compute the slack-term for ELBO

        Returns
        ------
        Lslack : scalar real
        '''
        slack = np.inner(self.gamma1 - self.eta1, self.ElogU) \
            + np.inner(self.gamma0 - self.eta0, self.Elog1mU) \
            + np.inner(SS.N, self.Elogbeta)
        return slack


    def getBestMergePairInvolvingComp(self, SS, k, partnerIDs):
        """

        """
        Hcur = -1 * SS.getELBOTerm('Elogqz').sum()
        
        for j in partnerIDs:
            kA = np.minimum(j, k)
            kB = np.maximum(j, k)
            Gap[j] = self.calcHardMergeGapFastSinglePair(SS, kA, kB)        
            Hgap = Hcur - Hprop
            Gap[j] += Hgap
        return Gap

    def calcHardMergeEntropyGap(self, SS, kA, kB):
        ''' Calc scalar improvement in entropy for merge of kA, kB
        '''
        Hmerge = SS.getMergeTerm('ElogqZ')
        Hcur = SS.getELBOTerm('ElogqZ')
        if Hmerge.ndim == 1:
            gap = Hcur[kB] - Hmerge[kB]
        else:
            gap = - Hmerge[kA, kB] + Hcur[kA] + Hcur[kB]
        return gap

    def calcHardMergeGap(self, SS, kA, kB):
        ''' Calculate scalar improvement in ELBO for merge of kA, kB

        For speed, use one of
        * calcHardMergeGapFast
        * calcHardMergeGapFastSinglePair.

        Does *not* include the entropy term for soft assignments.

        Returns
        -------
        L : float
            difference of partial ELBO functions
        '''
        cPrior = c_Beta(self.gamma1, self.gamma0)
        cB = c_Beta(self.eta1[kB], self.eta0[kB])

        gap = cB - cPrior
        # Add terms for changing kA to kA+kB
        gap += c_Beta(self.eta1[kA], self.eta0[kA]) \
            - c_Beta(self.eta1[kA] + SS.N[kB], self.eta0[kA] - SS.N[kB])

        # Add terms for each index kA+1, kA+2, ... kB-1
        # where only \gamma_0 has changed
        for k in xrange(kA + 1, kB):
            a1 = self.eta1[k]
            a0old = self.eta0[k]
            a0new = self.eta0[k] - SS.N[kB]
            gap += c_Beta(a1, a0old) - c_Beta(a1, a0new)
        return gap

    def calcHardMergeGapFast(self, SS, kA, kB):
        ''' Calculate scalar improvement in ELBO for merge of kA, kB

            Returns
            -------
            gap : float
        '''
        if not hasattr(self, 'cPrior'):
            self.cPrior = c_Beta(self.gamma1, self.gamma0)
        if not hasattr(self, 'cBetaCur'):
            self.cBetaCur = c_Beta_Vec(self.eta1, self.eta0)
        if not hasattr(self, 'cBetaNewB') \
           or not (hasattr(self, 'kB') and self.kB == kB):
            self.kB = kB
            self.cBetaNewB = c_Beta_Vec(self.eta1[:kB],
                                        self.eta0[:kB] - SS.N[kB])
        cDiff_A = self.cBetaCur[kA] \
            - c_Beta(self.eta1[kA] + SS.N[kB], self.eta0[kA] - SS.N[kB])
        cDiff_AtoB = np.sum(self.cBetaCur[kA + 1:kB] - self.cBetaNewB[kA + 1:])
        gap = self.cBetaCur[kB] - self.cPrior + cDiff_A + cDiff_AtoB
        return gap

    def calcHardMergeGapFastSinglePair(self, SS, kA, kB):
        ''' Calculate scalar improvement in ELBO for merge of kA, kB

            Returns
            -------
            gap : float
        '''
        if not hasattr(self, 'cPrior'):
            self.cPrior = c_Beta(self.gamma1, self.gamma0)
        if not hasattr(self, 'cBetaCur'):
            self.cBetaCur = c_Beta_Vec(self.eta1, self.eta0)

        cBetaNew_AtoB = c_Beta_Vec(self.eta1[kA + 1:kB],
                                   self.eta0[kA + 1:kB] - SS.N[kB])
        cDiff_A = self.cBetaCur[kA] \
            - c_Beta(self.eta1[kA] + SS.N[kB], self.eta0[kA] - SS.N[kB])
        cDiff_AtoB = np.sum(self.cBetaCur[kA + 1:kB] - cBetaNew_AtoB)
        gap = self.cBetaCur[kB] - self.cPrior + cDiff_A + cDiff_AtoB
        return gap

    def calcHardMergeGap_AllPairs(self, SS):
        ''' Calc matrix of improvement in ELBO for all possible pairs of comps
        '''
        Gap = np.zeros((SS.K, SS.K))
        for kB in xrange(1, SS.K):
            for kA in xrange(0, kB):
                Gap[kA, kB] = self.calcHardMergeGapFast(SS, kA, kB)
        if hasattr(self, 'cBetaNewB'):
            del self.cBetaNewB
            del self.kB
        if hasattr(self, 'cPrior'):
            del self.cPrior
        if hasattr(self, 'cBetaCur'):
            del self.cBetaCur
        return Gap

    def calcHardMergeGap_SpecificPairs(self, SS, PairList):
        ''' Calc matrix of improvement in ELBO for all possible pairs of comps
        '''
        Gaps = np.zeros(len(PairList))
        for ii, (kA, kB) in enumerate(PairList):
            Gaps[ii] = self.calcHardMergeGapFastSinglePair(SS, kA, kB)
        if hasattr(self, 'cPrior'):
            del self.cPrior
        if hasattr(self, 'cBetaCur'):
            del self.cBetaCur
        return Gaps

    def calcCachedELBOTerms_SinglePair(self, SS, kA, kB, delCompID=None):
        """ Compute ELBO terms after merge of kA, kB

        Returns
        -------
        ELBOTerms : dict
            Key/value pairs are field name (str) and array
        """
        return calcCachedELBOTerms_SinglePair(SS, kA, kB, delCompID=delCompID)

    def calcCachedELBOGap_SinglePair(self, SS, kA, kB, 
                                     delCompID=None):
        """ Compute (lower bound on) gap in cacheable ELBO

        Returns
        ------
        gap : scalar
            L'_entropy - L_entropy >= gap
        """
        return calcCachedELBOGap_SinglePair(SS, kA, kB, delCompID=delCompID)

    def calcCachedELBOGap_FromSS(self, curSS, propSS):
        """ Compute gap on cachable ELBO term directly
        """
        L_cur = -1 * curSS.getELBOTerm('ElogqZ').sum()
        L_prop = -1 * propSS.getELBOTerm('ElogqZ').sum()
        return L_prop - L_cur

    def calcCachedELBOGapForDeleteProposal(self, SSall_before,
                                           SStarget_before,
                                           SStarget_after,
                                           delCompUIDs):
        ''' Calculate improvement in entropy term after a delete
        '''
        remCompIDs = list()
        for k in xrange(SSall_before.K):
            if SSall_before.uIDs[k] not in delCompUIDs:
                remCompIDs.append(k)

        Hvec_all_before = -1 * SSall_before.getELBOTerm('ElogqZ')
        Hvec_target_before = -1 * SStarget_before.getELBOTerm('ElogqZ')
        Hvec_target_after = -1 * SStarget_after.getELBOTerm('ElogqZ')

        Hvec_rest_before = Hvec_all_before - Hvec_target_before
        if not np.all(Hvec_rest_before > -1e-10):
            print 'ASSUMPTION ABOUT Hvec_rest_before > 0 VIOLATED'

        Htarget_after = Hvec_target_after.sum()
        Hrest_after = Hvec_rest_before[remCompIDs].sum()

        gap = Hvec_all_before.sum() - (Htarget_after + Hrest_after)
        return gap

    def get_info_string(self):
        ''' Returns one-line human-readable terse description of this object
        '''
        msgPattern = 'DP mixture with K=%d. Concentration gamma0= %.2f'
        return msgPattern % (self.K, self.gamma0)

    def to_dict(self):
        return dict(eta1=self.eta1, eta0=self.eta0)

    def from_dict(self, myDict):
        self.inferType = myDict['inferType']
        self.K = myDict['K']
        self.eta1 = myDict['eta1']
        self.eta0 = myDict['eta0']
        if self.eta0.ndim == 0:
            self.eta0 = self.eta1[np.newaxis]
        if self.eta0.ndim == 0:
            self.eta0 = self.eta0[np.newaxis]
        self.set_helper_params()

    def get_prior_dict(self):
        return dict(gamma1=self.gamma1,
                    gamma0=self.gamma0,
                    K=self.K,
                    )

    def make_hard_asgn_local_params(self, LP):
        ''' Convert soft assignments to hard assignments for provided local params

            Returns
            --------
            LP : local params dict, with new fields
                 Z : 1D array, size N
                        Z[n] is an integer in range {0, 1, 2, ... K-1}
                 resp : 2D array, size N x K+1 (with final column empty)
                        resp[n,k] = 1 iff Z[n] == k
        '''
        LP['Z'] = np.argmax(LP['resp'], axis=1)
        K = LP['resp'].shape[1]
        LP['resp'].fill(0)
        for k in xrange(K):
            LP['resp'][LP['Z'] == k, k] = 1
        return LP

    def removeEmptyComps_SSandLP(self, SS, LP):
        ''' Remove all parameters related to empty components from SS and LP

            Returns
            --------
            SS : bnpy SuffStatBag
            LP : dict for local params
        '''
        badks = np.flatnonzero(SS.N[:-1] < 1)
        # Remove in order, from largest index to smallest
        for k in badks[::-1]:
            SS.removeComp(k)
            mask = LP['Z'] > k
            LP['Z'][mask] -= 1
        if 'resp' in LP:
            del LP['resp']
        return SS, LP

    def insertEmptyCompAtLastIndex_SSandLP(self, SS, LP):
        ''' Create empty component and insert last in order into SS

            Returns
            --------
            SS
            LP
        '''
        SS.insertEmptyComps(1)
        return SS, LP

    def sample_local_params(self, obsModel, Data, SS, LP, PRNG, **algParams):
        ''' Sample local assignments of all data items to components
        '''
        Z = LP['Z']
        # Iteratively sample data allocations
        for dataindex in xrange(Data.nObs):
            x = Data.X[dataindex]

            # de-update current assignment and suff stats
            kcur = Z[dataindex]
            SS.N[kcur] -= 1
            obsModel.decrementSS(SS, kcur, x)

            SS, LP = self.removeEmptyComps_SSandLP(SS, LP)

            doKeepFinalCompEmpty = SS.K < algParams['Kmax']
            if SS.N[-1] > 0 and doKeepFinalCompEmpty:
                SS, LP = self.insertEmptyCompAtLastIndex_SSandLP(SS, LP)

            # Calculate probs
            alloc_prob = self.getConditionalProbVec_Unnorm(
                SS, doKeepFinalCompEmpty)
            pvec = obsModel.calcPredProbVec_Unnorm(SS, x)
            pvec *= alloc_prob
            psum = np.sum(pvec)

            if np.isnan(psum) or psum <= 0:
                print pvec
                print psum
                raise ValueError('BAD VALUES FOR PROBS!')

            pvec /= psum
            # sample new allocation
            knew = PRNG.choice(SS.K, p=pvec)

            # update with new assignment
            SS.N[knew] += 1
            obsModel.incrementSS(SS, knew, x)
            Z[dataindex] = knew

        LP['Z'] = Z
        print ' '.join(['%.1f' % (x) for x in SS.N])
        return LP, SS

    def getConditionalProbVec_Unnorm(self, SS, doKeepFinalCompEmpty):
        ''' Returns a K vector of positive values \propto p(z_i|z_-i)
        '''
        if doKeepFinalCompEmpty:
            assert SS.N[-1] == 0
            return np.hstack([SS.N[:-1], self.gamma0])
        else:
            return np.hstack([SS.N[:-1], np.maximum(SS.N[-1], self.gamma0)])

    def calcMargLik(self, SS):
        ''' Calculate marginal likelihood of assignments, summed over all comps
        '''
        mask = SS.N > 0
        Nvec = SS.N[mask]
        K = Nvec.size
        return gammaln(self.gamma0) \
            + K * np.log(self.gamma0) \
            + np.sum(gammaln(Nvec)) \
            - gammaln(np.sum(Nvec) + self.gamma0)
