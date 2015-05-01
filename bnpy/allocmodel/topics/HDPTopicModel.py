import numpy as np
import logging

import LocalStepManyDocs
import OptimizerRhoOmega

from HDPTopicUtil import calcELBO
from HDPTopicUtil import calcELBO_LinearTerms, calcELBO_NonlinearTerms
from HDPTopicUtil import calcHrespForMergePairs

from bnpy.allocmodel.AllocModel import AllocModel
from bnpy.allocmodel.mix.DPMixtureModel import convertToN0
from bnpy.suffstats import SuffStatBag
from bnpy.util import digamma, gammaln
from bnpy.util import as1D
from bnpy.util.StickBreakUtil import rho2beta

from bnpy.util.NumericUtil import calcRlogRdotv, calcRlogR
from bnpy.util.NumericUtil import calcRlogRdotv_allpairs
from bnpy.util.NumericUtil import calcRlogRdotv_specificpairs
from bnpy.util.NumericUtil import calcRlogR_allpairs, calcRlogR_specificpairs
from bnpy.util import sharedMemToNumpyArray, numpyToSharedMemArray

Log = logging.getLogger('bnpy')


class HDPTopicModel(AllocModel):
    '''
    Bayesian nonparametric topic model with a K active components.

    Uses a direct construction that truncates unbounded posterior to
    K active components (assigned to data), indexed 0, 1, ... K-1.
    Remaining mass for inactive topics is represented at index K.

    Attributes
    -------
    inferType : string {'VB', 'moVB', 'soVB'}
        indicates which updates to perform for local/global steps
    K : int
        number of components
    alpha : float
        scalar pseudo-count
        used in Dirichlet prior on document-topic probabilities.
    gamma : float
        scalar concentration of the top-level stick breaking process.

    Attributes for VB
    ---------
    rho : 1D array, size K
        rho[k] defines the mean of each stick-length u[k]
    omega : 1D array, size K
        omega[k] controls variance of stick-length u[k]

    Together, rho/omega define the approximate posterior factor q(u[k]):
        eta1 = rho * omega
        eta0 = (1-rho) * omega
        q(u) = \prod_k Beta(u[k] | eta1[k], eta0[k])

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
            raise ValueError('HDPTopicModel annot do EM.')
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
        return self.E_beta_active()

    def E_beta_active(self):
        ''' Get vector of appearance probabilities for each active comp.

        Returns
        -------
        beta : 1D array, size K
            beta[k] gives probability of comp. k under this model.
        '''
        if not hasattr(self, 'Ebeta'):
            self.Ebeta = self.E_beta()
        return self.Ebeta[:-1]

    def E_beta_rem(self):
        ''' Get scalar probability of remaining/inactive topics.

        Returns
        -------
        EbetaRem : float
        '''
        if not hasattr(self, 'Ebeta'):
            self.Ebeta = self.E_beta()
        return self.Ebeta[-1]

    def E_beta(self):
        ''' Get vector of probabilities for active and inactive topics.

        Returns
        -------
        beta : 1D array, size K + 1
            beta[k] gives probability of comp. k under this model.
            beta[K] (last index) is aggregated over all inactive topics.
        '''
        if not hasattr(self, 'Ebeta'):
            self.Ebeta = rho2beta(self.rho)
        return self.Ebeta

    def alpha_E_beta(self):
        ''' Return scaled vector alpha * E[beta] for all topics.

        The vector alpha * Ebeta parameterizes the Dirichlet
        distribution over the prior on document-topic probabilities.

        Returns
        -------
        abeta : 1D array, size K
            abeta[k] gives scaled parameter for comp. k under this model.
            abeta[K] (last index) is aggregated over all inactive topics.
        '''
        if not hasattr(self, 'alphaEbeta'):
            self.alphaEbeta = self.alpha * self.E_beta_active()
        return self.alphaEbeta

    def alpha_E_beta_rem(self):
        ''' Return scalar  alpha * E[beta_{>K}] for inactive topics.

        Returns
        -------
        abetaRem : scalar
        '''
        if not hasattr(self, 'alphaEbetaRem'):
            self.alphaEbetaRem = self.alpha * self.E_beta_rem()
        return self.alphaEbetaRem

    def ClearCache(self):
        """ Clear cached computations stored as attributes.

        Should always be called after a global update.

        Post Condition
        --------------
        Cached computations stored as attributes in this object
        will be erased.
        """
        if hasattr(self, 'Ebeta'):
            del self.Ebeta
        if hasattr(self, 'alphaEbeta'):
            del self.alphaEbeta
        if hasattr(self, 'alphaEbetaRem'):
            del self.alphaEbetaRem

    def set_prior(self, gamma=1.0, alpha=1.0, **kwargs):
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def to_dict(self):
        return dict(rho=self.rho, omega=self.omega)

    def from_dict(self, Dict):
        self.inferType = Dict['inferType']
        self.K = Dict['K']
        self.rho = as1D(Dict['rho'])
        self.omega = as1D(Dict['omega'])

    def get_prior_dict(self):
        return dict(alpha=self.alpha, gamma=self.gamma,
                    K=self.K,
                    inferType=self.inferType)

    def get_info_string(self):
        ''' Returns human-readable name of this object
        '''
        return 'HDP model with K=%d active comps. gamma=%.2f. alpha=%.2f' \
            % (self.K, self.gamma, self.alpha)

    def calc_local_params(self, Data, LP, **kwargs):
        ''' Calculate document-specific quantities (E-step)

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
                Positive pseudo-count parameter for active topics,
                in the approximate posterior on doc-topic probabilities.
            * thetaRem : scalar float
                Positive pseudo-count parameter for inactive topics.
                in the approximate posterior on doc-topic probabilities.
            * ElogPi : 2D array, nDoc x K
                Expected value E[log pi[d,k]] under q(pi).
                This is a function of theta and thetaRem.
        '''
        alphaEbeta = self.alpha_E_beta().copy()
        alphaEbetaRem = self.alpha_E_beta_rem()
        LP = LocalStepManyDocs.calcLocalParams(
            Data, LP, alphaEbeta, alphaEbetaRem=alphaEbetaRem, **kwargs)
        assert 'resp' in LP
        assert 'DocTopicCount' in LP
        return LP

    def initLPFromResp(self, Data, LP, deleteCompID=None):
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

        if deleteCompID is None:
            remMass = np.minimum(0.1, 1.0 / (K * K))
            newEbeta = (1 - remMass) / K
        else:
            assert K < self.K
            assert deleteCompID < self.K
            Ebeta = self.get_active_comp_probs()
            newEbeta = np.delete(Ebeta, deleteCompID, axis=0)
            newEbeta += Ebeta[deleteCompID] / K
            remMass = 1.0 - np.sum(newEbeta)

        theta = DocTopicCount + self.alpha * newEbeta
        thetaRem = self.alpha * remMass

        digammaSumTheta = digamma(theta.sum(axis=1) + thetaRem)
        ElogPi = digamma(theta) - digammaSumTheta[:, np.newaxis]
        ElogPiRem = digamma(thetaRem) - digammaSumTheta

        LP['DocTopicCount'] = DocTopicCount
        LP['theta'] = theta
        LP['thetaRem'] = thetaRem
        LP['ElogPi'] = ElogPi
        LP['ElogPiRem'] = ElogPiRem
        return LP

    def applyHardMergePairToLP(self, LP, kA, kB):
        ''' Apply hard merge pair to provided local parameters

        Returns
        --------
        mergeLP : dict of updated local parameters
        '''
        resp = np.delete(LP['resp'], kB, axis=1)
        theta = np.delete(LP['theta'], kB, axis=1)
        DocTopicCount = np.delete(LP['DocTopicCount'], kB, axis=1)

        resp[:, kA] += LP['resp'][:, kB]
        theta[:, kA] += LP['theta'][:, kB]
        DocTopicCount[:, kA] += LP['DocTopicCount'][:, kB]

        ElogPi = np.delete(LP['ElogPi'], kB, axis=1)
        ElogPi[:, kA] = digamma(theta[:, kA]) - LP['digammaSumTheta']

        return dict(resp=resp, theta=theta, thetaRem=LP['thetaRem'],
                    ElogPi=ElogPi, ElogPiRem=LP['ElogPiRem'],
                    DocTopicCount=DocTopicCount,
                    digammaSumTheta=LP['digammaSumTheta'])

    def selectSubsetLP(self, Data, LP, docIDs):
        subsetLP = dict()
        subsetLP['DocTopicCount'] = LP['DocTopicCount'][docIDs].copy()
        subsetLP['theta'] = LP['theta'][docIDs].copy()
        subsetLP['ElogPi'] = LP['ElogPi'][docIDs].copy()

        subsetLP['thetaRem'] = LP['thetaRem']
        subsetLP['ElogPiRem'] = LP['ElogPiRem'][docIDs]

        subsetTokenIDs = list()
        for docID in docIDs:
            start = Data.doc_range[docID]
            stop = Data.doc_range[docID + 1]
            subsetTokenIDs.extend(range(start, stop))
        subsetLP['resp'] = LP['resp'][subsetTokenIDs].copy()
        return subsetLP

    def get_global_suff_stats(
            self, Data, LP,
            doPrecompEntropy=None,
            doPrecompMergeEntropy=None,
            mPairIDs=None,
            mergePairSelection=None,
            trackDocUsage=0,
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

        Returns
        -------
        SS : SuffStatBag with K components
            Relevant fields
            * nDoc : scalar float
                Counts total documents available in provided data.
            * sumLogPi : 1D array, size K
                Entry k equals \sum_{d in docs} E[ \log \pi_{dk} ]
            * sumLogPiRem : scalar float
                Equals sum over docs of probability of inactive topics.

            Also has optional ELBO field when precompELBO is True
            * Hvec : 1D array, size K
                Vector of entropy contributions from each comp.
                Hvec[k] = \sum_{n=1}^N H[q(z_n)], a function of 'resp'
        '''
        SS = calcSummaryStats(Data, LP, doPrecompEntropy=doPrecompEntropy)

        if doPrecompMergeEntropy:
            if mPairIDs is None:
                raise NotImplementedError("TODO: all pairs for merges")

            Hmat = calcHrespForMergePairs(LP['resp'], Data, mPairIDs)
            SS.setMergeTerm('Hresp', Hmat, dims=('K', 'K'))

            alphaEbeta = self.alpha_E_beta()
            sumLogPi = np.zeros((SS.K, SS.K))
            gammalnTheta = np.zeros((SS.K, SS.K))
            slackTheta = np.zeros((SS.K, SS.K))
            for (kA, kB) in mPairIDs:
                theta_vec = LP['theta'][:, kA] + LP['theta'][:, kB]
                ElogPi_vec = digamma(theta_vec) - LP['digammaSumTheta']
                gammalnTheta[kA, kB] = np.sum(gammaln(theta_vec))
                sumLogPi[kA, kB] = np.sum(ElogPi_vec)
                # slack = (Ndm - theta_dm) * E[log pi_dm]
                slack_vec = ElogPi_vec
                slack_vec *= -1 * (alphaEbeta[kA] + alphaEbeta[kB])
                slackTheta[kA, kB] = np.sum(slack_vec)
            SS.setMergeTerm('gammalnTheta', gammalnTheta, dims=('K', 'K'))
            SS.setMergeTerm('sumLogPi', sumLogPi, dims=('K', 'K'))
            SS.setMergeTerm('slackTheta', slackTheta, dims=('K', 'K'))

            # Uncomment this for verification of merge calculations.
            # for (kA, kB) in mPairIDs:
            #      self.verifySSForMergePair(Data, SS, LP, kA, kB)
            # .... end merge computations

        # Selection terms (computes doc-topic correlation)
        if mergePairSelection is not None:
            if mergePairSelection.count('corr') > 0:
                Tmat = LP['DocTopicCount']
                SS.setSelectionTerm('DocTopicPairMat',
                                    np.dot(Tmat.T, Tmat), dims=('K', 'K'))
                SS.setSelectionTerm(
                    'DocTopicSum', np.sum(Tmat, axis=0), dims='K')

        if trackDocUsage:
            # Track number of times a topic appears with "signif. mass" in a
            # doc
            DocUsage = np.sum(LP['DocTopicCount'] > 0.01, axis=0)
            SS.setSelectionTerm('DocUsageCount', DocUsage, dims='K')
        return SS

    def verifySSForMergePair(self, Data, SS, LP, kA, kB):
        mergeLP = self.applyHardMergePairToLP(LP, kA, kB)
        mergeSS = self.get_global_suff_stats(Data, mergeLP, doPrecompEntropy=1)

        sumLogPi_direct = mergeSS.sumLogPi[kA]
        sumLogPi_cached = SS.getMergeTerm('sumLogPi')[kA, kB]
        assert np.allclose(sumLogPi_direct, sumLogPi_cached)

        glnTheta_direct = mergeSS.getELBOTerm('gammalnTheta')[kA]
        glnTheta_cached = SS.getMergeTerm('gammalnTheta')[kA, kB]
        assert np.allclose(glnTheta_direct, glnTheta_cached)

        slack_direct = mergeSS.getELBOTerm('slackTheta')[kA]
        slack_cached = SS.getMergeTerm('slackTheta')[kA, kB]
        assert np.allclose(slack_direct, slack_cached)

        ElogqZ_direct = mergeSS.getELBOTerm('Hresp')[kA]
        ElogqZ_cached = SS.getMergeTerm('Hresp')[kA, kB]
        assert np.allclose(ElogqZ_direct, ElogqZ_cached)

    def update_global_params_VB(self, SS, rho=None,
                                mergeCompA=None, mergeCompB=None,
                                **kwargs):
        ''' Update global parameters.
        '''
        if mergeCompA is None:
            # Standard case:
            # Update via gradient descent.
            rho, omega = self._find_optimum_rhoomega(SS, **kwargs)
        else:
            # Special update case for merges:
            # Fast, heuristic update for rho and omega directly from existing
            # values
            beta = OptimizerRhoOmega.rho2beta_active(self.rho)
            beta[mergeCompA] += beta[mergeCompB]
            beta = np.delete(beta, mergeCompB, axis=0)
            rho = OptimizerRhoOmega.beta2rho(beta, SS.K)
            omega = self.omega
            omega[mergeCompA] += omega[mergeCompB]
            omega = np.delete(omega, mergeCompB, axis=0)
        self.rho = rho
        self.omega = omega
        self.K = SS.K
        self.ClearCache()

    def _find_optimum_rhoomega(self, SS, **kwargs):
        ''' Run numerical optimization to find optimal rho, omega parameters

            Args
            --------
            SS : bnpy SuffStatBag, with K components

            Returns
            --------
            rho : 1D array, length K
            omega : 1D array, length K
        '''
        if hasattr(self, 'rho') and self.rho.size == SS.K:
            initrho = self.rho
            initomega = self.omega
        else:
            initrho = None   # default initialization
            initomega = None
        try:
            sumLogPi = np.append(SS.sumLogPi, SS.sumLogPiRem)
            rho, omega, f, Info = OptimizerRhoOmega.\
                find_optimum_multiple_tries(
                    sumLogPi=sumLogPi,
                    nDoc=SS.nDoc,
                    gamma=self.gamma, alpha=self.alpha,
                    initrho=initrho, initomega=initomega)
        except ValueError as error:
            if str(error).count('FAILURE') == 0:
                raise error
            if hasattr(self, 'rho') and self.rho.size == SS.K:
                Log.error(
                    '***** Optim failed. Remain at cur val. ' + str(error))
                rho = self.rho
                omega = self.omega
            else:
                Log.error(
                    '***** Optim failed. Set to default init. ' + str(error))
                omega = (1 + self.gamma) * np.ones(SS.K)
                rho = OptimizerRhoOmega.create_initrho(SS.K)
        return rho, omega

    def update_global_params_soVB(self, SS, rho=None,
                                  mergeCompA=None, mergeCompB=None,
                                  **kwargs):
        ''' Update global parameters via stochastic update rule.
        '''
        rhoStar, omegaStar = self._find_optimum_rhoomega(SS, **kwargs)
        eta1Star = rhoStar * omegaStar
        eta0Star = (1 - rhoStar) * omegaStar
        eta1_t = self.rho * self.omega
        eta0_t = (1 - self.rho) * self.omega
        eta1 = (1 - rho) * eta1_t + rho * eta1Star
        eta0 = (1 - rho) * eta0_t + rho * eta0Star
        self.rho = eta1 / (eta1 + eta0)
        self.omega = eta1 + eta0
        self.K = SS.K
        self.ClearCache()

    def init_global_params(self, Data, K=0, **kwargs):
        ''' Initialize rho, omega to reasonable values
        '''
        self.setParamsFromCountVec(K, np.ones(K))

    def set_global_params(self, hmodel=None, K=None,
                          beta=None,
                          eta1=None, eta0=None,
                          rho=None, omega=None,
                          **kwargs):
        """ Set global parameters to provided values.

        Post Condition for VB
        -------
        rho/omega set to define valid posterior over K components.
        """
        self.ClearCache()
        if hmodel is not None:
            self.setParamsFromHModel(hmodel)
        elif beta is not None:
            self.setParamsFromBeta(K, beta=beta)
        elif eta1 is not None:
            self.K = int(K)
            self.omega = eta1 + eta0
            self.rho = eta1 / self.omega
        elif rho is not None:
            self.K = int(K)
            self.rho = rho.copy()
            self.omega = omega.copy()
        else:
            raise ValueError("Unrecognized set_global_params args")
        print 'rho ', self.rho[:3]
        print 'omega ', self.omega[:3]

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
        Attributes rho/omega are set so q(beta) equals its posterior
        given count vector N.
        """
        self.ClearCache()
        self.K = int(K)
        if N is None:
            N = 1.0 * np.ones(K)
        assert N.ndim == 1
        assert N.size == K
        eta1 = 1 + N
        eta0 = self.gamma + convertToN0(N)
        self.rho = eta1 / (eta1 + eta0)
        self.omega = eta1 + eta0

    def setParamsFromBeta(self, K, beta=None, oldWay=1):
        """ Set params to reasonable values given comp probabilities.

        Parameters
        --------
        K : int
            number of components
        beta : 1D array, size K. optional, default=[1/K 1/K ... 1/K]
            probability of each component

        Post Condition for VB
        ---------
        Attributes rho, omega set so q(beta) has properties:
        * mean of (nearly) beta, allowing for some small remaining mass.
        * moderate variance.
        """
        self.ClearCache()
        if beta is None:
            beta = 1.0 / K * np.ones(K)
        assert beta.ndim == 1
        assert np.allclose(np.sum(beta), 1.0)
        self.K = int(K)

        if oldWay:
            betaRem =  np.minimum(0.05, 1./(K))
            betaWithRem = np.hstack([beta, betaRem])
            betaWithRem /= betaWithRem.sum()
            self.rho = OptimizerRhoOmega.beta2rho(betaWithRem, self.K)
            self.omega = (10 + self.gamma) * np.ones(self.K)
            return

        if beta.size == K:
            # Append in small remaining/leftover mass
            betaRem = np.minimum(1.0 / (2 * K), 0.05)
            betaWithRem = np.hstack([beta * (1.0 - betaRem), betaRem])
            assert np.allclose(np.sum(betaWithRem), 1.0)
            
        else:
            assert beta.size == K + 1
            betaWithRem = beta
        
        # Convert beta to eta1, eta0
        theta = self.K * betaWithRem
        eta1 = theta[:-1].copy()
        eta0 = theta[::-1].cumsum()[::-1][1:]
        self.rho = eta1 / (eta1 + eta0)
        self.omega = eta1 + eta0

    def setParamsFromHModel(self, hmodel):
        """ Set parameters exactly as in provided HModel object.

        Parameters
        ------
        hmodel : bnpy.HModel
            The model to copy parameters from.

        Post Condition
        ------
        Attributes rho/omega set exactly equal to hmodel's allocModel.
        """
        self.ClearCache()
        self.K = hmodel.allocModel.K
        if hasattr(hmodel.allocModel, 'eta1'):
            eta1 = hmodel.allocModel.eta1.copy()
            eta0 = hmodel.allocModel.eta0.copy()
            self.rho = eta1 / (eta1 + eta0)
            self.omega = eta1 + eta0
        elif hasattr(hmodel.allocModel, 'rho'):
            self.rho = hmodel.allocModel.rho.copy()
            self.omega = hmodel.allocModel.omega.copy()
        else:
            raise AttributeError('Unrecognized hmodel')


    def calc_evidence(self, Data, SS, LP, todict=0, **kwargs):
        return calcELBO(Data=Data, SS=SS, LP=LP, todict=todict, 
            alpha=self.alpha, gamma=self.gamma,
            rho=self.rho, omega=self.omega,
            **kwargs)

    def calcELBO_LinearTerms(self, **kwargs):
        ''' Compute sum of ELBO terms that are linear/const wrt suff stats

        Returns
        -------
        L : float
        '''
        return calcELBO_LinearTerms(
            alpha=self.alpha, gamma=self.gamma,
            rho=self.rho, omega=self.omega,
            **kwargs)

    def calcELBO_NonlinearTerms(self, **kwargs):
        ''' Compute sum of ELBO terms that are nonlinear wrt suff stats

        Returns
        -------
        L : float
        '''
        return calcELBO_NonlinearTerms(
            alpha=self.alpha, gamma=self.gamma,
            rho=self.rho, omega=self.omega,
            **kwargs)


    def getSerializableParamsForLocalStep(self):
        """ Get compact dict of params for parallel local step.

        Returns
        -------
        Info : dict
        """
        return dict(inferType=self.inferType, 
                    K=self.K, 
                    alphaEbetaRem=self.alpha_E_beta_rem())

    def fillSharedMemDictForLocalStep(self, ShMem=None):
        """ Get dict of shared mem arrays needed for parallel local step. 

        Returns
        -------
        ShMem : dict of RawArray objects
        """
        # No shared memory required here.
        if not isinstance(ShMem, dict):
            ShMem = dict()
        
        alphaEbeta = self.alpha_E_beta()
        if 'alphaEbeta' in ShMem:
            shared_alphaEbeta = sharedMemToNumpyArray(ShMem['alphaEbeta'])
            assert shared_alphaEbeta.size == self.K
            shared_alphaEbeta[:] = alphaEbeta
        else:
            ShMem['alphaEbeta'] = numpyToSharedMemArray(alphaEbeta.copy())
        return ShMem

    def getLocalAndSummaryFunctionHandles(self):
        """ Get function handles for local step and summary step

        Useful for parallelized algorithms.

        Returns
        -------
        calcLocalParams : f handle
        calcSummaryStats : f handle
        """
        return LocalStepManyDocs.calcLocalParams, calcSummaryStats
    # .... end class HDPTopicModel


def calcSummaryStats(Dslice, LP=None, alpha=None,
                     doPrecompEntropy=0,
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
        Relevant fields
        * nDoc : scalar float
            Counts total documents available in provided data.
        * sumLogPi : 1D array, size K
            Entry k equals \sum_{d in docs} E[ \log \pi_{dk} ]
        * sumLogPiRem : scalar float
            Equals sum over docs of probability of inactive topics.

        Also has optional ELBO field when precompELBO is True
        * Hvec : 1D array, size K
            Vector of entropy contributions from each comp.
            Hvec[k] = \sum_{n=1}^N H[q(z_n)], a function of 'resp'
    """
    resp = LP['resp']
    _, K = resp.shape

    SS = SuffStatBag(K=K, D=Dslice.dim)
    SS.setField('nDoc', Dslice.nDoc, dims=None)
    SS.setField('sumLogPi', np.sum(LP['ElogPi'], axis=0), dims='K')
    SS.setField('sumLogPiRem', np.sum(LP['ElogPiRem']), dims=None)

    if doPrecompEntropy:
        Mdict = calcELBO_NonlinearTerms(Data=Dslice, 
            LP=LP, returnMemoizedDict=1)
        SS.setELBOTerm('Hresp', Mdict['Hresp'], dims='K')
        SS.setELBOTerm('slackTheta', Mdict['slackTheta'], dims='K')
        SS.setELBOTerm('slackThetaRem', Mdict['slackThetaRem'], dims=None)
        SS.setELBOTerm('gammalnSumTheta',
            Mdict['gammalnSumTheta'], dims=None)
        SS.setELBOTerm('gammalnTheta', 
            Mdict['gammalnTheta'], dims='K')
        SS.setELBOTerm('gammalnThetaRem', 
            Mdict['gammalnThetaRem'], dims=None)
    return SS