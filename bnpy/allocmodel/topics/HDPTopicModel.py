import numpy as np
import logging
Log = logging.getLogger('bnpy')

from bnpy.allocmodel.AllocModel import AllocModel
from bnpy.allocmodel.mix.DPMixtureModel import convertToN0
from bnpy.suffstats import SuffStatBag
from bnpy.util import digamma, gammaln
from bnpy.util import as1D

import LocalStepManyDocs
import OptimizerRhoOmega

from bnpy.util.StickBreakUtil import rho2beta
from bnpy.util.NumericUtil import calcRlogRdotv, calcRlogR
from bnpy.util.NumericUtil import calcRlogRdotv_allpairs
from bnpy.util.NumericUtil import calcRlogRdotv_specificpairs
from bnpy.util.NumericUtil import calcRlogR_allpairs, calcRlogR_specificpairs


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
        abeta : 1D array, size K + 1
            abeta[k] gives scaled parameter for comp. k under this model.
            abeta[K] (last index) is aggregated over all inactive topics.
        '''
        if not hasattr(self, 'alphaEbeta'):
            self.alphaEbeta = self.alpha * self.E_beta()
        return self.alphaEbeta

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
        self.alpha_E_beta()  # create cached copy
        LP = LocalStepManyDocs.calcLocalParamsForDataSlice(
            Data, LP, self, **kwargs)
        assert 'resp' in LP
        assert 'DocTopicCount' in LP
        return LP

    def updateLPGivenDocTopicCount(self, LP, DocTopicCount):
        ''' Update local parameters given doc-topic counts for many docs.

        Returns
        --------
        LP : dict of local params, with updated fields
            * theta : 2D array, nDoc x K
            * thetaRem : scalar
            * ElogPi : 2D array, nDoc x K
            * ElogPiRem : scalar
        '''
        alphaEbeta = self.alpha * self.E_beta()

        theta = DocTopicCount + alphaEbeta[:-1]
        digammaSumTheta = digamma(theta.sum(axis=1) + alphaEbeta[-1])
        ElogPi = digamma(theta) - digammaSumTheta[:, np.newaxis]
        ElogPiRem = digamma(alphaEbeta[-1]) - digammaSumTheta
        LP['theta'] = theta
        LP['thetaRem'] = alphaEbeta[-1]
        LP['ElogPi'] = ElogPi
        LP['ElogPiRem'] = ElogPiRem
        LP['digammaSumTheta'] = digammaSumTheta
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
            Summarizes for this mixture model, with fields
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
        SS.setField('nDoc', Data.nDoc, dims=None)
        SS.setField('sumLogPi', np.sum(LP['ElogPi'], axis=0), dims='K')
        SS.setField('sumLogPiRem', np.sum(LP['ElogPiRem']), dims=None)

        if doPrecompEntropy:
            Hvec = calcHvec(Data, LP)
            SS.setELBOTerm('Hvec', Hvec, dims='K')

            slackTheta, slackThetaRem = calcMemoPieces_L_slacklocal(LP=LP)
            SS.setELBOTerm('slackTheta', slackTheta, dims='K')
            SS.setELBOTerm('slackThetaRem', slackThetaRem, dims=None)

            glnSumTheta, glnTheta, glnThetaRem = calcMemoPieces_L_alloclocal(
                theta=LP['theta'], thetaRem=LP['thetaRem'])
            SS.setELBOTerm('gammalnSumTheta', glnSumTheta, dims=None)
            SS.setELBOTerm('gammalnTheta', glnTheta, dims='K')
            SS.setELBOTerm('gammalnThetaRem', glnThetaRem, dims=None)

        if doPrecompMergeEntropy:
            if mPairIDs is None:
                raise NotImplementedError("TODO: all pairs for merges")

            ElogqZMat = self.calcElogqZForMergePairs(
                LP['resp'], Data, mPairIDs)
            SS.setMergeTerm('ElogqZ', ElogqZMat, dims=('K', 'K'))

            alphaEbeta = self.alpha_E_beta()

            sumLogPi = np.zeros((SS.K, SS.K))
            gammalnTheta = np.zeros((SS.K, SS.K))
            slack_NmT = np.zeros((SS.K, SS.K))
            for (kA, kB) in mPairIDs:
                theta_vec = LP['theta'][:, kA] + LP['theta'][:, kB]
                ElogPi_vec = digamma(theta_vec) - LP['digammaSumTheta']
                gammalnTheta[kA, kB] = np.sum(gammaln(theta_vec))
                sumLogPi[kA, kB] = np.sum(ElogPi_vec)
                ElogPi_vec *= alphaEbeta[kA] + alphaEbeta[kB]
                slack_NmT[kA, kB] = -1 * np.sum(ElogPi_vec)
            SS.setMergeTerm('gammalnTheta', gammalnTheta, dims=('K', 'K'))
            SS.setMergeTerm('sumLogPi', sumLogPi, dims=('K', 'K'))
            SS.setMergeTerm('slackNminusTheta', slack_NmT, dims=('K', 'K'))

            # for (kA, kB) in mPairIDs:
            #  self.verifySSForMergePair(Data, SS, LP, kA, kB)

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

        slack_direct = mergeSS.getELBOTerm('slackNminusTheta')[kA]
        slack_cached = SS.getMergeTerm('slackNminusTheta')[kA, kB]
        assert np.allclose(slack_direct, slack_cached)

        ElogqZ_direct = mergeSS.getELBOTerm('ElogqZ')[kA]
        ElogqZ_cached = SS.getMergeTerm('ElogqZ')[kA, kB]
        assert np.allclose(ElogqZ_direct, ElogqZ_cached)

    def calcElogqZForMergePairs(self, resp, Data, mPairIDs):
        ''' Calculate resp entropy terms for all candidate merge pairs

            Returns
            ---------
            ElogqZ : 2D array, size K x K
        '''
        if hasattr(Data, 'word_count'):
            if mPairIDs is None:
                ElogqZMat = calcRlogRdotv_allpairs(resp, Data.word_count)
            else:
                ElogqZMat = calcRlogRdotv_specificpairs(
                    resp, Data.word_count, mPairIDs)
        else:
            if mPairIDs is None:
                ElogqZMat = calcRlogR_allpairs(resp)
            else:
                ElogqZMat = calcRlogR_specificpairs(resp, mPairIDs)
        return ElogqZMat

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
            rho, omega, f, Info = OptimizerRhoOmega.find_optimum_multiple_tries(
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
        #self.rho = (1-rho) * self.rho + rho * rhoStar
        #self.omega = (1-rho) * self.omega + rho * omegaStar
        g1 = (1 - rho) * (self.rho * self.omega) + rho * (rhoStar * omegaStar)
        g0 = (1 - rho) * ((1 - self.rho) * self.omega) + \
            rho * ((1 - rhoStar) * omegaStar)
        self.rho = g1 / (g1 + g0)
        self.omega = g1 + g0
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

    def setParamsFromBeta(self, K, beta=None):
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
        assert beta.size == K
        assert np.allclose(np.sum(beta), 1.0)
        self.K = int(K)

        # Append in small remaining/leftover mass
        betaRem = np.minimum(1.0 / (2 * K), 0.05)
        betaWithRem = np.hstack([beta * (1.0 - betaRem), betaRem])
        assert np.allclose(np.sum(betaWithRem), 1.0)

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
        """ Calculate ELBO objective function value for provided state.

        Returns
        -------
        L : float
            Represents sum of all terms in ELBO objective.
        """
        Lallocglobal = L_allocglobal(
            self.rho, self.omega, 
            gamma=self.gamma, 
            alpha=self.alpha, 
            nDoc=SS.nDoc)
        Lslackglobal = L_slackglobal(
            alphaEbeta=self.alpha_E_beta(),
            sumLogPi=SS.sumLogPi, sumLogPiRem=SS.sumLogPiRem,
            )
        if SS.hasELBOTerms():
            Lentropy = SS.getELBOTerm('Hvec').sum()
            Lalloclocal = L_alloclocal(
                SS.getELBOTerm('gammalnSumTheta'),
                SS.getELBOTerm('gammalnTheta'),
                SS.getELBOTerm('gammalnThetaRem'))
            Lslacklocal = L_slacklocal(
                SS.getELBOTerm('slackTheta'),
                SS.getELBOTerm('slackThetaRem')
                )
        else:
            Lentropy = L_entropy(Data, LP)
            Lalloclocal = L_alloclocal(
                theta=LP['theta'],
                thetaRem=LP['thetaRem'])
            Lslacklocal = L_slacklocal(LP=LP)

        if todict:
            return dict(Lalloclocal=Lalloclocal,
                        Lallocglobal=Lallocglobal,
                        Lentropy=Lentropy,
                        Lslack=Lslackglobal+Lslacklocal,
                        )
        return Lallocglobal + Lalloclocal + Lentropy +\
            Lslackglobal + Lslacklocal

    """
    def E_cDir_alphabeta__Numeric(self):
        ''' Numeric integration of the expectation
        '''
        g1 = self.rho * self.omega
        g0 = (1 - self.rho) * self.omega
        assert self.K <= 2
        if self.K == 1:
            us = np.linspace(1e-14, 1 - 1e-14, 1000)
            logpdf = gammaln(g1 + g0) - gammaln(g1) - gammaln(g0) \
                + (g1 - 1) * np.log(us) + (g0 - 1) * np.log(1 - us)
            pdf = np.exp(logpdf)
            b1 = us
            bRem = 1 - us
            Egb1 = np.trapz(gammaln(self.alpha * b1) * pdf, us)
            EgbRem = np.trapz(gammaln(self.alpha * bRem) * pdf, us)
            EcD = gammaln(self.alpha) - Egb1 - EgbRem
        return EcD

    def E_cDir_alphabeta__MonteCarlo(self, S=1000, seed=123):
        ''' Monte Carlo approximation to the expectation
        '''
        PRNG = np.random.RandomState(seed)
        g1 = self.rho * self.omega
        g0 = (1 - self.rho) * self.omega
        cD_abeta = np.zeros(S)
        for s in range(S):
            u = PRNG.beta(g1, g0)
            u = np.minimum(np.maximum(u, 1e-14), 1 - 1e-14)
            beta = np.hstack([u, 1.0])
            beta[1:] *= np.cumprod(1.0 - u)
            cD_abeta[s] = gammaln(
                self.alpha) - gammaln(self.alpha * beta).sum()
        return np.mean(cD_abeta)

    def E_cDir_alphabeta__Surrogate(self):
        calpha = gammaln(self.alpha) + (self.K + 1) * np.log(self.alpha)

        g1 = self.rho * self.omega
        g0 = (1 - self.rho) * self.omega
        digammaBoth = digamma(g1 + g0)
        ElogU = digamma(g1) - digammaBoth
        Elog1mU = digamma(g0) - digammaBoth
        OFFcoef = OptimizerRhoOmega.kvec(self.K)
        cRest = np.sum(ElogU) + np.inner(OFFcoef, Elog1mU)

        return calpha + cRest
    """

    # .... end class HDPTopicModel


def L_allocglobal(rho, omega, gamma=1.0, alpha=1.0, nDoc=0):
    """ Compute global-only term of the ELBO objective.
    """
    K = rho.size
    eta1 = rho * omega
    eta0 = (1-rho) * omega
    digammaBoth = digamma(eta1 + eta0)
    ElogU = digamma(eta1) - digammaBoth
    Elog1mU = digamma(eta0) - digammaBoth

    ONcoef = nDoc + 1.0 - eta1
    OFFcoef = nDoc * OptimizerRhoOmega.kvec(K) + gamma - eta0

    calpha = gammaln(alpha) + (K + 1) * np.log(alpha)
    cDiff = K * c_Beta(1, gamma) - c_Beta(eta1, eta0)

    return calpha + \
           cDiff + \
           np.inner(ONcoef, ElogU) + np.inner(OFFcoef, Elog1mU)

def L_slackglobal(alphaEbeta=None, 
                  sumLogPi=None, sumLogPiRem=0,
                  rho=None, omega=None, alpha=1.0):
    """ Returns arrays whos sum is ELBO term.

    Returns
    -------
    L : scalar float
    """
    if alphaEbeta is None:
        alphaEbeta = alpha * Ebeta
    Svec = alphaEbeta[:-1] * sumLogPi
    Srem = alphaEbeta[-1] * sumLogPiRem
    return Svec.sum() + Srem

def L_slacklocal(slackTheta=None, slackThetaRem=None, LP=None):
    ''' Calculate part of ELBO depending on doc-topic slack terms

    Returns
    -------
    L : scalar float
    '''
    if slackTheta is None:
        slackTheta, slackThetaRem = calcMemoPieces_L_slacklocal(LP)
    return slackTheta.sum() + slackThetaRem

def calcMemoPieces_L_slacklocal(LP):
    """ Calculate part of ELBO depending on doc-topic slack terms.

    Returns
    -------
    slackTheta : 1D array, size K
    slackThetaRem : float
    """
    slackTheta = LP['DocTopicCount'] - LP['theta']
    slackTheta *= LP['ElogPi']
    slackTheta = np.sum(slackTheta, axis=0)
    slackThetaRem = -1 * np.sum(LP['thetaRem'] * LP['ElogPiRem'])
    return slackTheta, slackThetaRem

def L_alloclocal(
        gammalnSumTheta=None,
        gammalnTheta=None,
        gammalnThetaRem=None,
        theta=None, thetaRem=None, **kwargs):
    """ Calculate local allocation term of the ELBO objective.

    Returns
    -------
    L : scalar float
    """
    if theta is not None:
        gammalnSumTheta, gammalnTheta, gammalnThetaRem =\
            calcMemoPieces_L_alloclocal(theta, thetaRem)
    cDiff = gammalnSumTheta - gammalnTheta.sum() - gammalnThetaRem
    return -1 * cDiff

def calcMemoPieces_L_alloclocal(theta, thetaRem):
    sumTheta = np.sum(theta, axis=1) + thetaRem
    gammalnSumTheta = np.sum(gammaln(sumTheta))
    gammalnTheta = np.sum(gammaln(theta), axis=0)    
    nDoc = theta.shape[0]
    gammalnThetaRem = nDoc * gammaln(thetaRem)
    return gammalnSumTheta, gammalnTheta, gammalnThetaRem

def L_entropy(Data=None, LP=None, resp=None):
    """ Calculate entropy of soft assignments term in ELBO objective.

    Returns
    -------
    L_entropy : scalar float
    """
    return calcHvec(Data, LP, resp).sum()

def calcHvec(Data=None, LP=None, resp=None):
    """ Calculate vector whos sume is the entropy term in ELBO objective.

    Returns
    -------
    Hvec : 1D array, size K
    """
    if LP is not None:
        resp = LP['resp']
    if hasattr(Data, 'word_count'):
        Hvec = -1 * calcRlogRdotv(resp, Data.word_count)
    else:
        Hvec = -1 * calcRlogR(resp)
    return Hvec

def E_cDalphabeta_surrogate(alpha, rho, omega):
    """ Compute expected value of cumulant function of alpha * beta.

    Returns
    -------
    csur : scalar float
    """
    K = rho.size
    eta1 = rho * omega
    eta0 = (1-rho) * omega
    digammaBoth = digamma(eta1 + eta0)
    ElogU = digamma(eta1) - digammaBoth
    Elog1mU = digamma(eta0) - digammaBoth
    OFFcoef = OptimizerRhoOmega.kvec(K)
    calpha = gammaln(alpha) + (K + 1) * np.log(alpha)
    return calpha + np.sum(ElogU) + np.inner(OFFcoef, Elog1mU)

def c_Beta(a1, a0):
    ''' Evaluate cumulant function of the Beta distribution

        When input is vectorized, we compute sum over all entries.

        Returns
        -------
        c : scalar real
    '''
    return np.sum(gammaln(a1 + a0)) - np.sum(gammaln(a1)) - np.sum(gammaln(a0))


def c_Dir(AMat, arem=None):
    ''' Evaluate cumulant function of the Dir distribution

    When input is vectorized, we compute sum over all entries.

    Returns
    -------
    c : scalar real
    '''
    AMat = np.asarray(AMat)
    D = AMat.shape[0]
    if arem is None:
        if AMat.ndim == 1:
            return gammaln(np.sum(AMat)) - np.sum(gammaln(AMat))
        else:
            return np.sum(gammaln(np.sum(AMat, axis=1))) - np.sum(gammaln(AMat))

    return  np.sum(gammaln(np.sum(AMat, axis=1) + arem)) \
        - np.sum(gammaln(AMat)) \
        - D * np.sum(gammaln(arem))


def c_Dir__big(AMat, arem):
    AMatBig = np.hstack([AMat, arem * np.ones(AMat.shape[0])[:, np.newaxis]])
    return np.sum(gammaln(np.sum(AMatBig, axis=1))) - np.sum(gammaln(AMatBig))


def c_Dir__slow(AMat, arem):
    c = 0
    for d in xrange(AMat.shape[0]):
        avec = np.hstack([AMat[d], arem])
        c += gammaln(np.sum(avec)) - np.sum(gammaln(avec))
    return c
