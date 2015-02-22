'''
Bayesian parametric mixture model with finite number of components K.

'''
import numpy as np

from bnpy.allocmodel import AllocModel
from bnpy.suffstats import SuffStatBag
from bnpy.util import logsumexp, np2flatstr, flatstr2np
from bnpy.util import gammaln, digamma, EPS


class FiniteMixtureModel(AllocModel):
    """ Parametric mixture model with finite number of components K

    Attributes
    -------
    * inferType : string {'EM', 'VB', 'moVB', 'soVB'}
        indicates which updates to perform for local/global steps
    * K : int
        number of components
    * gamma : float
        scalar symmetric Dirichlet prior on mixture weights

    Attributes for EM
    --------
    * w : 1D array, size K
        estimated mixture weights for each component
        w[k] > 0 for all k, sum of vector w is equal to one

    Attributes for VB/soVB/moVB
    ---------
    * theta : 1D array, size K
        Estimated parameters for Dirichlet posterior over mix weights
        theta[k] > 0 for all k
    * Elogw : 1D array, size K
        Expected value E[ log w[k] ] for each component
        This is a deterministic function of theta
    """

    def __init__(self, inferType, priorDict=dict()):
        self.inferType = inferType
        self.set_prior(**priorDict)
        self.K = 0

    def set_prior(self, gamma=1.0, **kwargs):
        self.gamma = float(gamma)
        if self.gamma < 1.0 and self.inferType == 'EM':
            raise ValueError("Cannot perform MAP inference if param gamma < 1")

    def get_active_comp_probs(self):
        ''' Return K vector of appearance probabilities for each of the K comps
        '''
        if self.inferType == 'EM':
            return self.w
        else:
            return self.theta / np.sum(self.theta)

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
        if self.inferType.count('EM') > 0:
            # Using point estimates, for EM algorithm
            lpr += np.log(self.w)
            lprPerItem = logsumexp(lpr, axis=1)
            np.exp(lpr - lprPerItem[:, np.newaxis], out=lpr)
            LP['evidence'] = lprPerItem.sum()
        else:
            # Full Bayesian approach, for VB or GS algorithms
            lpr += self.Elogw
            # Calculate exp in numerically stable manner
            lpr -= np.max(lpr, axis=1)[:, np.newaxis]
            np.exp(lpr, out=lpr)
            # Normalize, so rows sum to one
            lpr /= lpr.sum(axis=1)[:, np.newaxis]
        LP['resp'] = lpr
        assert np.allclose(lpr.sum(axis=1), 1)
        return LP


    def get_global_suff_stats(self, Data, LP, doPrecompEntropy=None, **kwargs):
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
            * N : 1D array, size K
                N[k] = expected number of items assigned to comp k

            Also has optional ELBO field when precompELBO is True
            * ElogqZ : 1D array, size K
                Vector of entropy contributions from each comp.
                ElogqZ[k] = \sum_{n=1}^N resp[n,k] log resp[n,k]
        '''
        Nvec = np.sum(LP['resp'], axis=0)
        if hasattr(Data, 'dim'):
            SS = SuffStatBag(K=Nvec.size, D=Data.dim)
        elif hasattr(Data, 'vocab_size'):
            SS = SuffStatBag(K=Nvec.size, D=Data.vocab_size)

        SS.setField('N', Nvec, dims=('K'))
        if doPrecompEntropy is not None:
            ElogqZ_vec = self.E_logqZ(LP)
            SS.setELBOTerm('ElogqZ', ElogqZ_vec, dims=('K'))
        return SS

    def update_global_params_EM(self, SS, **kwargs):
        if np.allclose(self.gamma, 1.0):
            w = SS.N
        else:
            w = SS.N + self.gamma - 1.0  # MAP estimate. Requires gamma>1
        self.w = w / w.sum()
        self.K = SS.K

    def update_global_params_VB(self, SS, **kwargs):
        self.theta = self.gamma + SS.N
        self.Elogw = digamma(self.theta) - digamma(self.theta.sum())
        self.K = SS.K

    def update_global_params_soVB(self, SS, rho, **kwargs):
        thetaStar = self.gamma + SS.N
        self.theta = rho * thetaStar + (1 - rho) * self.theta
        self.Elogw = digamma(self.theta) - digamma(self.theta.sum())
        self.K = SS.K

    def init_global_params(self, Data, K=0, **kwargs):
        ''' Initialize global parameters "from scratch" to prep for learning.

            Will yield uniform distribution (or close to) for all K components,
            by performing a "pseudo" update in which only one observation was
            assigned to each of the K comps.

            Internal Updates
            --------
            Sets attributes w (for EM) or alpha (for VB)

            Returns
            --------
            None.
        '''
        self.K = K
        if self.inferType == 'EM':
            self.w = 1.0 / K * np.ones(K)
        else:
            # one "pseudo count" per state
            self.theta = self.gamma + np.ones(K)
            self.Elogw = digamma(self.theta) - digamma(self.theta.sum())

    def set_global_params(self, hmodel=None, K=None, w=None, beta=None,
                          gamma=None, nObs=10, **kwargs):
        ''' Directly set global parameters to provided values
        '''
        if beta is not None:
            w = beta
        if hmodel is not None:
            self.K = hmodel.allocModel.K
            if self.inferType == 'EM':
                self.w = hmodel.allocModel.w
            else:
                self.theta = hmodel.allocModel.theta
                self.Elogw = digamma(self.theta) - digamma(self.theta.sum())
            return
        else:
            self.K = K
            if self.inferType == 'EM':
                self.w = w
            else:
                if w is not None:
                    self.theta = w * nObs
                elif theta is not None:
                    self.theta = theta
                self.Elogw = digamma(self.theta) - digamma(self.theta.sum())


    def setParamsFromHModel(self, hmodel):
        self.K = hmodel.allocModel.K
        if self.inferType == 'EM':
            self.w = hmodel.allocModel.w
        else:
            self.theta = hmodel.allocModel.theta
            self.Elogw = digamma(self.theta) - digamma(self.theta.sum())

    def setParamsFromScratch(self, K, N=None):
        """
        """
        if N is None:
            N = 1.0 * np.ones(K)
        elif N.ndim == 0:
            N = N * np.ones(K)
        assert N.ndim == 1
        assert N.size == K
        if self.inferType == 'EM':
            self.w = (N + self.gamma) / (N + self.gamma).sum()
        else:
            self.theta = N + self.gamma
            self.Elogw = digamma(self.theta) - digamma(self.theta.sum())

        

    def calc_evidence(self, Data, SS, LP, todict=False, **kwargs):
        if self.inferType == 'EM':
            return LP['evidence'] + self.log_pdf_dirichlet(self.w)
        elif self.inferType.count('VB') > 0:
            evW = self.E_logpW() - self.E_logqW()
            if SS.hasELBOTerm('ElogqZ'):
                ElogqZ = np.sum(SS.getELBOTerm('ElogqZ'))
            else:
                ElogqZ = np.sum(self.E_logqZ(LP))
            if SS.hasAmpFactor():
                evZ = self.E_logpZ(SS) - SS.ampF * ElogqZ
            else:
                evZ = self.E_logpZ(SS) - ElogqZ
            return evZ + evW
        else:
            raise NotImplementedError(
                'Unrecognized inferType ' + self.inferType)

    def E_logpZ(self, SS):
        ''' Bishop PRML eq. 10.72
        '''
        return np.inner(SS.N, self.Elogw)

    def E_logqZ(self, LP):
        ''' Bishop PRML eq. 10.75
        '''
        return np.sum(LP['resp'] * np.log(LP['resp'] + EPS), axis=0)

    def E_logpW(self):
        ''' Bishop PRML eq. 10.73
        '''
        return gammaln(self.K * self.gamma) \
            - self.K * gammaln(self.gamma) + \
            (self.gamma - 1) * self.Elogw.sum()

    def E_logqW(self):
        ''' Bishop PRML eq. 10.76
        '''
        return gammaln(self.theta.sum()) - gammaln(self.theta).sum() \
            + np.inner((self.theta - 1), self.Elogw)

    def log_pdf_dirichlet(self, wvec=None, avec=None):
        ''' Return scalar log probability for Dir(wvec | avec)
        '''
        if wvec is None:
            wvec = self.w
        if avec is None:
            avec = self.gamma * np.ones(self.K)
        logC = gammaln(np.sum(avec)) - np.sum(gammaln(avec))
        return logC + np.sum((avec - 1.0) * np.log(wvec))

    def get_info_string(self):
        ''' Returns one-line human-readable terse description of this object
        '''
        msgPattern = 'Finite mixture with K=%d. Dir prior param %.2f'
        return msgPattern % (self.K, self.gamma)

    def to_dict(self):
        if self.inferType == 'EM':
            return dict(w=self.w)
        elif self.inferType.count('VB') > 0:
            return dict(theta=self.theta)
        elif self.inferType.count('GS') > 0:
            return dict(theta=self.theta)
        return dict()

    def from_dict(self, myDict):
        self.inferType = myDict['inferType']
        self.K = myDict['K']
        if self.inferType == 'EM':
            self.w = myDict['w']
        else:
            self.theta = myDict['theta']
            self.Elogw = digamma(self.theta) - digamma(self.theta.sum())

    def get_prior_dict(self):
        return dict(gamma=self.gamma, K=self.K)

    # ----    Sampler functions
    def sample_local_params(self, obsModel, Data, SS, LP, PRNG, **kwargs):
        ''' Sample local assignments for each data item

        Returns
        --------
        LP : dict
            Local parameters, with updated fields
            * Z : 1D array, size N
                Z[n] = k iff item n is assigned to component k
        '''
        Z = LP['Z']
        # Iteratively sample data allocations
        for dataindex in xrange(Data.nObs):
            x = Data.X[dataindex]

            # de-update current assignment and suff stats
            kcur = Z[dataindex]
            SS.N[kcur] -= 1
            obsModel.decrementSS(SS, kcur, x)

            # Calculate probs
            alloc_prob = self.getConditionalProbVec_Unnorm(SS)
            pvec = obsModel.calcPredProbVec_Unnorm(SS, x)
            pvec *= alloc_prob
            pvec /= np.sum(pvec)

            # sample new allocation
            knew = PRNG.choice(SS.K, p=pvec)

            # update with new assignment
            SS.N[knew] += 1
            obsModel.incrementSS(SS, knew, x)
            Z[dataindex] = knew

        LP['Z'] = Z
        return LP, SS

    def getConditionalProbVec_Unnorm(self, SS):
        ''' Returns a K vector of positive values \propto p(z_i|z_-i)
        '''
        return SS.N + self.gamma

    def calcMargLik(self, SS):
        ''' Calculate marginal likelihood of assignments, summed over all comps
        '''
        theta = self.gamma + SS.N
        cPrior = gammaln(SS.K * self.gamma) - SS.K * gammaln(self.gamma)
        cPost = gammaln(np.sum(theta)) - np.sum(gammaln(theta))
        return cPrior - cPost
