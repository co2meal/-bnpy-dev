import numpy as np
import scipy.linalg
from scipy.special import gammaln, digamma

from bnpy.suffstats import ParamBag, SuffStatBag
from bnpy.util import LOGTWO, LOGPI, LOGTWOPI, EPS
from bnpy.util import dotATA, dotATB, dotABT
from bnpy.util import as1D, as2D, as3D, toCArray
from bnpy.util import numpyToSharedMemArray, fillSharedMemArray
from bnpy.util.SparseRespStatsUtil import calcSpRXXT
from AbstractObsModel import AbstractObsModel

class GaussRegressYFromFixedXObsModel(AbstractObsModel):

    ''' Observation model for producing 1D observations from covariates

    Attributes for Prior
    --------------------
    w_E : 1D array, size E
        mean of the regression weights
    P_EE : 2D array, size E x E
        precision matrix for regression weights
    nu : positive float
        effective sample size of prior on regression precision
    tau : positive float
        effective scale parameter of prior on regression precision

    Attributes for Point Estimation
    -------------------------------
    TODO

    Attributes for Approximate Posterior
    ------------------------------------
    w_E : 1D array, size E
    P_EE : 2D array, size E x E
    nu : positive float
    tau : positive float
    '''

    def __init__(self, inferType='VB', D=0, Data=None, **PriorArgs):
        ''' Initialize bare obsmodel with valid prior hyperparameters.

        Returns
        -------
        obsmodel : bare observation model
            Resulting object lacks either EstParams or Post attributes.
            which must be created separately (see init_global_params).
        '''
        if Data is not None:
            self.D = Data.dim
        else:
            self.D = int(D)
        self.E = self.D + 1
        self.K = 0
        self.inferType = inferType
        self.createPrior(Data, **PriorArgs)
        self.Cache = dict()


    def createPrior(
            self, Data, nu=0, tau=None,
            w_E=0,
            P_EE=None, P_diag_E=None, P_diag_val=1.0,
            **kwargs):
        ''' Initialize Prior ParamBag attribute.

        Post Condition
        --------------
        Adds valid attribute named Prior of type ParamBag
        '''
        D = self.D
        E = self.E

        # Init parameters of Wishart prior on delta
        nu = np.maximum(nu, 1e-9)
        tau = np.maximum(tau, 1e-9)

        # Initialize precision matrix of the weight vector
        if P_EE is not None:
            P_EE = np.asarray(P_EE)
        elif P_diag_E is not None:
            P_EE = np.diag(np.asarray(P_diag_E))
        else:
            P_EE = np.diag(P_diag_val * np.ones(E))
        assert P_EE.ndim == 2
        assert P_EE.shape == (E,E)

        # Initialize mean of the weight vector
        w_E = as1D(np.asarray(w_E))
        if w_E.size < E:
            w_E = np.tile(w_E, E)[:E]
        assert w_E.ndim == 1
        assert w_E.size == E

        self.Prior = ParamBag(K=0, D=D, E=E)
        self.Prior.setField('nu', nu, dims=None)
        self.Prior.setField('tau', tau, dims=None)
        self.Prior.setField('w_E', w_E, dims=('E'))
        self.Prior.setField('P_EE', P_EE, dims=('E', 'E'))

        Pw_E = np.dot(P_EE, w_E)
        wPw_1 = np.dot(w_E, Pw_E)
        self.Prior.setField('Pw_E', Pw_E, dims=('E'))
        self.Prior.setField('wPw_1', wPw_1, dims=None)


    def get_name(self):
        return 'GaussRegressYFromFixedX'

    def get_info_string(self):
        return 'Gaussian regression model for 1D output y from fixed input x'

    def get_info_string_prior(self):
        msg = 'Gaussian-Wishart joint prior on regression weights/prec\n'
        msg += "Wishart on precision scalar\n"
        msg += "     nu = %.3g\n" % (self.Prior.nu)
        msg += "    tau = %.3g\n" % (self.Prior.tau)
        msg += "   mean = %.3g\n" % (E_d(nu=self.Prior.nu, tau=self.Prior.tau))
        w_E_str = ' '.join(['%5.2f' % x for x in self.Prior.w_E])
        P_EE_str = ' '.join(['%.3e' % x for x in np.diag(self.Prior.P_EE)])
        msg += "Gaussian on regression weight vector\n"
        msg += "          mean = %s\n" % w_E_str
        msg += "    diag[prec] = %s\n" % P_EE_str
        return msg


    def setPostFactors(self, obsModel=None, SS=None, LP=None, Data=None,
                       nu_K=None, tau_K=None, w_KE=None, P_KEE=None,
                       **kwargs):
        ''' Set attribute Post to provided values.
        '''
        self.ClearCache()
        if obsModel is not None:
            if hasattr(obsModel, 'Post'):
                self.Post = obsModel.Post.copy()
                self.K = self.Post.K
            else:
                self.setPostFromEstParams(obsModel.EstParams)
            return

        if LP is not None and Data is not None:
            SS = self.calcSummaryStats(Data, None, LP)
        if SS is not None:
            self.updatePost(SS)
        else:
            nu_K = as1D(nu_K)
            tau_K = as1D(tau_K)
            w_KE = as2D(w_KE)
            P_KEE = as3D(P_KEE)

            K = nu_K.size
            self.Post = ParamBag(K=K, D=self.D, E=self.D+1)
            self.Post.setField('nu_K', nu_K, dims=('K'))
            self.Post.setField('tau_K', tau_K, dims=('K'))
            self.Post.setField('w_KE', w_KE, dims=('K', 'E'))
            self.Post.setField('P_KEE', P_KEE, dims=('K', 'E', 'E'))
        self.K = self.Post.K

    def calcSummaryStats(self, Data, SS, LP, **kwargs):
        ''' Calculate summary statistics for given dataset and local parameters

        Returns
        --------
        SS : SuffStatBag object, with K components.
        '''
        return calcSummaryStats(Data, SS, LP, **kwargs)

    def getDatasetScale(self, SS):
        ''' Get number of observed scalars in dataset from suff stats.

        Used for normalizing the ELBO so it has reasonable range.

        Returns
        ---------
        s : scalar positive integer
        '''
        return SS.N.sum() * SS.D

    def calcLogSoftEvMatrix_FromPost(self, Data, **kwargs):
        ''' Compute expected log soft evidence of each item under each cluster

        Returns
        -------
        E_log_soft_ev_NK : 2D array, size N x K
        '''
        return calcLogSoftEvMatrix_FromPost(
            Data,
            nu_K=self.Post.nu_K,
            tau_K=self.Post.tau_K, 
            w_KE=self.Post.w_KE,
            P_KEE=self.Post.P_KEE)

    def updatePost(self, SS):
        ''' Update attribute Post for all comps given suff stats.

        Optimizes the variational objective for approximating the posterior

        Post Condition
        --------------
        Attributes K and Post updated in-place.
        '''
        self.ClearCache()
        if not hasattr(self, 'Post') or self.Post.K != SS.K:
            self.Post = ParamBag(K=SS.K, D=SS.D, E=SS.D+1)

        nu_K, tau_K, w_KE, P_KEE = calcPostParamsFromSS(
            SS=SS, Prior=self.Prior)
        self.Post.setField('nu_K', nu_K, dims=('K'))
        self.Post.setField('tau_K', tau_K, dims=('K'))
        self.Post.setField('w_KE', w_KE, dims=('K', 'E'))
        self.Post.setField('P_KEE', P_KEE, dims=('K', 'E', 'E'))
        self.K = SS.K


    def calcELBO_Memoized(self, SS, returnVec=0, afterMStep=False, **kwargs):
        """ Calculate obsModel's objective using suff stats SS and Post.

        Args
        -------
        SS : bnpy SuffStatBag
        afterMStep : boolean flag
            if 1, elbo calculated assuming M-step just completed

        Returns
        -------
        elbo_K : scalar float
            Equal to E[ log p(x) + log p(phi) - log q(phi)]
        """
        elbo_K = np.zeros(SS.K)
        Post = self.Post
        Prior = self.Prior
        for k in xrange(SS.K):
            elbo_K[k] = - (0.5 * LOGTWOPI) * SS.N_K[k] \
                + c_Func(
                    nu=self.Prior.nu,
                    tau=self.Prior.tau,
                    w_E=self.Prior.w_E,
                    P_EE=self.Prior.P_EE) \
                - c_Func(
                    nu=self.Post.nu_K[k],
                    tau=self.Post.tau_K[k],
                    w_E=self.Post.w_KE[k],
                    P_EE=self.Post.P_KEE[k])

            if not afterMStep:
                Post_Pw_E_k =  np.dot(Post.P_KEE[k], Post.w_KE[k])
                Post_wPw_1_k = np.dot(Post.w_KE[k], Post_Pw_E_k)

                # A : log delta term
                A_1 = SS.N_K[k] + Prior.nu - Post.nu_K[k]
                elbo_K[k] -= 0.5 * A_1 * self.GetCached('E_log_d', k)

                # B : delta term
                B_1 = SS.yy_K[k] \
                    + Prior.tau + Prior.wPw_1 \
                    - Post.tau_K[k] - Post_wPw_1_k
                elbo_K[k] -= 0.5 * B_1 * self.GetCached('E_d', k)

                # C : delta * w_E term, size E
                C_E = SS.yx_KE[k] + \
                    + Prior.Pw_E \
                    - Post_Pw_E_k
                elbo_K[k] += np.inner(C_E, self.GetCached('E_d_w', k))

                # D : delta * w_E w_E term, size E * E
                D_EE = SS.xxT_KEE[k] + Prior.P_EE - Post.P_KEE[k]
                elbo_K[k] -= 0.5 * np.trace(
                    np.dot(D_EE, self.GetCached('E_d_w_wT', k)))

        if returnVec:
            return elbo_K
        return np.sum(elbo_K)

    def _unpack_params(self, k=None):
        ''' Unpack internal attributes into dict of parameters

        Args
        ----
        k : int or None
            if None, use Prior
            otherwise, use specific cluster identified by k

        Returns
        -------
        pdict : dict, with fields
            nu, tau, w_E, P_EE
        '''
        if k is None:
            nu = self.Prior.nu
            tau = self.Prior.tau
            w_E = self.Prior.w_E
            P_EE = self.Prior.P_EE
        else:
            nu = self.Post.nu_K[k]
            tau = self.Post.tau_K[k]
            w_E = self.Post.w_KE[k]
            P_EE = self.Post.P_KEE[k]
        return dict(nu=nu, tau=tau, w_E=w_E, P_EE=P_EE)

    def _E_log_d(self, k=None):
        ''' Cacheable function computing expectation of log delta

        Returns
        -------
        scalar
        '''
        pdict = self._unpack_params(k=k)
        return E_log_d(**pdict)

    def _E_d(self, k=None):
        ''' Cacheable function computing expectation of delta

        Returns
        -------
        scalar
        '''
        pdict = self._unpack_params(k=k)
        return E_d(**pdict)

    def _E_d_w(self, k=None):
        ''' Cacheable function computing expectation of delta * weight vector

        Returns
        -------
        val_E : 1D array 
        '''
        pdict = self._unpack_params(k=k)
        return E_d_w(**pdict)

    def _E_d_w_wT(self, k=None):
        ''' Cacheable function computing expectation of delta * outer product

        Returns
        -------
        val_EE : 2D array, size E x E
        '''
        pdict = self._unpack_params(k=k)
        return E_d_w_wT(**pdict)


def calcLocalParams(Dslice, **kwargs):
    E_log_soft_ev_NK = calcLogSoftEvMatrix_FromPost(Dslice, **kwargs)
    LP = dict(E_log_soft_ev=E_log_soft_ev_NK)
    return LP


def calcLogSoftEvMatrix_FromPost(
        Dslice,
        E_log_d_K=None,
        E_d_K=None,
        nu_K=None,
        tau_K=None,
        w_KE=None,
        P_KEE=None,
        chol_P_KEE=None,
        **kwargs):
    ''' Calculate expected log soft ev matrix under approximate posterior

    Returns
    -------
    E_log_soft_ev_NK : 2D array, size N x K
    '''
    if not hasattr(Dslice, 'X_NE'):
        Dslice.X_NE = np.hstack([Dslice.X, np.ones(Dslice.nObs)[:,np.newaxis]])

    if E_log_d_K is None:
        K = nu_K.size
        E_log_d_K = np.zeros(K)
        for k in range(K):
            E_log_d_K[k] = E_log_d(nu=nu_K[k], tau=tau_K[k])
        E_d_K = np.zeros(K)
        for k in range(K):
            E_d_K[k] = E_d(nu=nu_K[k], tau=tau_K[k])
    else:
        K = E_log_d_K.size
    E_log_soft_ev_NK = np.zeros((Dslice.nObs, K))
    for k in xrange(K):
        E_log_soft_ev_NK[:, k] = (
            - 0.5 * LOGTWOPI \
            + 0.5 * E_log_d_K[k] 
            - 0.5 * E_mahal_dist_N(
                Dslice.Y, Dslice.X_NE,
                E_d=E_d_K[k],
                w_E=w_KE[k],
                P_EE=P_KEE[k],
                )
            )
    return E_log_soft_ev_NK

def E_mahal_dist_N(Y_N, X_NE,
        E_d=None,
        nu=None,
        tau=None,
        w_E=None,
        P_EE=None,
        chol_P_EE=None):
    ''' Calculate expected mahalanobis distance under regression model

    $$
    d_n = delta_k (y_n - w_k^T x_n)^2
    $$

    Returns
    -------
    d_N : 1D array, size N
    '''
    if E_d is None:
        E_d = E_d(nu=nu, tau=tau)
    if chol_P_EE is None:
        chol_P_EE = scipy.linalg.cholesky(P_EE, lower=True)

    # Squared diff term
    sq_diff_YX_N = Y_N[:,0] - np.dot(X_NE, w_E)
    sq_diff_YX_N *= sq_diff_YX_N
    sq_diff_YX_N *= E_d

    xPx_EN = np.linalg.solve(chol_P_EE, X_NE.T)
    xPx_EN *= xPx_EN
    xPx_N = np.sum(xPx_EN, axis=0) 

    E_mahal_dist_N = sq_diff_YX_N
    E_mahal_dist_N += xPx_N
    assert E_mahal_dist_N.min() > -1e-9
    return E_mahal_dist_N

def calcSummaryStats(Data, SS, LP, **kwargs):
    ''' Calculate summary statistics for given dataset and local parameters

    Returns
    --------
    SS : SuffStatBag object, with K components.
    '''
    if not hasattr(Data, 'X_NE'):
        Data.X_NE = np.hstack([Data.X, np.ones(Data.nObs)[:,np.newaxis]])

    Y_N = Data.Y
    X_NE = Data.X_NE
    E = X_NE.shape[1]

    if 'resp' in LP:
        # Dense responsibility calculations
        resp = LP['resp']
        K = resp.shape[1]
        S_yy_K = dotATB(resp, np.square(Y_N)).flatten()
        S_yx_KE = dotATB(resp, Y_N * X_NE)

        # Expected outer product
        S_xxT_KEE = np.zeros((K, E, E))
        sqrtResp_k_N = np.sqrt(resp[:, 0])
        sqrtR_X_k_NE = sqrtResp_k_N[:, np.newaxis] * X_NE
        S_xxT_KEE[0] = dotATA(sqrtR_X_k_NE)
        for k in xrange(1, K):
            np.sqrt(resp[:, k], out=sqrtResp_k_N)
            np.multiply(sqrtResp_k_N[:, np.newaxis], X_NE, out=sqrtR_X_k_NE)
            S_xxT_KEE[k] = dotATA(sqrtR_X_k_NE)
    else:
        raise ValueError("TODO")
        spR = LP['spR']
        K = spR.shape[1]

    if SS is None:
        SS = SuffStatBag(K=K, D=Data.dim, E=E)
    elif not hasattr(SS, 'E'):
        SS._Fields.E = E
    SS.setField('xxT_KEE', S_xxT_KEE, dims=('K', 'E', 'E'))
    SS.setField('yx_KE', S_yx_KE, dims=('K', 'E'))
    SS.setField('yy_K', S_yy_K, dims=('K'))
    # Expected count for each k
    # Usually computed by allocmodel. But just in case...
    if not hasattr(SS, 'N'):
        if 'resp' in LP:
            SS.setField('N', LP['resp'].sum(axis=0), dims='K')
        else:
            SS.setField('N', as1D(toCArray(LP['spR'].sum(axis=0))), dims='K')

    SS.setField("N_K", SS.N, dims="K")
    return SS


def calcPostParamsFromSS(
        SS=None, xxT_KEE=None, yx_KE=None, yy_K=None, N_K=None,
        Prior=None,
        **kwargs):
    ''' Calc updated posterior parameters for all clusters from suff stats

    Returns
    --------
    nu_K : 1D array, size K
    tau_K : 1D array, size K
    w_KE : 2D array, size K x E
    P_KEE : 3D array, size K x E x E
    '''
    K = SS.K
    E = SS.E

    nu_K = Prior.nu + SS.N_K
    P_KEE = Prior.P_EE[np.newaxis,:] + SS.xxT_KEE

    w_KE = np.zeros((K, E))
    for k in range(K):
        w_KE[k] = np.linalg.solve(
            P_KEE[k],
            SS.yx_KE[k] + Prior.Pw_E)

    tau_K = np.zeros(K)
    tau_K[:] = SS.yy_K + Prior.tau + Prior.wPw_1
    for k in xrange(K):
        tau_K[k] -= np.dot(w_KE[k], np.dot(P_KEE[k], w_KE[k]))

    return nu_K, tau_K, w_KE, P_KEE


def calcPostParamsFromSSForComp(
        SS=None, kA=0, kB=None,
        Prior=None,
        **kwargs):
    ''' Calc posterior parameters for specific cluster from SS

    Returns
    --------
    nu_K : float
    tau_K : float
    w_KE : 1D array, size E
    P_KEE : 2D array, size E x E
    '''
    K = 1
    E = SS.E
    
    if kB is None:
        SS_N_K = SS.N_K[kA]
        SS_xxT_KEE = SS.xxT_KEE[kA]
        SS_yx_KE = SS.yx_KE[kA]
        SS_yy_K = SS.yy_K[kA]
    else:
        SS_N_K = SS.N_K[kA] + SS.N_K[kB]
        SS_xxT_KEE = SS.xxT_KEE[kA] + SS.xxT_KEE[kB]
        SS_yx_KE = SS.yx_KE[kA] + SS.yx_KE[kB]
        SS_yy_K = SS.yy_K[kA] + SS.yy_K[kB]

    nu_K = Prior.nu + SS_N_K
    P_KEE = Prior.P_EE[np.newaxis,:] + SS_xxT_KEE

    w_KE = np.zeros((K, E))
    for k in range(K):
        w_KE[k] = np.linalg.solve(
            P_KEE[k],
            SS_yx_KE[k] + Prior.Pw_E)

    tau_K = np.zeros(K)
    tau_K[:] = SS_yy_K + Prior.tau + Prior.wPw_1
    for k in xrange(K):
        tau_K[k] -= np.dot(w_KE[k], np.dot(P_KEE[k], w_KE[k]))

    return nu_K, tau_K, w_KE, P_KEE

def E_log_d(w_E=None, P_EE=None, nu=None, tau=None, **kwargs):
    ''' Expected value of log of precision parameter delta

    Returns
    -------
    Elogdelta : scalar
    '''
    return digamma(0.5 * nu) - np.log(0.5 * tau)

def E_d(w_E=None, P_EE=None, nu=None, tau=None, **kwargs):
    ''' Expected value of precision parameter delta

    Returns
    -------
    Edelta : positive scalar
    '''
    return nu / tau

def E_d_w(w_E=None, P_EE=None, nu=None, tau=None, **kwargs):
    ''' Expected value of weight vector scaled by delta

    $$
    E[ \delta_k w_k ]
    $$

    Returns
    -------
    E_d_w : 1D array, size E
    '''
    return (nu / tau) * w_E

def E_d_w_wT(w_E=None, P_EE=None, nu=None, tau=None, **kwargs):
    ''' Expected value of outer product of weight vector, scaled by delta

    $$
    E[ \delta_k w_k w_k^T ]
    $$

    Returns
    -------
    E_d_wwT : 2D array, size E x E
    '''
    return np.linalg.inv(P_EE) + (nu / tau) * np.outer(w_E, w_E)

def c_Func(nu=1e-9, tau=1e-9, w_E=None, P_EE=None, logdet_P_EE=None):
    ''' Compute cumulant function for Multivariate-Normal-Univariate-Wishart

    Returns
    -------
    c : float
        scalar output of cumulant function
    '''
    if logdet_P_EE is None:
        logdet_P_EE = np.log(np.linalg.det(P_EE))
    E = w_E.size
    c_wish_1dim = (0.5 * nu) * np.log(0.5 * tau) - gammaln(0.5 * nu)
    c_normal_Edim = - 0.5 * E * LOGTWOPI + 0.5 * logdet_P_EE
    return c_wish_1dim + c_normal_Edim