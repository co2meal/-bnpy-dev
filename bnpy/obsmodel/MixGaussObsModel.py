import numpy as np
import scipy.linalg
from scipy.special import gammaln, digamma

from bnpy.suffstats import ParamBag, SuffStatBag
from bnpy.util import LOGTWO, LOGPI, LOGTWOPI, EPS
from bnpy.util import dotATA, dotATB, dotABT
from bnpy.util import numpyToSharedMemArray, fillSharedMemArray
from bnpy.util import as1D, as2D, as3D
from bnpy.util import NumericUtil

from AbstractObsModel import AbstractObsModel

class MixGaussObsModel(AbstractObsModel):

    ''' Full-covariance gaussian data generation model for real vectors.


    Attributes for Prior (Normal-Wishart)
    --------
    nu : float
        degrees of freedom
    B  : 2D array, size D x D
        scale parameters that set mean of parameter sigma
    m  : 1D array, size D
        mean of the parameter mu
    kappa : float
        scalar precision on parameter mu
    
    Attributes for Prior (Dirichlet)
    --------
    eta : float
        concentration parameter

    Attributes for k-th component, c-th substate of Post (VB parameter)
    ---------
    nu[k,c]    : float
    B[k,c]     : 1D array, size D
    m[k,c]     : 1D array, size D
    kappa[k,c] : float
    eta[k,c]   : float
    '''

    def get_name(self):
        return 'MixGauss'

    def get_info_string(self):
        return 'Mixture of Gaussians with full covariance.'

    def get_info_string_prior(self):
        return 'Mixture of Gaussians with full covariance.'

    def _cholB(self, kc='all'):
        if kc == 'all':
            retArr = np.zeros((self.K, self.C, self.D, self.D))
            for kk in xrange(self.K):
                for cc in xrange(self.C):
                    retArr[kk, cc] = self.GetCached('cholB', (kk, cc))
            return retArr
        else:
            k, c = kc
            B = self.Post.B[k, c]
        return scipy.linalg.cholesky(B, lower=True)

    def _logdetB(self, kc=None):
        if kc is None:
            B = self.Prior.B
            return 2 * np.sum(np.log(np.diag(scipy.linalg.cholesky(B, lower=True))))
        else:
            k, c = kc
            cholB = self.GetCached('cholB', (k, c))
            return 2 * np.sum(np.log(np.diag(cholB)))

    def _E_logdetL(self, kc='all'):
        dvec = np.arange(1, self.D + 1, dtype=np.float)
        if kc == 'all':
            dvec = dvec[:, np.newaxis, np.newaxis]
            retVec = self.D * LOGTWO * np.ones(self.K, self.C)
            for kk in xrange(self.K):
                for cc in  xrange(self.C):
                    retVec[kk, cc] -= self.GetCached('logdetB', (kk, cc))
            nuT = self.Post.nu[np.newaxis, :, :]
            retVec += np.sum(digamma(0.5 * (nuT + 1 - dvec)), axis=0)
            return retVec
        else:
            k, c = kc
            nu = self.Post.nu[k, c]
        return self.D * LOGTWO \
            - self.GetCached('logdetB', (k, c)) \
            + np.sum(digamma(0.5 * (nu + 1 - dvec)))

    def _E_logpsi(self, kc='all'):
        if kc == 'all':
            retArr = np.zeros((self.K, self.C))
            for kk in xrange(self.K):
                for cc in xrange(self.C):
                    retArr[kk, cc] = self.GetCached('E_logpsi', (kk, cc))
            return retArr
        elif kc is None:
            return digamma(self.Prior.eta) - digamma(self.C * self.Prior.eta)
        else:
            k, c = kc
            eta = self.Post.eta[k,:]
            return digamma(eta[c]) - digamma(np.sum(eta))

    def __init__(self, inferType='VB',C=3, D=0, min_covar=None,
                 Data=None,
                 **PriorArgs):
        ''' Initialize bare obsmodel with valid prior hyperparameters.

        Resulting object lacks either EstParams or Post,
        which must be created separately (see init_global_params).
        '''
        if Data is not None:
            self.D = Data.dim
        else:
            self.D = int(D)
        self.C = C
        self.K = 0
        self.inferType = inferType
        self.min_covar = min_covar
        self.createPrior(Data, **PriorArgs)
        self.Cache = dict()

    def createPrior(self, Data, eta=0, nu=0, B=None,
                    m=None, kappa=None,
                    MMat='zero',
                    ECovMat=None, sF=1.0, **kwargs):
        ''' Initialize Prior ParamBag attribute.

        Post Condition
        ------
        Prior expected covariance matrix set to match provided value.
        '''
        D = self.D
        nu = np.maximum(nu, D + 2)
        if B is None:
            if ECovMat is None or isinstance(ECovMat, str):
                ECovMat = createECovMatFromUserInput(D, Data, ECovMat, sF)
            B = ECovMat * (nu - D - 1)
        if B.ndim == 1:
            B = np.asarray([B], dtype=np.float)
        elif B.ndim == 0:
            B = np.asarray([[B]], dtype=np.float)
        if m is None:
            if MMat == 'data':
                m = np.mean(Data.X, axis=0)
            else:
                m = np.zeros(D)
        elif m.ndim < 1:
            m = np.asarray([m], dtype=np.float)
        kappa = np.maximum(kappa, 1e-8)
        eta = np.maximum(eta, 1.)
        self.Prior = ParamBag(K=0, D=D)
        self.Prior.setField('eta', eta, dims=None)
        self.Prior.setField('nu', nu, dims=None)
        self.Prior.setField('kappa', kappa, dims=None)
        self.Prior.setField('m', m, dims=('D'))
        self.Prior.setField('B', B, dims=('D', 'D'))

    def updatePost(self, SS):
        ''' Update attribute Post for all comps given suff stats.

        Update uses the variational objective.

        Post Condition
        ---------
        Attributes K and Post updated in-place.
        '''
        self.ClearCache()
        if not hasattr(self, 'Post') or self.Post.K != SS.K:
            self.Post = ParamBag(K=SS.K, D=SS.D, C=SS.C)

        nu, B, m, kappa = self.calcPostGaussParams(SS)
        self.Post.setField('nu', nu, dims=('K','C'))
        self.Post.setField('kappa', kappa, dims=('K','C'))
        self.Post.setField('m', m, dims=('K', 'C', 'D')) 
        self.Post.setField('B', B, dims=('K', 'C', 'D', 'D')) 
        eta = self.calcPostSubstateParams(SS)
        self.Post.setField('eta', eta, dims=('K', 'C'))
        self.K = SS.K
        self.C = SS.C

    def calcPostGaussParams(self, SS):
        ''' Calc updated params (nu, B, m, kappa) for all state/substate 
            pairs given suff stats

            These params define the common-form of the exponential family
            Normal-Wishart posterior distribution over mu, diag(Lambda)

            Returns
            --------
            nu : 2D array, size K x C
            B : 4D array, size K x C x D x D, each B[k, c] is symmetric and pos. def.
            m : 3D array, size K x C x D
            kappa : 2D array, size K x C
        '''
        Prior = self.Prior 
        nu = Prior.nu + SS.N_full
        kappa = Prior.kappa + SS.N_full
        m = (Prior.kappa * np.reshape(Prior.m, ((1,1,SS.x.shape[2]))) + SS.x) / kappa[:, :, np.newaxis] 
        Bmm = Prior.B + Prior.kappa * np.outer(Prior.m, Prior.m) 
        B = SS.xxT + Bmm[np.newaxis, np.newaxis, :] 
        for k in xrange(B.shape[0]): 
            for c in xrange(B.shape[1]): 
               B[k,c] -= kappa[k,c] * np.outer(m[k,c], m[k,c]) 
        return nu, B, m, kappa 

    def calcPostSubstateParams(self, SS):
        ''' Calc updated params (eta) for all state/substate pairs given suff stats

            Returns 
            --------
            eta : 2D array, size K x C
        '''
        eta = self.Prior.eta + SS.N_full
        return eta

    # first step of local inference procedure : calculate
    # the conditional log likelihood under each superstate
    # by summing over substates
    def calc_local_params(self, Data, LP=None, **kwargs): 
        if LP is None:
            LP = dict()
        L = self.calcLogSoftEvMatrix_FromPost_Full(Data, **kwargs)
        LP['E_log_soft_ev_full'] = L
        LP['E_log_soft_ev'] = np.sum(L, axis=2)
        return LP

    def calcLogSoftEvMatrix_FromPost_Full(self, Data, **kwargs):
        ''' Calculate expected log soft ev matrix for variational.

        Returns
        ------
        L : 3D array, size N x K x C
        '''
        K = self.Post.K
        C = self.Post.C
        L = np.zeros((Data.nObs, K, C))
        for k in xrange(K):
            for c in xrange(C):
                L[:, k, c] = - 0.5 * Data.dim * LOGTWOPI \
                    + 0.5 * self.GetCached('E_logdetL', (k,c)) \
                    - 0.5 * self._mahalDist_Post(Data.X,k=k,c=c) 
        return L + self.GetCached('E_logpsi') # NxKxC + KxC 

    def _mahalDist_Post(self, X, k, c): 
        ''' Calc expected mahalonobis distance from comp k to each data atom

        Returns
        --------
        distvec : 1D array, size N
               distvec[n] gives E[ (x-\mu) \Lam (x-\mu) ] for comp k and substate c
        '''
        Q = np.linalg.solve(self.GetCached('cholB', (k,c)),
                            (X - self.Post.m[k,c]).T)
        Q *= Q
        return self.Post.nu[k,c] * np.sum(Q, axis=0) + self.D / self.Post.kappa[k,c]

    def calcSummaryStats(self, Data, SS, LP, doPrecompEntropy=False, **kwargs):
        ''' Calculate summary statistics for given dataset and local parameters

        Returns
        --------
        SS : SuffStatBag object, with K components.
        '''
        if 'substate_resp' not in LP:
            # should check if post is around and if not, calculate heuristic init (init should be called at most once)
            print '\n no substate resp in LP\n'
            #LP['substate_resp'] = self.calcInitSubstateResp(Data, SS, LP['resp'], **kwargs)
            LP = self.calcSubstateLocalParams(Data, LP, **kwargs)
        else:
            print '\n yes substate resp in LP \n'
        return self.calcSSGivenSubstateResp(Data, SS, LP, doPrecompEntropy=doPrecompEntropy, **kwargs)

    def calcInitSubstateResp(self, Data, SS, resp, **kwargs): 
        N,K,C = Data.nObs,SS.K,self.C
        substate_resp = np.zeros((N,K,C))

        # better alternative -- k-means?
        for c in xrange(C):
            substate_resp[:,:,c] = resp/C

        return substate_resp

    # input  : LP dict computed by alloc model, 
    #          containing resp field which holds
    #          marginal assignment probabilities 
    #          at each time point t,
    # output : LP dict containing substate
    #          marginal probabilities  
    def calcSubstateLocalParams(self, Data, LP, **kwargs):

        L = self.calcSubstateMarginalProbabilities(Data, LP, **kwargs)
        LP['substate_resp'] = L
        return LP

    def calcSubstateMarginalProbabilities(self, Data, LP, **kwargs):

        """
        print LP.keys()

        resp = LP['resp'] # N x K
        E_log_soft_ev = self.calcLogSoftEvMatrix_FromPost_Full(Data, **kwargs) # N x K x C
        Z = np.sum(E_log_soft_ev, axis=2)
        E_log_soft_ev /= Z[:,:,np.newaxis]
        L = resp[:,:,np.newaxis] * E_log_soft_ev
        """
        
        """
        vartheta = self.calcLogSoftEvMatrix_FromPost_Full(Data, **kwargs) # N x K x C
        fwdbwd   = LP['E_log_soft_ev'][:,:,np.newaxis]
        L = vartheta * fwdbwd # elementwise product
        Z = np.sum(np.sum(L,axis=2),axis=1)
        L = L / Z[:,np.newaxis,np.newaxis]
        """
        resp = LP['resp']
        vartheta = LP['E_log_soft_ev_full']
        varthetaMax = np.max(vartheta, axis=2)
        vartheta -= varthetaMax[:, :, np.newaxis]
        np.exp(vartheta, out=vartheta)
        vartheta /= np.sum(vartheta, axis=2)[:,:,np.newaxis]
        return resp[:,:,np.newaxis] * vartheta

    def calcSSGivenSubstateResp(self, Data, SS, LP, doPrecompEntropy=False, **kwargs):
        substate_resp = LP['substate_resp']
        X = Data.X # N x D
        D = Data.dim
        N, K, C = substate_resp.shape
        self.K = K
        if SS is None:
            SS = SuffStatBag(K=K, D=D, C=C) 
        else:
            SS.C = C
            SS._Fields.C = C

        # Expected count for each k, c
        #  Usually computed by allocmodel. But just in case...
        if not hasattr(SS, 'N'):
            SS.setField('N', np.sum(substate_resp, axis=0), dims=('K','C'))
        SS.setField('N_full', np.sum(substate_resp, axis=0), dims=('K','C'))
        # Expected mean for each k
        SS.setField('x', np.dot(substate_resp.transpose(1,2,0), X), dims=('K','C','D')) # NOT OPTIMIZED USING BLAS ROUTINES
        
        # Expected outer-product for each k, c
        sqrtSResp = np.sqrt(substate_resp) # N x K x C
        xxT = np.zeros((K, C, D, D))
        for k in xrange(K):
            for c in xrange(C):
                xxT[k,c] = dotATA(sqrtSResp[:, k, c][:, np.newaxis] * Data.X)
        SS.setField('xxT', xxT, dims=('K', 'C', 'D', 'D'))

        if doPrecompEntropy or True:
            resp = LP['resp']

            eps = 1e-100

            T = substate_resp + eps 
            T /= (resp[:,:,np.newaxis] + eps)
            np.log(T, out=T)
            T *= substate_resp
            
            Hsubstate = np.sum(T, axis=0)
            #np.sum(substate_resp * np.log(substate_resp / resp[:,:,np.newaxis]), axis=0)
            SS.setField('Hsubstate', Hsubstate, dims=('K','C'))

        return SS

    def getDatasetScale(self, SS):
        ''' Get number of observed scalars in dataset from suff stats.

        Used for normalizing the ELBO so it has reasonable range.

        Returns
        ---------
        s : scalar positive integer
        '''
        return SS.N_full.sum() * SS.D

    def _trace__E_L(self, Smat, kc=None):
        if kc is None:
            nu = self.Prior.nu
            B = self.Prior.B
        else:
            k, c = kc
            nu = self.Post.nu[k,c]
            B = self.Post.B[k,c]
        return nu * np.trace(np.linalg.solve(B, Smat))

    def _E_Lmu(self, kc=None):
        if kc is None:
            nu = self.Prior.nu
            B = self.Prior.B
            m = self.Prior.m
        else:
            k, c = kc
            nu = self.Post.nu[k,c]
            B = self.Post.B[k,c]
            m = self.Post.m[k,c]
        return nu * np.linalg.solve(B, m)

    def _E_muLmu(self, kc=None):
        if kc is None:
            nu = self.Prior.nu
            kappa = self.Prior.kappa
            m = self.Prior.m
            B = self.Prior.B
        else:
            k, c = kc
            nu = self.Post.nu[k,c]
            kappa = self.Post.kappa[k,c]
            m = self.Post.m[k,c]
            B = self.Post.B[k,c]
        Q = np.linalg.solve(self.GetCached('cholB', (k,c)), m.T)
        return self.D / kappa + nu * np.inner(Q, Q)

    def _E_log_ppsi_qpsi(self, kc=None):
        eta = self.Post.eta
        K,C = eta.shape
        elogpq = 0 
        for k in xrange(K):
            for c in xrange(C):
                elogpq -= (eta[k,c] - self.Prior.eta)*self.GetCached('E_logpsi',(k,c)) - gammaln(eta[k,c])
            elogpq -= gammaln(np.sum(eta[k]))
        return elogpq + K * gammaln(C*self.Prior.eta) - K*C*gammaln(self.Prior.eta)


    def calcELBO_Memoized(self, SS, afterMStep=False):
        """ Calculate obsModel's objective using suff stats SS and Post.

        Args
        -------
        SS : bnpy SuffStatBag
        afterMStep : boolean flag
            if 1, elbo calculated assuming M-step just completed

        Returns
        -------
        obsELBO : scalar float
            Equal to E[ log p(x) + log p(phi) - log q(phi)]
            want     E[ log p(x | z,y,phi) + log p(phi) - log q(phi) 
                        + log p(y | z,psi) - log q(y | z,psi) 
                        + log p(psi) - log q(psi)]
            need to calc E[ log p(y | z,psi) - log q(y | z,psi) 
                        + log p(psi) - log q(psi)]

        """
        K = SS.K
        C = SS.C
        elbo = np.zeros((K, C))
        Post = self.Post
        Prior = self.Prior
        for k in xrange(K):
            for c in xrange(C):
                elbo[k, c] = c_Diff(Prior.nu,
                                 self.GetCached('logdetB'),
                                 Prior.m, Prior.kappa,
                                 Post.nu[k,c],
                                 self.GetCached('logdetB', (k,c)),
                                Post.m[k,c], Post.kappa[k,c],
                                ) \
                             + SS.N_full[k,c] * self.GetCached('E_logpsi', (k,c)) \
                            # E[p(y | z, psi)]
            if not afterMStep:
                aDiff = SS.N_full[k,c] + Prior.nu - Post.nu[k,c]
                bDiff = SS.xxT[k,c] + Prior.B \
                                  + Prior.kappa * np.outer(Prior.m, Prior.m) \
                    - Post.B[k,c] \
                    - Post.kappa[k,c] * np.outer(Post.m[k,c], Post.m[k,c])
                cDiff = SS.x[k,c] + Prior.kappa * Prior.m \
                    - Post.kappa[k,c] * Post.m[k,c]
                dDiff = SS.N_full[k,c] + Prior.kappa - Post.kappa[k,c]
                elbo[k,c] += 0.5 * aDiff * self.GetCached('E_logdetL', (k,c)) \
                    - 0.5 * self._trace__E_L(bDiff, (k,c)) \
                    + np.inner(cDiff, self.GetCached('E_Lmu', (k,c))) \
                    - 0.5 * dDiff * self.GetCached('E_muLmu', (k,c))
        H = SS.Hsubstate 
        return elbo.sum() - 0.5 * np.sum(SS.N_full) * SS.D * LOGTWOPI \
               + self.GetCached('E_log_ppsi_qpsi') +  np.sum(H[np.nonzero(H<np.float('inf'))])

    def setPostFactors(self, obsModel=None, SS=None, LP=None, Data=None,
                       nu=0, B=0, m=0, kappa=0,
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
            m = as3D(m)
            if m.shape[2] != self.D:
                m = m.T.copy()
            K,C, _ = m.shape
            self.Post = ParamBag(K=K, C=C, D=self.D)
            self.Post.setField('nu', nu, dims=('K','C')) 
            self.Post.setField('B', B, dims=('K','C','D', 'D'))
            self.Post.setField('m', m, dims=('K','C', 'D'))
            self.Post.setField('kappa', kappa, dims=('K','C')) 
        self.K = self.Post.K

    def get_mean_for_comp(self, k=None, c=None):
        if k is None or k == 'prior':
            return self.Prior.m
        else:
            return self.Post.m[k,c]

    def get_covar_mat_for_comp(self, k=None, c=None):
        if k is None or k == 'prior':
            return self.Prior.B / self.Prior.nu
        else:
            return self.Post.B[k,c] / self.Post.nu[k,c]





def createECovMatFromUserInput(D=0, Data=None, ECovMat='eye', sF=1.0):
    ''' Create expected covariance matrix defining Wishart prior.

        The creation process follows user-specified criteria.

    Args
    --------
    D : positive integer, size of each observation
    Data : [optional] dataset to use to make Sigma in data-driven way
    ECovMat : string name of the procedure to use to create Sigma
        'eye' : make Sigma a multiple of the identity matrix
        'covdata' : set Sigma to a multiple of the data covariance matrix
        'fromtruelabels' : set Sigma to the empirical mean of the
            covariances for each true cluster in the dataset

    Returns
    --------
    Sigma : 2D array, size D x D
        Symmetric and positive definite.
    '''
    if Data is not None:
        assert D == Data.dim
    if ECovMat == 'eye':
        Sigma = sF * np.eye(D)
    elif ECovMat == 'covdata':
        Sigma = sF * np.cov(Data.X.T, bias=1)
    elif ECovMat == 'diagcovdata':
        Sigma = sF * np.diag(np.diag(np.cov(Data.X.T, bias=1)))
    elif ECovMat == 'covfirstdiff':
        if not hasattr(Data, 'Xprev'):
            raise ValueError(
                'covfirstdiff only applies to auto-regressive datasets')
        Xdiff = Data.X - Data.Xprev
        Sigma = sF * np.cov(Xdiff.T, bias=1)
    elif ECovMat == 'diagcovfirstdiff':
        if not hasattr(Data, 'Xprev'):
            raise ValueError(
                'covfirstdiff only applies to auto-regressive datasets')
        Xdiff = Data.X - Data.Xprev
        Sigma = sF * np.diag(np.diag(np.cov(Xdiff.T, bias=1)))

    elif ECovMat == 'fromtruelabels':
        ''' Set Cov Matrix Sigma using the true labels in empirical Bayes style
            Sigma = \sum_{c : class labels} w_c * SampleCov[ data from class c]
        '''
        if hasattr(Data, 'TrueLabels'):
            Z = Data.TrueLabels
        else:
            Z = Data.TrueParams['Z']
        Zvals = np.unique(Z)
        Kmax = len(Zvals)
        wHat = np.zeros(Kmax)
        SampleCov = np.zeros((Kmax, D, D))
        for kLoc, kVal in enumerate(Zvals):
            mask = Z == kVal
            wHat[kLoc] = np.sum(mask)
            SampleCov[kLoc] = np.cov(Data.X[mask].T, bias=1)
        wHat = wHat / np.sum(wHat)
        Sigma = 1e-8 * np.eye(D)
        for k in range(Kmax):
            Sigma += wHat[k] * SampleCov[k]
    else:
        raise ValueError('Unrecognized ECovMat procedure %s' % (ECovMat))
    return Sigma

def c_Diff(nu1, logdetB1, m1, kappa1,
           nu2, logdetB2, m2, kappa2):
    ''' Evaluate difference of cumulant functions c(params1) - c(params2)

    May be more numerically stable than directly using c_Func
    to find the difference.

    Returns
    -------
    diff : scalar real value of the difference in cumulant functions
    '''
    if logdetB1.ndim >= 2:
        logdetB1 = np.log(np.linalg.det(logdetB1))
    if logdetB2.ndim >= 2:
        logdetB2 = np.log(np.linalg.det(logdetB2))
    D = m1.size
    dvec = np.arange(1, D + 1, dtype=np.float)
    return - 0.5 * D * LOGTWO * (nu1 - nu2) \
           - np.sum(gammaln(0.5 * (nu1 + 1 - dvec))) \
        + np.sum(gammaln(0.5 * (nu2 + 1 - dvec))) \
        + 0.5 * D * (np.log(kappa1) - np.log(kappa2)) \
        + 0.5 * (nu1 * logdetB1 - nu2 * logdetB2)