import numpy as np
import scipy.linalg
from scipy.special import gammaln, digamma
from sklearn.preprocessing import normalize

from bnpy.suffstats import ParamBag, SuffStatBag
from bnpy.util import LOGTWO, LOGPI, LOGTWOPI, EPS
from bnpy.util import dotATA, dotATB, dotABT
from bnpy.util import numpyToSharedMemArray, fillSharedMemArray

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

    ####
    def get_name(self):
        return 'MixGauss'

    def get_info_string(self):
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
        else:
            k, c = kc
            eta = self.Post.eta[k,:]
            return digamma(eta[c]) - digamma(np.sum(eta))
    ####

    def __init__(self, inferType='VB',C=1, D=0, min_covar=None,
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


    def calcSummaryStats(self, Data, SS, LP, **kwargs):
        ''' Calculate summary statistics for given dataset and local parameters

        Returns
        --------
        SS : SuffStatBag object, with K components.
        '''
        return calcSummaryStats(Data, SS, LP, **kwargs)

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
        nu = Prior.nu + SS.N 
        kappa = Prior.kappa + SS.N 
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
        Prior = self.Prior
        eta = Prior.eta + SS.N
        return eta

####
# first step of local inference procedure : calculate
# the conditional log likelihood under each superstate
# by summing over substates
def calcLocalParams(Dslice, **kwargs):
    L = calcLogSoftEvMatrix_FromPost(Dslice, **kwargs)
    LP = dict(E_log_soft_ev_full=L)
    LP = dict(E_log_soft_ev=np.sum(L, axis=2))
    return LP

def calcLogSoftEvMatrix_FromPost(Dslice, **kwargs):
    ''' Calculate expected log soft ev matrix for variational.

    Returns
    ------
    L : 2D array, size N x K
    '''
    K = kwargs['K']
    C = kwargs['C']
    L = np.zeros((Dslice.nObs, K, C))
    for k in xrange(K):
        for c in xrange(C):
            # sum varTheta[:,k,c] for c =1:C
            L[:, k, c] = - 0.5 * Dslice.dim * LOGTWOPI \
                + 0.5 * kwargs['E_logdetL'][k, c]  \
                - 0.5 * _mahalDist_Post(Dslice.X, k, c, **kwargs) 
                ### SHOULDN'T SECOND LINE BE - 0.5 * kwargs['E_logdetL'][k, c]  \ (not plus?)
    return L

def _mahalDist_Post(X, k, c, D=None,
                    cholB=None,
                    m=None, nu=None, kappa=None, **kwargs):
    ''' Calc expected mahalonobis distance from comp k to each data atom

    Returns
    --------
    distvec : 1D array, size N
           distvec[n] gives E[ (x-\mu) \Lam (x-\mu) ] for comp k and substate c
    '''
    Q = np.linalg.solve(cholB[k],
                        (X - m[k,c]).T)
    Q *= Q
    return nu[k,c] * np.sum(Q, axis=0) + D / kappa[k,c]

# input  : LP dict computed by alloc model, 
#          containing resp field which holds
#          marginal assignment probabilities 
#          at each time point t,
# output : LP dict containing substate
#          marginal probabilities  
def calcSubstateLocalParams(Data, LP, **kwargs):
    L = calcSubstateMarginalProbabilities(Data, LP, **kwargs)
    LP['substate_resp'] = L
    return LP

def calcSubstateMarginalProbabilities(Data, LP, **kwargs):
    N = Dslice.nObs
    K = kwargs['K']
    C = kwargs['C']
    L = np.zeros((N, K, C))

    resp = LP['resp'] # N x K
    E_log_soft_ev = LP['E_log_soft_ev_full'] # N x K x C
    E_logpsi = kwargs['E_logpsi'] # K x C
    bpPotential = np.tile(E_logpsi[np.newaxis,:,:], (N,1,1)) + E_log_soft_ev
    L = resp[:,:,np.newaxis] * normalize(bpPotential, axis=2, norm='l1')
    return L

####

def calcSummaryStats(Data, SS, LP, **kwargs):
    ''' Calculate summary statistics for given dataset and local parameters

    Returns
    --------
    SS : SuffStatBag object, with K components.
    '''
    X = Data.X # N x D
    D = Data.dim
    substate_resp = LP['substate_resp'] # N x K x C
    K = substate_resp.shape[1]
    C = substate_resp.shape[2]

    if SS is None:
        SS = SuffStatBag(K=K, C=C, D=D)

    # Expected count for each k, c
    #  Usually computed by allocmodel. But just in case...
    if not hasattr(SS, 'N'):
        SS.setField('N', np.sum(substate_resp, axis=0), dims=('K','C'))
    # Expected mean for each k
    SS.setField('x', np.dot(substate_resp.transpose(1,2,0), X), dims=('K','C','D')) # NOT OPTIMIZED USING BLAS ROUTINES
    
    # Expected outer-product for each k, c
    sqrtSResp = np.sqrt(substate_resp) # N x K x C
    xxT = np.zeros((K, C, D, D))
    for k in xrange(K):
        for c in xrange(C):
            xxT[k,c] = dotATA(sqrtSResp[:, k, c][:, np.newaxis] * Data.X)
    SS.setField('xxT', xxT, dims=('K', 'C', 'D', 'D'))
    return SS

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