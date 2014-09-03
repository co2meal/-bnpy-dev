'''
AutoRegGaussObsModel

Prior : Matrix-Normal-Wishart
* nu
* B
* M
* V

EstParams 
* A
* Sigma

Posterior : Normal-Wishart
* nu[k]
* B[k]
* M[k]
* V[k]

'''
import numpy as np
import scipy.linalg
from scipy.special import gammaln, digamma

from bnpy.suffstats import ParamBag, SuffStatBag
from bnpy.util import LOGTWO, LOGPI, LOGTWOPI, EPS
from bnpy.util import dotATA, dotATB, dotABT
from bnpy.util import as1D, as2D, as3D

from AbstractObsModel import AbstractObsModel 
from GaussObsModel import createECovMatFromUserInput

class AutoRegGaussObsModel(AbstractObsModel):

  def __init__(self, inferType='EM', D=0, min_covar=None, 
                     Data=None, 
                     **PriorArgs):
    ''' Initialize bare Gaussian obsmodel with MatrixNormal-Wishart prior. 
        Resulting object lacks either EstParams or Post, 
          which must be created separately.
    '''
    if Data is not None:
      self.D = Data.dim
    else:
      self.D = int(D)
    self.K = 0
    self.inferType = inferType
    self.min_covar = min_covar
    self.createPrior(Data, **PriorArgs)
    self.Cache = dict()

  def createPrior(self, Data, nu=0, B=None,
                              M=None, V=None,
                              ECovMat=None, sF=1.0, sV=1.0, **kwargs):
    ''' Initialize Prior ParamBag object, with fields nu, B, m, kappa
          set according to match desired mean and expected covariance matrix.
    '''
    D = self.D
    nu = np.maximum(nu, D+2)
    if B is None:
      if ECovMat is None or type(ECovMat) == str:
        ECovMat = createECovMatFromUserInput(D, Data, ECovMat, sF)    
      B = ECovMat * (nu - D - 1)
    B = as2D(B)

    if M is None:
      M = np.zeros((D,D))
    else:
      M = as2D(M)

    if V is None:
      V = sV * np.eye(D)
    else:
      V = as2D(V)
    self.Prior = ParamBag(K=0, D=D)
    self.Prior.setField('nu', nu, dims=None)
    self.Prior.setField('B', B, dims=('D','D'))
    self.Prior.setField('V', V, dims=('D', 'D'))
    self.Prior.setField('M', M, dims=('D', 'D'))


  def get_mean_for_comp(self, k=None):
    if hasattr(self, 'EstParams'):
      return np.diag(self.EstParams.A[k])
    elif k is None or k == 'prior':
      return np.diag(self.Prior.M)
    else:
      return np.diag(self.Post.M[k])

  def get_covar_mat_for_comp(self, k=None):
    if hasattr(self, 'EstParams'):
      return self.EstParams.Sigma[k]
    elif k is None or k == 'prior':
      return self._E_CovMat()
    else:
      return self._E_CovMat(k)
    

  ######################################################### I/O Utils
  #########################################################   for humans
  def get_name(self):
    return 'AutoRegGauss'

  def get_info_string(self):
    return 'Auto-Regressive Gaussian with full covariance.'
  
  def get_info_string_prior(self):
    msg = 'MatrixNormal-Wishart on each mean/prec matrix pair: A, Lam\n'
    if self.D > 2:
      sfx = ' ...'
    else:
      sfx = ''
    M = self.Prior.M[:2, :2]
    S = self._E_CovMat()[:2,:2]
    msg += 'E[ A ] = \n' 
    msg += str(M) + sfx + '\n'
    msg += 'E[ Sigma ] = \n'
    msg += str(S) + sfx
    msg = msg.replace('\n', '\n  ')
    return msg

  ######################################################### Set EstParams
  #########################################################
  def setEstParams(self, obsModel=None, SS=None, LP=None, Data=None,
                          A=None, Sigma=None,
                          **kwargs):
    ''' Create EstParams ParamBag with fields A, Sigma
    '''
    self.ClearCache()
    if obsModel is not None:
      self.EstParams = obsModel.EstParams.copy()
      return
    
    if LP is not None and Data is not None:
      SS = self.calcSummaryStats(Data, None, LP)

    if SS is not None:
      self.updateEstParams(SS)
    else:
      A = as3D(A)
      Sigma = as3D(Sigma)
      self.EstParams = ParamBag(K=A.shape[0], D=A.shape[1])
      self.EstParams.setField('A', A, dims=('K', 'D', 'D'))
      self.EstParams.setField('Sigma', Sigma, dims=('K', 'D', 'D'))

  def setEstParamsFromPost(self, Post):
    ''' Convert from Post (nu, B, m, kappa) to EstParams (mu, Sigma),
         each EstParam is set to its posterior mean.
    '''
    D = Post.D
    self.EstParams = ParamBag(K=Post.K, D=D)    
    A = Post.M.copy()
    Sigma = Post.B / (Post.nu - D - 1)[:, np.newaxis, np.newaxis]
    self.EstParams.setField('A', A, dims=('K','D','D'))
    self.EstParams.setField('Sigma', Sigma, dims=('K','D','D'))
    self.K = self.EstParams.K

  
  ######################################################### Set Post
  #########################################################
  def setPostFactors(self, obsModel=None, SS=None, LP=None, Data=None,
                            nu=0, B=0, M=0, V=0,
                            **kwargs):
    ''' Create Post ParamBag with fields nu, B, M, V
    '''
    self.ClearCache()
    if obsModel is not None:
      if hasattr(obsModel, 'Post'):
        self.Post = obsModel.Post.copy()
      else:
        self.setPostFromEstParams(obsModel.EstParams)
      return
    
    if LP is not None and Data is not None:
      SS = self.calcSummaryStats(Data, None, LP)

    if SS is not None:
      self.updatePost(SS)
    else:
      M = as3D(M)
      B = as3D(B)
      V = as3D(V)

      K, D, D2 = M.shape
      assert D == self.D
      assert D == D2
      self.Post = ParamBag(K=K, D=self.D)
      self.Post.setField('nu', as1D(nu), dims=('K'))
      self.Post.setField('B', B, dims=('K','D','D'))
      self.Post.setField('M', M, dims=('K','D','D'))
      self.Post.setField('V', V, dims=('K','D','D'))
    self.K = self.Post.K

  def setPostFromEstParams(self, EstParams, Data=None, N=None):
    ''' Convert from EstParams (A, Sigma) to Post (nu, B, M, V),
        
        Each posterior hyperparam is set so EstParam is the posterior mean
    '''
    K = EstParams.K
    D = EstParams.D
    if Data is not None:
      N = Data.nObsTotal
    N = np.asarray(N, dtype=np.float)
    if N.ndim == 0:
      N = N/K * np.ones(K)

    nu = self.Prior.nu + N
    B = EstParams.Sigma * (nu - D - 1)[:,np.newaxis, np.newaxis]
    M = EstParams.A.copy()
    V = as3D(self.Prior.V)

    self.Post = ParamBag(K=K, D=D)
    self.Post.setField('nu', nu, dims=('K'))
    self.Post.setField('B', B, dims=('K','D','D'))
    self.Post.setField('M', M, dims=('K','D','D'))
    self.Post.setField('V', V, dims=('K','D','D'))
    self.K = self.Post.K

  ########################################################### Suff Stats
  ########################################################### 
  def calcSummaryStats(self, Data, SS, LP):
    X = Data.X
    Xprev = Data.Xprev
    resp = LP['resp']
    K = resp.shape[1]

    if SS is None:
      SS = SuffStatBag(K=K, D=Data.dim)
    
    # Expected count for each k
    #  Usually computed by allocmodel. But just in case...
    if not hasattr(SS, 'N'):      
      SS.setField('N', np.sum(resp, axis=0), dims='K')

    ## Expected outer products
    sqrtResp = np.sqrt(resp)
    xxT = np.empty((K, self.D, self.D))
    ppT = np.empty((K, self.D, self.D))
    pxT = np.empty((K, self.D, self.D))
    for k in xrange(K):
      sqrtResp_k = sqrtResp[:,k][:,np.newaxis]
      xxT[k] = dotATA(sqrtResp_k * Data.X )
      ppT[k] = dotATA(sqrtResp_k * Data.Xprev )
      pxT[k] = np.dot(Data.Xprev.T, resp[:,k][:,np.newaxis] * Data.X)
    SS.setField('xxT', xxT, dims=('K','D','D'))
    SS.setField('ppT', ppT, dims=('K','D','D'))
    SS.setField('pxT', pxT, dims=('K','D','D'))
    return SS 

  def incrementSS(self, SS, k, x):
    pass

  def decrementSS(self, SS, k, x):
    pass

  ########################################################### EM E step
  ########################################################### 
  def calcLogSoftEvMatrix_FromEstParams(self, Data):
    ''' Calculate log soft evidence matrix for given Dataset under EstParams

        Returns
        ---------
        L : 2D array, size N x K
            L[n,k] = log p( data n | EstParams for comp k )
    '''
    K = self.EstParams.K
    L = np.empty((Data.nObs, K))
    for k in xrange(K):
      L[:,k] = - 0.5 * self.D * LOGTWOPI \
               - 0.5 * self._logdetSigma(k)  \
               - 0.5 * self._mahalDist_EstParam(Data.X, Data.Xprev, k)
    return L

  def _mahalDist_EstParam(self, X, Xprev, k):
    ''' Calc Mahalanobis distance from EstParams of comp k to every row of X

        Args
        ---------
        X : 2D array, size N x D
        k : integer ID of comp

        Returns
        ----------
        dist : 1D array, size N
    '''
    deltaX = X - np.dot(Xprev, self.EstParams.A[k].T)
    Q = np.linalg.solve(self.GetCached('cholSigma', k), \
                        deltaX.T)
    Q *= Q
    return np.sum(Q, axis=0)

  def _cholSigma(self, k):
    ''' Calculate lower cholesky decomposition of EstParams.Sigma for comp k

        Returns
        --------
        L : 2D array, size D x D, lower triangular
            Sigma = np.dot(L, L.T)
    '''
    return scipy.linalg.cholesky(self.EstParams.Sigma[k], lower=1)    

  def _logdetSigma(self, k):
    ''' Calculate log determinant of EstParam.Sigma for comp k

        Returns
        ---------
        logdet : scalar real
    '''
    return 2 * np.sum(np.log(np.diag(self.GetCached('cholSigma', k))))

  ######################################################### EM M step
  #########################################################
  def updateEstParams_MaxLik(self, SS):
    ''' Update EstParams for all comps via maximum likelihood given suff stats

        Returns
        ---------
        None. Fields K and EstParams updated in-place.
    '''
    self.ClearCache()
    if not hasattr(self, 'EstParams') or self.EstParams.K != SS.K:
      self.EstParams = ParamBag(K=SS.K, D=SS.D)

    minCovMat = self.min_covar * np.eye(SS.D)
    A = np.zeros((SS.K, self.D, self.D))
    Sigma = np.zeros((SS.K, self.D, self.D))
    for k in xrange(SS.K):
      if SS.N[k] < 2:
        A[k] = np.linalg.solve(SS.ppT[k] + self.min_covar*np.eye(self.D),
                               SS.pxT[k]).T
      else:
        A[k] = np.linalg.solve(SS.ppT[k], SS.pxT[k]).T
      Sigma[k] = SS.xxT[k] \
                  - 2*np.dot(SS.pxT[k].T, A[k].T) \
                  + np.dot(A[k], np.dot(SS.ppT[k], A[k].T))
      Sigma[k] /= SS.N[k]  
      #Sigma[k] = 0.5 * (Sigma[k] + Sigma[k].T) # symmetry!
      Sigma[k] += minCovMat

    self.EstParams.setField('A', A, dims=('K','D','D'))
    self.EstParams.setField('Sigma', Sigma, dims=('K','D','D'))
    self.K = SS.K

  def updateEstParams_MAP(self, SS):
    ''' Update EstParams for all comps via MAP estimation given suff stats

        Returns
        ---------
        None. Fields K and EstParams updated in-place.
    '''
    self.ClearCache()
    if not hasattr(self, 'EstParams') or self.EstParams.K != SS.K:
      self.EstParams = ParamBag(K=SS.K, D=SS.D)
    raise NotImplemented('TODO')

  ########################################################### Post updates
  ########################################################### 
  def updatePost(self, SS):
    ''' Update (in place) posterior params for all comps given suff stats

        Afterwards, self.Post contains Normal-Wishart posterior params
        given self.Prior and provided suff stats SS

        Returns
        ---------
        None. Fields K and Post updated in-place.
    '''
    self.ClearCache()
    if not hasattr(self, 'Post') or self.Post.K != SS.K:
      self.Post = ParamBag(K=SS.K, D=SS.D)

    nu, B, M, V = self.calcPostParams(SS)
    self.Post.setField('nu', nu, dims=('K'))
    self.Post.setField('B', B, dims=('K','D','D'))
    self.Post.setField('M', M, dims=('K','D','D'))
    self.Post.setField('V', V, dims=('K','D','D'))
    self.K = SS.K

  def calcPostParams(self, SS):
    ''' Calc updated params (nu, B, m, kappa) for all comps given suff stats

        These params define the common-form of the exponential family 
        Normal-Wishart posterior distribution over mu, diag(Lambda)

        Returns
        --------
        nu : 1D array, size K
        B : 3D array, size K x D x D, each B[k] is symmetric and pos. def.
        m : 2D array, size K x D
        kappa : 1D array, size K
    '''
    Prior = self.Prior
    nu = Prior.nu + SS.N

    B_MVM = Prior.B + np.dot(Prior.M, np.dot(Prior.V, Prior.M.T))
    B = SS.xxT + B_MVM[np.newaxis,:]
    V = SS.ppT + Prior.V[np.newaxis,:]
    M = np.zeros((SS.K, self.D, self.D))
    for k in xrange(B.shape[0]):
      M[k] = np.linalg.solve(V[k], SS.pxT[k] + np.dot(Prior.V, Prior.M.T)).T
      B[k] -= np.dot(M[k], np.dot(V[k], M[k].T))
    return nu, B, M, V

  def calcPostParamsForComp(self, SS, kA=None, kB=None):
    ''' Calc params (nu, B, m, kappa) for specific comp, given suff stats

        These params define the common-form of the exponential family 
        Normal-Wishart posterior distribution over mu[k], diag(Lambda)[k]

        Returns
        --------
        nu : positive scalar
        B : 2D array, size D x D, symmetric and positive definite
        m : 1D array, size D
        kappa : positive scalar
    '''
    raise NotImplementedError('ToDo')

  ########################################################### Stochastic Post
  ########################################################### update
  def updatePost_stochastic(self, SS, rho):
    ''' Stochastic update (in place) posterior for all comps given suff stats
    '''
    raise NotImplementedError('ToDo')


  def calcNaturalPostParams(self, SS):
    ''' Calc updated params (nu, b, km, kappa) for all comps given suff stats

        These params define the natural-form of the exponential family 
        Normal-Wishart posterior distribution over mu, Lambda

        Returns
        --------
        nu : 1D array, size K
        Bnat : 3D array, size K x D x D
        km : 2D array, size K x D
        kappa : 1D array, size K
    '''
    raise NotImplementedError('ToDo')

    
  def convertPostToNatural(self):
    ''' Convert (in-place) current posterior params from common to natural form
    '''
    raise NotImplementedError('ToDo')


  def convertPostToCommon(self):
    ''' Convert (in-place) current posterior params from natural to common form
    '''
    raise NotImplementedError('ToDo')


  ########################################################### VB E/Local step
  ########################################################### 
  def calcLogSoftEvMatrix_FromPost(self, Data):
    ''' Calculate expected log soft ev matrix for given dataset under posterior

        Returns
        ------
        L : 2D array, size N x K
    '''
    K = self.Post.K
    L = np.zeros((Data.nObs, K))
    for k in xrange(K):
      L[:,k] = - 0.5 * self.D * LOGTWOPI \
               + 0.5 * self.GetCached('E_logdetL', k)  \
               - 0.5 * self._mahalDist_Post(Data.X, Data.Xprev, k)
    return L

  def _mahalDist_Post(self, X, Xprev, k):
    ''' Calc expected mahalonobis distance from comp k to each data atom 

        Returns
        --------
        distvec : 1D array, size nObs
               distvec[n] gives E[ (x-\mu) \Lam (x-\mu) ] for comp k
    '''
    ## Calc: (x-M*xprev)' * B * (x-M*xprev)
    deltaX = X - np.dot(Xprev, self.Post.M[k].T)
    Q = np.linalg.solve(self.GetCached('cholB', k),
                        deltaX.T)
    Q *= Q

    ## Calc: xprev' * V * xprev
    Qprev = np.linalg.solve(self.GetCached('cholV', k),
                            Xprev.T)
    Qprev *= Qprev

    return self.Post.nu[k] * np.sum(Q, axis=0) \
           + self.D * np.sum(Qprev, axis=0)

  ########################################################### VB ELBO step
  ########################################################### 
  def calcELBO_Memoized(self, SS, afterMStep=False):
    ''' Calculate obsModel's ELBO using sufficient statistics SS and Post.

        Args
        -------
        SS : bnpy SuffStatBag, contains fields for N, x, xxT
        afterMStep : boolean flag
                 if 1, elbo calculated assuming M-step just completed

        Returns
        -------
        obsELBO : scalar float, = E[ log p(x) + log p(phi) - log q(phi)]
    '''
    elbo = np.zeros(SS.K)
    Post = self.Post
    Prior = self.Prior
    for k in xrange(SS.K):
      elbo[k] = c_Diff(Prior.nu,
                        self.GetCached('logdetB'),
                        Prior.M,
                        self.GetCached('logdetV'),
                        Post.nu[k],
                        self.GetCached('logdetB', k),
                        Post.M[k],
                        self.GetCached('logdetV', k),
                        )
      if not afterMStep:
        aDiff = SS.N[k] + Prior.nu - Post.nu[k]
        bDiff = SS.xxT[k] + Prior.B \
                          + np.dot(Prior.M, np.dot(Prior.V, Prior.M.T)) \
                          - Post.B[k] \
                          - np.dot(Post.M[k], np.dot(Post.V[k], Post.M[k].T))
        cDiff = SS.pxT[k] + np.dot(Prior.V, Prior.M.T) \
                          - np.dot(Post.V[k], Post.M[k].T)
        dDiff = SS.ppT[k] + Prior.V - Post.V[k]
        elbo[k] += 0.5 * aDiff * self.GetCached('E_logdetL', k) \
                 - 0.5 * self._trace__E_L(bDiff, k) \
                 + self._trace__E_LA(cDiff, k) \
                 - 0.5 * self._trace__E_ALA(dDiff, k)
    return elbo.sum() - 0.5 * np.sum(SS.N) * SS.D * LOGTWOPI

  def getDatasetScale(self, SS):
    ''' Get scale factor for dataset, indicating number of observed scalars. 

        Used for normalizing the ELBO so it has reasonable range.

        Returns
        ---------
        s : scalar positive integer
    '''
    return SS.N.sum() * SS.D

  ######################################################### Hard Merge
  #########################################################
  def calcHardMergeGap(self, SS, kA, kB):
    ''' Calculate change in ELBO after a hard merge applied to this model

        Returns
        ---------
        gap : scalar real, indicates change in ELBO after merge of kA, kB
    '''
    Post = self.Post
    Prior = self.Prior
    cA = c_Func(Post.nu[kA], Post.B[kA], Post.m[kA], Post.kappa[kA])
    cB = c_Func(Post.nu[kB], Post.B[kB], Post.m[kB], Post.kappa[kB])
    cPrior = c_Func(Prior.nu,   Prior.B,      Prior.m,      Prior.kappa)

    nu, B, m, kappa = self.calcPostParamsForComp(SS, kA, kB)
    cAB = c_Func(nu, B, m, kappa)
    return cA + cB - cPrior - cAB


  def calcHardMergeGap_AllPairs(self, SS):
    ''' Calculate change in ELBO for all possible candidate hard merge pairs 

        Returns
        ---------
        Gap : 2D array, size K x K, upper-triangular entries non-zero
              Gap[j,k] : scalar change in ELBO after merge of k into j
    '''
    Post = self.Post
    Prior = self.Prior
    cPrior = c_Func(Prior.nu, Prior.B, Prior.m, Prior.kappa)
    c = np.zeros(SS.K)
    for k in xrange(SS.K):
      c[k] = c_Func(Post.nu[k], Post.B[k], Post.m[k], Post.kappa[k])

    Gap = np.zeros((SS.K, SS.K))
    for j in xrange(SS.K):
      for k in xrange(j+1, SS.K):
        nu, B, m, kappa = self.calcPostParamsForComp(SS, j, k)
        cjk = c_Func(nu, B, m, kappa)
        Gap[j,k] = c[j] + c[k] - cPrior - cjk
    return Gap

  def calcHardMergeGap_SpecificPairs(self, SS, PairList):
    ''' Calc change in ELBO for specific list of candidate hard merge pairs

        Returns
        ---------
        Gaps : 1D array, size L
              Gap[j] : scalar change in ELBO after merge of pair in PairList[j]
    '''
    Gaps = np.zeros(len(PairList))
    for ii, (kA, kB) in enumerate(PairList):
        Gaps[ii] = self.calcHardMergeGap(SS, kA, kB)
    return Gaps

  ######################################################### Soft Merge
  #########################################################
  def calcSoftMergeGap(self, SS, kdel, alph):
    ''' Calculate net improvement in ELBO after multi-way merge.

        Comp kdel is deleted, and its suff stats are redistributed among 
        other remaining components, according to parameter vector alph.
    '''
    raise NotImplementedError('TODO')

  def calcSoftMergeGap_alph(self, SS, kdel, alph):
    ''' Calculate net improvement in ELBO after multi-way merge as fcn of alph.
        
        This keeps only terms that depend on redistribution vector alph
    '''
    raise NotImplementedError('TODO')



  ########################################################### Marg Lik
  ###########################################################
  def calcLogMargLikForComp(self, SS, kA, kB=None, **kwargs):
    ''' Calc log marginal likelihood of data assigned to given component
          (up to an additive constant that depends on the prior)
        Requires Data pre-summarized into sufficient stats for each comp.
        If multiple comp IDs are provided, we combine into a "merged" component.
        
        Args
        -------
        SS : bnpy suff stats object
        kA : integer ID of target component to compute likelihood for
        kB : (optional) integer ID of second component.
             If provided, we merge kA, kB into one component for calculation.
        Returns
        -------
        logM : scalar real
               logM = log p( data assigned to comp kA ) 
                      computed up to an additive constant
    '''
    nu, B, M, V = self.calcPostParamsForComp(SS, kA, kB)
    return -1 * c_Func(nu, B, M, V)

  def calcMargLik(self, SS):
    ''' Calc log marginal likelihood across all comps, given suff stats

        Returns
        --------
        logM : scalar real
               logM = \sum_{k=1}^K log p( data assigned to comp k | Prior)
    '''
    return self.calcMargLik_CFuncForLoop(SS)

  def calcMargLik_CFuncForLoop(self, SS):
    Prior = self.Prior
    logp = np.zeros(SS.K)
    for k in xrange(SS.K):
      nu, B, m, kappa = self.calcPostParamsForComp(SS, k)
      logp[k] = c_Diff(Prior.nu, Prior.B, Prior.m, Prior.kappa,
                       nu, B, m, kappa)
    return np.sum(logp) - 0.5 * np.sum(SS.N) * LOGTWOPI

  ########################################################### Gibbs Pred Prob
  ########################################################### 
  def calcPredProbVec_Unnorm(self, SS, x):
    ''' Calculate predictive probability that each comp assigns to vector x

        Returns
        --------
        p : 1D array, size K, all entries positive
            p[k] \propto p( x | SS for comp k)
    '''
    return self._calcPredProbVec_Fast(SS, x)

  def _calcPredProbVec_cFunc(self, SS, x):
    raise NotImplementedError('TODO')


  def _calcPredProbVec_Fast(self, SS, x):
    raise NotImplementedError('TODO')

  
  def _Verify_calcPredProbVec(self, SS, x):
    raise NotImplementedError('TODO')

  
  ########################################################### VB Expectations
  ########################################################### 
  def _E_CovMat(self, k=None):
    if k is None:
      B = self.Prior.B
      nu = self.Prior.nu
    else:
      B = self.Post.B[k]
      nu = self.Post.nu[k]
    return B / (nu - self.D - 1)

  def _cholB(self, k=None):
    if k is None:
      B = self.Prior.B
    else:
      B = self.Post.B[k]
    return scipy.linalg.cholesky(B, lower=True)

  def _logdetB(self, k=None):
    cholB = self.GetCached('cholB', k)
    return  2 * np.sum(np.log(np.diag(cholB)))

  def _cholV(self, k=None):
    if k is None:
      V = self.Prior.V
    else:
      V = self.Post.V[k]
    return scipy.linalg.cholesky(V, lower=True)

  def _logdetV(self, k=None):
    cholV = self.GetCached('cholV', k)
    return  2 * np.sum(np.log(np.diag(cholV)))
    
  def _E_logdetL(self, k=None):
    dvec = np.arange(1, self.D+1, dtype=np.float)
    if k is None:
      nu = self.Prior.nu
    else:
      nu = self.Post.nu[k]
    return self.D * LOGTWO \
           - self.GetCached('logdetB', k) \
           + np.sum(digamma(0.5 * (nu + 1 - dvec)))

  def _E_LA(self, k=None):
    if k is None:
      nu = self.Prior.nu
      B = self.Prior.B
      M = self.Prior.M
    else:
      nu = self.Post.nu[k]
      B = self.Post.B[k]
      M = self.Post.M[k]
    return nu * np.linalg.solve(B, M)

  def _E_ALA(self, k=None):
    if k is None:
      nu = self.Prior.nu
      M = self.Prior.M
      B = self.Prior.B
      V = self.Prior.V
    else:
      nu = self.Post.nu[k]
      M = self.Post.M[k]
      B = self.Post.B[k]
      V = self.Post.V[k]
    Q = np.linalg.solve(self.GetCached('cholB', k), M)
    return self.D * np.linalg.inv(V) \
           + nu * np.dot(Q.T, Q)

  def _trace__E_L(self, S, k=None):
    if k is None:
      nu = self.Prior.nu
      B = self.Prior.B
    else:
      nu = self.Post.nu[k]
      B = self.Post.B[k]
    return nu * np.trace(np.linalg.solve(B, S))

  def _trace__E_LA(self, S, k=None):
    E_LA = self._E_LA(k)
    return np.trace(np.dot(E_LA, S))

  def _trace__E_ALA(self, S, k=None):
    E_ALA = self._E_ALA(k)
    return np.trace(np.dot(E_ALA, S))


  # ......................................................########
  # ......................................................########
  ################################################################ end class

def MAPEstParams_inplace(nu, B, m, kappa=0):
  ''' MAP estimate parameters mu, Sigma given Normal-Wishart hyperparameters
  '''
  raise NotImplementedError('TODO')


def c_Func(nu, logdetB, M, logdetV):
  ''' Evaluate cumulant function c, aka log partition function, at given params

      c is the cumulant of the MatrixNormal-Wishart, using common params.

      Returns
      --------
      c : scalar real value of cumulant function at provided args
  '''
  if logdetB.ndim >= 2:
    logdetB = np.log(np.linalg.det(logdetB))
  if logdetV.ndim >= 2:
    logdetV = np.log(np.linalg.det(logdetV))
  D = M.shape[-1]
  dvec = np.arange(1, D+1, dtype=np.float)
  return - 0.25 * D * (D-1) * LOGPI \
         - 0.5 * D * LOGTWO * nu \
         - np.sum( gammaln( 0.5 * (nu + 1 - dvec) )) \
         + 0.5 * nu * logdetB \
         - 0.5 * D * D * LOGTWOPI \
         + 0.5 * D * logdetV


def c_Diff(nu, logdetB, M, logdetV,
           nu2, logdetB2, M2, logdetV2):
  return c_Func(nu, logdetB, M, logdetV) \
         - c_Func(nu2, logdetB2, M2, logdetV2)

