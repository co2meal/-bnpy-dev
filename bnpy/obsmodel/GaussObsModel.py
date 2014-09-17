'''
GaussObsModel

Prior : Normal-Wishart
* nu
* B
* m
* kappa

EstParams 
* mu
* Sigma

Posterior : Normal-Wishart
* nu[k]
* B[k]
* m[k]
* kappa[k]

'''
import numpy as np
import scipy.linalg
from scipy.special import gammaln, digamma

from bnpy.suffstats import ParamBag, SuffStatBag
from bnpy.util import LOGTWO, LOGPI, LOGTWOPI, EPS
from bnpy.util import dotATA, dotATB, dotABT
from bnpy.util import as1D, as2D, as3D

from AbstractObsModel import AbstractObsModel 

class GaussObsModel(AbstractObsModel):

  def __init__(self, inferType='EM', D=0, min_covar=None, 
                     Data=None, 
                     **PriorArgs):
    ''' Initialize bare Gaussian obsmodel with Normal-Wishart prior. 
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
                              m=None, kappa=None,
                              ECovMat=None, sF=1.0, **kwargs):
    ''' Initialize Prior ParamBag object, with fields nu, B, m, kappa
          set according to match desired mean and expected covariance matrix.
    '''
    D = self.D
    nu = np.maximum(nu, D+2)
    if B is None:
      if ECovMat is None or type(ECovMat) == str:
        ECovMat = createECovMatFromUserInput(D, Data, ECovMat, sF)    
      B = ECovMat * (nu - D - 1)
    if B.ndim == 1:
      B = np.asarray([B], dtype=np.float)
    elif B.ndim == 0:
      B = np.asarray([[B]], dtype=np.float)
    if m is None:
      m = np.zeros(D)
    elif m.ndim < 1:
      m = np.asarray([m], dtype=np.float)      
    kappa = np.maximum(kappa, 1e-8)
    self.Prior = ParamBag(K=0, D=D)
    self.Prior.setField('nu', nu, dims=None)
    self.Prior.setField('kappa', kappa, dims=None)
    self.Prior.setField('m', m, dims=('D'))
    self.Prior.setField('B', B, dims=('D','D'))

  def get_mean_for_comp(self, k=None):
    if hasattr(self, 'EstParams'):
      return self.EstParams.mu[k]
    elif k is None or k == 'prior':
      return self.Prior.m
    else:
      return self.Post.m[k]

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
    return 'Gauss'

  def get_info_string(self):
    return 'Gaussian with full covariance.'
  
  def get_info_string_prior(self):
    msg = 'Gauss-Wishart on each mean/prec matrix pair: mu, Lam\n'
    if self.D > 2:
      sfx = ' ...'
    else:
      sfx = ''
    S = self._E_CovMat()[:2,:2]
    msg += 'E[ mu[k] ]     = %s%s\n' % (str(self.Prior.m[:2]), sfx)
    msg += 'E[ CovMat[k] ] = \n'
    msg += str(S) + sfx
    msg = msg.replace('\n', '\n  ')
    return msg

  ######################################################### Set EstParams
  #########################################################
  def setEstParams(self, obsModel=None, SS=None, LP=None, Data=None,
                          mu=None, Sigma=None,
                          **kwargs):
    ''' Create EstParams ParamBag with fields mu, Sigma
    '''
    self.ClearCache()
    if obsModel is not None:
      self.EstParams = obsModel.EstParams.copy()
      self.K = self.EstParams.K
      return
    
    if LP is not None and Data is not None:
      SS = self.calcSummaryStats(Data, None, LP)

    if SS is not None:
      self.updateEstParams(SS)
    else:
      self.EstParams = ParamBag(K=mu.shape[0], D=mu.shape[1])
      self.EstParams.setField('mu', mu, dims=('K', 'D'))
      self.EstParams.setField('Sigma', Sigma, dims=('K', 'D', 'D'))
      self.K = self.EstParams.K

  def setEstParamsFromPost(self, Post):
    ''' Convert from Post (nu, B, m, kappa) to EstParams (mu, Sigma),
         each EstParam is set to its posterior mean.
    '''
    self.EstParams = ParamBag(K=Post.K, D=Post.D)    
    mu = Post.m.copy()
    Sigma = Post.B / (Post.nu[:,np.newaxis,np.newaxis] - Post.D - 1)
    self.EstParams.setField('mu', mu, dims=('K','D'))
    self.EstParams.setField('Sigma', Sigma, dims=('K','D','D'))
    self.K = self.EstParams.K

  
  ######################################################### Set Post
  #########################################################
  def setPostFactors(self, obsModel=None, SS=None, LP=None, Data=None,
                            nu=0, B=0, m=0, kappa=0,
                            **kwargs):
    ''' Create Post ParamBag with fields nu, B, m, kappa
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
      m = as2D(m)
      if m.shape[1] != self.D:
        m = m.T.copy()
      K, _ = m.shape
      self.Post = ParamBag(K=K, D=self.D)
      self.Post.setField('nu', as1D(nu), dims=('K'))
      self.Post.setField('B', B, dims=('K', 'D', 'D'))
      self.Post.setField('m', m, dims=('K', 'D'))
      self.Post.setField('kappa', as1D(kappa), dims=('K'))
    self.K = self.Post.K

  def setPostFromEstParams(self, EstParams, Data=None, N=None):
    ''' Convert from EstParams (mu, Sigma) to Post (nu, B, m, kappa),
          each posterior hyperparam is set so EstParam is the posterior mean
    '''
    K = EstParams.K
    D = EstParams.D
    if Data is not None:
      N = Data.nObsTotal
    N = np.asarray(N, dtype=np.float)
    if N.ndim == 0:
      N = N/K * np.ones(K)

    nu = self.Prior.nu + N
    B = np.zeros( (K, D, D))
    for k in xrange(K):
      B[k] = (nu[k] - D - 1) * EstParams.Sigma[k]
    m = EstParams.mu.copy()
    kappa = self.Prior.kappa + N

    self.Post = ParamBag(K=K, D=D)
    self.Post.setField('nu', nu, dims=('K'))
    self.Post.setField('B', B, dims=('K', 'D', 'D'))
    self.Post.setField('m', m, dims=('K', 'D'))
    self.Post.setField('kappa', kappa, dims=('K'))
    self.K = self.Post.K

  ########################################################### Suff Stats
  ########################################################### 
  def calcSummaryStats(self, Data, SS, LP):
    X = Data.X
    resp = LP['resp']
    K = resp.shape[1]

    if SS is None:
      SS = SuffStatBag(K=K, D=Data.dim)
    
    # Expected count for each k
    #  Usually computed by allocmodel. But just in case...
    if not hasattr(SS, 'N'):      
      SS.setField('N', np.sum(resp, axis=0), dims='K')

    # Expected mean for each k
    SS.setField('x', dotATB(resp, X), dims=('K','D'))

    # Expected outer-product for each k 
    sqrtResp = np.sqrt(resp)
    xxT = np.zeros( (K, self.D, self.D) )
    for k in xrange(K):
      xxT[k] = dotATA(sqrtResp[:,k][:,np.newaxis]*Data.X )
    SS.setField('xxT', xxT, dims=('K','D','D'))
    return SS 

  def forceSSInBounds(self, SS):
    ''' Force count vector N to remain positive

        This avoids numerical problems due to incremental add/subtract ops
        which can cause computations like 
            x = 10.
            x += 1e-15
            x -= 10
            x -= 1e-15
        to be slightly different than zero instead of exactly zero.

        Returns
        -------
        None. SS.N updated in-place.
    '''
    np.maximum(SS.N, 0, out=SS.N)

  def incrementSS(self, SS, k, x):
    SS.x[k] += x
    SS.xxT[k] += np.outer(x,x)

  def decrementSS(self, SS, k, x):
    SS.x[k] -= x
    SS.xxT[k] -= np.outer(x,x)

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
               - 0.5 * self._mahalDist_EstParam(Data.X, k)
    return L

  def _mahalDist_EstParam(self, X, k):
    ''' Calc Mahalanobis distance from EstParams of comp k to every row of X

        Args
        ---------
        X : 2D array, size N x D
        k : integer ID of comp

        Returns
        ----------
        dist : 1D array, size N
    '''
    Q = np.linalg.solve(self.GetCached('cholSigma', k), \
                        (X-self.EstParams.mu[k]).T)
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

    mu = SS.x / SS.N[:,np.newaxis]
    minCovMat = self.min_covar * np.eye(SS.D)
    covMat = np.tile(minCovMat, (SS.K,1,1))
    for k in xrange(SS.K):
      covMat[k] += SS.xxT[k] / SS.N[k] - np.outer(mu[k], mu[k])      

    self.EstParams.setField('mu', mu, dims=('K', 'D'))
    self.EstParams.setField('Sigma', covMat, dims=('K', 'D', 'D'))
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

    Prior = self.Prior
    nu = Prior.nu + SS.N
    kappa = Prior.kappa + SS.N
    PB =  Prior.B + Prior.kappa * np.outer(Prior.m, Prior.m) 

    m = np.empty((SS.K, SS.D))
    B = np.empty((SS.K, SS.D, SS.D))
    for k in xrange(SS.K):
      km_x = Prior.kappa * Prior.m + SS.x[k]
      m[k] = 1.0/kappa[k] * km_x
      B[k] = PB + SS.xxT[k] - 1.0/kappa[k] * np.outer(km_x, km_x)
    
    mu, Sigma = MAPEstParams_inplace(nu, B, m, kappa)   
    self.EstParams.setField('mu', mu, dims=('K', 'D'))
    self.EstParams.setField('Sigma', Sigma, dims=('K', 'D', 'D'))
    self.K = SS.K

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

    nu, B, m, kappa = self.calcPostParams(SS)
    self.Post.setField('nu', nu, dims=('K'))
    self.Post.setField('kappa', kappa, dims=('K'))
    self.Post.setField('m', m, dims=('K', 'D'))
    self.Post.setField('B', B, dims=('K', 'D', 'D'))
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
    kappa = Prior.kappa + SS.N
    m = (Prior.kappa * Prior.m + SS.x) / kappa[:,np.newaxis]
    Bmm = Prior.B + Prior.kappa * np.outer(Prior.m, Prior.m)
    B = SS.xxT + Bmm[np.newaxis,:]
    for k in xrange(B.shape[0]):
      B[k] -= kappa[k] * np.outer(m[k], m[k])
    return nu, B, m, kappa

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
    if kB is None:
      SN = SS.N[kA]
      Sx = SS.x[kA]
      SxxT = SS.xxT[kA]
    else:
      SN = SS.N[kA] + SS.N[kB]
      Sx = SS.x[kA] + SS.x[kB]
      SxxT = SS.xxT[kA] + SS.xxT[kB]
    Prior = self.Prior
    nu = Prior.nu + SN
    kappa = Prior.kappa + SN
    m = (Prior.kappa * Prior.m + Sx) / kappa
    B = Prior.B + SxxT \
        + Prior.kappa * np.outer(Prior.m, Prior.m) \
        - kappa * np.outer(m, m)   
    return nu, B, m, kappa

  ########################################################### Stochastic Post
  ########################################################### update
  def updatePost_stochastic(self, SS, rho):
    ''' Stochastic update (in place) posterior for all comps given suff stats
    '''
    assert hasattr(self, 'Post')
    assert self.Post.K == SS.K
    self.ClearCache()
    
    self.convertPostToNatural()
    nu, Bnat, km, kappa = self.calcNaturalPostParams(SS)
    Post = self.Post
    Post.nu[:] = (1-rho) * Post.nu + rho * nu
    Post.Bnat[:] = (1-rho) * Post.Bnat + rho * Bnat
    Post.km[:] = (1-rho) * Post.km + rho * km
    Post.kappa[:] = (1-rho) * Post.kappa + rho * kappa
    self.convertPostToCommon()

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
    Prior = self.Prior
    nu = Prior.nu + SS.N
    kappa = Prior.kappa + SS.N
    km = Prior.kappa * Prior.m + SS.x
    Bnat = (Prior.B + Prior.kappa * np.outer(Prior.m, Prior.m)) + SS.xxT
    return nu, Bnat, km, kappa
    
  def convertPostToNatural(self):
    ''' Convert (in-place) current posterior params from common to natural form
    '''
    Post = self.Post
    assert hasattr(Post, 'nu')
    assert hasattr(Post, 'kappa')
    km = Post.m * Post.kappa[:,np.newaxis]
    Bnat = np.empty((self.K, self.D, self.D))
    for k in xrange(self.K):
      Bnat[k] = Post.B[k] + np.outer(km[k], km[k]) / Post.kappa[k]
    Post.setField('km', km, dims=('K','D'))
    Post.setField('Bnat', Bnat, dims=('K','D', 'D'))

  def convertPostToCommon(self):
    ''' Convert (in-place) current posterior params from natural to common form
    '''
    Post = self.Post
    assert hasattr(Post, 'nu')
    assert hasattr(Post, 'kappa')
    if hasattr(Post, 'm'):
      Post.m[:] = Post.km / Post.kappa[:,np.newaxis]
    else:
      m = Post.km / Post.kappa[:,np.newaxis]
      Post.setField('m', m, dims=('K','D'))

    if hasattr(Post, 'B'):
      B = Post.B # update in place, no reallocation!
    else:
      B = np.empty((self.K, self.D, self.D))
    for k in xrange(self.K):
      B[k] = Post.Bnat[k] - np.outer(Post.km[k], Post.km[k]) / Post.kappa[k]
    Post.setField('B', B, dims=('K','D', 'D'))


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
               - 0.5 * self._mahalDist_Post(Data.X, k)
    return L

  def _mahalDist_Post(self, X, k):
    ''' Calc expected mahalonobis distance from comp k to each data atom 

        Returns
        --------
        distvec : 1D array, size nObs
               distvec[n] gives E[ (x-\mu) \Lam (x-\mu) ] for comp k
    '''
    Q = np.linalg.solve(self.GetCached('cholB', k),
                        (X-self.Post.m[k]).T)
    Q *= Q
    return self.Post.nu[k] * np.sum(Q, axis=0) \
           + self.D / self.Post.kappa[k]

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
                        Prior.m, Prior.kappa,
                        Post.nu[k],
                        self.GetCached('logdetB', k),
                        Post.m[k], Post.kappa[k],
                        )
      if not afterMStep:
        aDiff = SS.N[k] + Prior.nu - Post.nu[k]
        bDiff = SS.xxT[k] + Prior.B \
                          + Prior.kappa * np.outer(Prior.m, Prior.m) \
                          - Post.B[k] \
                          - Post.kappa[k] * np.outer(Post.m[k], Post.m[k])
        cDiff = SS.x[k] + Prior.kappa * Prior.m \
                        - Post.kappa[k] * Post.m[k]
        dDiff = SS.N[k] + Prior.kappa - Post.kappa[k]
        elbo[k] += 0.5 * aDiff * self.GetCached('E_logdetL', k) \
                 - 0.5 * self._trace__E_L(bDiff, k) \
                 + np.inner(cDiff, self.GetCached('E_Lmu', k)) \
                 - 0.5 * dDiff * self.GetCached('E_muLmu', k)
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
    if alph.size < SS.K:
      alph = np.hstack([alph[:kdel], 0, alph[kdel:]])
    assert alph.size == SS.K
    assert np.allclose(alph[kdel], 0)

    Post = self.Post
    Prior = self.Prior
    gap = c_Func(Post.nu[kdel], Post.B[kdel], Post.m[kdel], Post.kappa[kdel]) \
           - c_Func(Prior.nu,   Prior.B,      Prior.m,      Prior.kappa)
    for k in xrange(SS.K):
      if k == kdel:
        continue
      nu = Post.nu[k] + alph[k] * SS.N[kdel]
      kappa = Post.kappa[k] + alph[k] * SS.N[kdel]
      km_x = Post.kappa[k] * Post.m[k] + alph[k] * SS.x[kdel]
      m = 1/kappa * (km_x)
      B = Post.B[k] + Post.kappa[k] * np.outer(Post.m[k], Post.m[k]) \
             + alph[k] * SS.xxT[kdel] \
             - 1.0/kappa * np.outer(km_x, km_x)
      gap += c_Func(Post.nu[k],
                     Post.B[k],
                     Post.m[k],
                     Post.kappa[k]) \
              - c_Func(nu, B, m, kappa)
    return gap

  def calcSoftMergeGap_alph(self, SS, kdel, alph):
    ''' Calculate net improvement in ELBO after multi-way merge as fcn of alph.
        
        This keeps only terms that depend on redistribution vector alph
    '''
    if alph.size < SS.K:
      alph = np.hstack([alph[:kdel], 0, alph[kdel:]])
    assert alph.size == SS.K
    assert np.allclose(alph[kdel], 0)

    gap = 0
    Post = self.Post
    assert np.allclose(alph.sum(), 1.0)
    for k in xrange(SS.K):
      if k == kdel:
        continue
      nu = Post.nu[k] + alph[k] * SS.N[kdel]
      kappa = Post.kappa[k] + alph[k] * SS.N[kdel]
      km_x = Post.kappa[k] * Post.m[k] + alph[k] * SS.x[kdel]
      m = 1/kappa * (km_x)
      B = Post.B[k] + Post.kappa[k] * np.outer(Post.m[k], Post.m[k]) \
             + alph[k] * SS.xxT[kdel] \
             - 1.0/kappa * np.outer(km_x, km_x)
      gap -= c_Func(nu, B, m, kappa)
    return gap




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
    nu, beta, m, kappa = self.calcPostParamsForComp(SS, kA, kB)
    return -1 * c_Func(nu, beta, m, kappa)

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
    nu, B, m, kappa = self.calcPostParams(SS)
    pSS = SS.copy()
    pSS.N += 1
    pSS.x += x[np.newaxis,:]
    pSS.xxT += np.outer(x,x)[np.newaxis,:,:]
    pnu, pB, pm, pkappa = self.calcPostParams(pSS)
    logp = np.zeros(SS.K)
    for k in xrange(SS.K):
      logp[k] = c_Diff(nu[k], B[k], m[k], kappa[k],
                       pnu[k], pB[k], pm[k], pkappa[k])
    return np.exp(logp - np.max(logp))

  def _calcPredProbVec_Fast(self, SS, x):
    nu, B, m, kappa = self.calcPostParams(SS)
    kB = B
    kB *= ((kappa+1)/kappa)[:,np.newaxis, np.newaxis]
    logp = np.zeros(SS.K)
    p = logp # Rename so its not confusing what we're returning
    for k in xrange(SS.K):
      cholKB = scipy.linalg.cholesky(kB[k], lower=1)
      logdetKB = 2 * np.sum(np.log(np.diag(cholKB)))
      mVec = np.linalg.solve(cholKB, x - m[k])
      mDist_k = np.inner(mVec, mVec)
      logp[k] = -0.5 * logdetKB - 0.5 * (nu[k]+1) * np.log(1.0 + mDist_k)
    logp += gammaln(0.5 * (nu+1)) - gammaln(0.5 * (nu+1-self.D))
    logp -= np.max(logp)
    np.exp(logp, out=p)
    return p
  
  def _Verify_calcPredProbVec(self, SS, x):
    ''' Verify that the predictive prob vector is correct,
          by comparing very different implementations
    '''
    pA = self._calcPredProbVec_Fast(SS, x)
    pC = self._calcPredProbVec_cFunc(SS, x)
    pA /= np.sum(pA)
    pC /= np.sum(pC)
    assert np.allclose(pA, pC)
  
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
    
  def _E_logdetL(self, k=None):
    dvec = np.arange(1, self.D+1, dtype=np.float)
    if k is None:
      nu = self.Prior.nu
    else:
      nu = self.Post.nu[k]
    return self.D * LOGTWO \
           - self.GetCached('logdetB', k) \
           + np.sum(digamma(0.5 * (nu + 1 - dvec)))

  def _trace__E_L(self, Smat, k=None):
    if k is None:
      nu = self.Prior.nu
      B = self.Prior.B
    else:
      nu = self.Post.nu[k]
      B = self.Post.B[k]
    return nu * np.trace(np.linalg.solve(B, Smat))
    
  def _E_Lmu(self, k=None):
    if k is None:
      nu = self.Prior.nu
      B = self.Prior.B
      m = self.Prior.m
    else:
      nu = self.Post.nu[k]
      B = self.Post.B[k]
      m = self.Post.m[k]
    return nu * np.linalg.solve(B, m)

  def _E_muLmu(self, k=None):
    if k is None:
      nu = self.Prior.nu
      kappa = self.Prior.kappa
      m = self.Prior.m
      B = self.Prior.B
    else:
      nu = self.Post.nu[k]
      kappa = self.Post.kappa[k]
      m = self.Post.m[k]
      B = self.Post.B[k]
    Q = np.linalg.solve(self.GetCached('cholB', k), m.T)
    return self.D / kappa + nu * np.inner(Q, Q)

  # ......................................................########
  # ......................................................########
  ################################################################ end class

def MAPEstParams_inplace(nu, B, m, kappa=0):
  ''' MAP estimate parameters mu, Sigma given Normal-Wishart hyperparameters
  '''
  D = m.size
  mu = m
  Sigma = B
  for k in xrange(B.shape[0]):
    Sigma[k] /= (nu[k] + D + 1)
  return mu, Sigma

def c_Func(nu, logdetB, m, kappa):
  ''' Evaluate cumulant function c, aka log partition function, at given params

      c is the cumulant of the multivariate Normal-Wishart, using common params.

      Returns
      --------
      c : scalar real value of cumulant function at provided args
  '''
  if logdetB.ndim >= 2:
    logdetB = np.log(np.linalg.det(logdetB))
  D = m.size
  dvec = np.arange(1, D+1, dtype=np.float)
  return - 0.5 * D * LOGTWOPI \
         - 0.25 * D * (D-1) * LOGPI \
         - 0.5 * D * LOGTWO * nu \
         - np.sum( gammaln( 0.5 * (nu + 1 - dvec) )) \
         + 0.5 * D * np.log(kappa) \
         + 0.5 * nu * logdetB

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
  dvec = np.arange(1, D+1, dtype=np.float)
  return - 0.5 * D * LOGTWO * (nu1 - nu2) \
         - np.sum( gammaln( 0.5 * (nu1 + 1 - dvec) )) \
         + np.sum( gammaln( 0.5 * (nu2 + 1 - dvec) )) \
         + 0.5 * D * (np.log(kappa1) - np.log(kappa2)) \
         + 0.5 * (nu1 * logdetB1 - nu2 * logdetB2)



def createECovMatFromUserInput(D=0, Data=None, ECovMat='eye', sF=1.0):
  ''' Create expected covariance matrix defining Wishart prior.

      The creation process follows user-specified criteria.

      Args
      --------
      D : positive integer, size of each observation
      Data : [optional] dataset to use to make Sigma in data-driven way
      ECovMat : string name of the procedure to use to create Sigma
             * 'eye' : make Sigma a multiple of the identity matrix
             * 'covdata' : set Sigma to a multiple of the data covariance matrix
             * 'fromtruelabels' : set Sigma to the empirical mean of the 
                                  covariances for each true cluster in the dataset

      Returns
      --------
      Sigma : 2D array, size D x D, symmetric and pos definite
  '''
  if Data is not None:
    assert D == Data.dim
  if ECovMat == 'eye':
    Sigma = sF * np.eye(D)
  elif ECovMat == 'covdata':
    Sigma = sF * np.cov(Data.X.T, bias=1)
  elif ECovMat == 'covfirstdiff':
    if not hasattr(Data, 'Xprev'):
      raise ValueError('covfirstdiff only applies to auto-regressive datasets')
    Xdiff = Data.X - Data.Xprev
    Sigma = sF * np.cov(Xdiff.T, bias=1)
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
    SampleCov = np.zeros((Kmax,D,D))
    for kLoc, kVal in enumerate(Zvals):
      mask = Z == kVal
      wHat[kLoc] = np.sum(mask)
      SampleCov[kLoc] = np.cov(Data.X[mask].T, bias=1)
    wHat = wHat/np.sum(wHat)
    Sigma = 1e-8 * np.eye(D)
    for k in range(Kmax):
      Sigma += wHat[k] * SampleCov[k]
  else:
    raise ValueError('Unrecognized ECovMat procedure %s' % (ECovMat))
  return Sigma
