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
    else:
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

  def get_mean_for_comp(self, k):
    if hasattr(self, 'EstParams'):
      return self.EstParams.mu[k]
    else:
      return self.Post.m[k]

  def get_covar_mat_for_comp(self, k):
    if hasattr(self, 'EstParams'):
      return self.EstParams.Sigma[k]
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
      return
    
    if LP is not None and Data is not None:
      SS = self.calcSummaryStats(Data, None, LP)

    if SS is not None:
      self.updateEstParams(SS)
    else:
      self.EstParams = ParamBag(K=mu.shape[0], D=mu.shape[1])
      self.EstParams.setField('mu', mu, dims=('K', 'D'))
      self.EstParams.setField('Sigma', Sigma, dims=('K', 'D', 'D'))

  def setEstParamsFromPost(self, Post):
    ''' Convert from Post (nu, B, m, kappa) to EstParams (mu, Sigma),
         each EstParam is set to its posterior mean.
    '''
    self.EstParams = ParamBag(K=K, D=D)    
    mu = Post.m.copy()
    Sigma = Post.B / (nu[k] - D - 1)
    self.EstParams.setField('mu', mu, dims=('K','D'))
    self.EstParams.setField('Sigma', Sigma, dims=('K','D','D'))

  
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
    if type(N) == float or N.ndim == 0:
      N = float(N)/K * np.ones(K)

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

  ########################################################### Summary
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

  def incrementSS(self, SS, k, x):
    SS.x[k] += x
    SS.xxT[k] += np.outer(x)

  def decrementSS(self, SS, k, x):
    SS.x[k] -= x
    SS.xxT[k] -= np.outer(x)

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

  def calcPostParamsForComp(SS, kA=None, kB=None):
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
    nu = Prior.nu + SN
    kappa = Prior.kapa + SN
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
    return self.calcMargLik_CFuncForLoop(SS)

  def calcMargLik_Vec(self, SS):
    ''' Calculate scalar marginal likelihood probability, summed over all comps
    '''
    Prior = self.Prior
    nu, beta, m, kappa = self.calcPostParams(SS)
    logp = 0.5 * np.sum(np.log(Prior.kappa) - np.log(kappa)) \
           + 0.5 * LOGTWO * np.sum(nu - Prior.nu) \
           + np.sum(gammaln(0.5*nu) - gammaln(0.5*Prior.nu)) \
           + 0.5 * np.sum(Prior.nu * np.log(Prior.beta) \
                          - nu[:,np.newaxis] * np.log(beta))
    return logp - 0.5 * np.sum(SS.N) * LOGTWOPI
  
  def calcMargLik_CFuncForLoop(self, SS):
    Prior = self.Prior
    logp = np.zeros(SS.K)
    for k in xrange(SS.K):
      nu, beta, m, kappa = self.calcPostParamsForComp(SS, k)
      logp[k] = c_Diff(Prior.nu, Prior.beta, Prior.m, Prior.kappa,
                       nu, beta, m, kappa)
    return np.sum(logp) - 0.5 * np.sum(SS.N) * LOGTWOPI
  
  def calcMargLik_ForLoop(self, SS):
    Prior = self.Prior
    logp = np.zeros(SS.K)
    for k in xrange(SS.K):
      nu, beta, m, kappa = self.calcPostParamsForComp(SS, k)
      logp[k] = 0.5 * SS.D * (np.log(Prior.kappa) - np.log(kappa)) \
                + 0.5 * SS.D * LOGTWO * (nu - Prior.nu) \
                + SS.D * (gammaln(0.5 * nu) - gammaln(0.5 * Prior.nu)) \
                + 0.5 * np.sum(Prior.nu * np.log(Prior.beta) \
                               - nu * np.log(beta))
    return np.sum(logp) - 0.5 * np.sum(SS.N) * LOGTWOPI

  ########################################################### Gibbs Pred Prob
  ########################################################### 
  def calcPredProbVec_Unnorm(self, SS, x):
    ''' Calculate K-vector of positive entries \propto p( x | SS[k] )
    '''
    return self._calcPredProbVec_Fast(SS, x)

  def _calcPredProbVec_Naive(self, SS, x):
    nu, beta, m, kappa = self.calcPostParams(SS)
    pSS = SS.copy()
    pSS.N += 1
    pSS.x += x
    pSS.xxT += np.outer(x)
    pnu, pbeta, pm, pkappa = self.calcPostParams(pSS)
    logp = np.zeros(SS.K)
    for k in xrange(SS.K):
      logp[k] = c_Diff(nu[k], beta[k], m[k], kappa[k],
                       pnu[k], pbeta[k], pm[k], pkappa[k])
    return np.exp(logp - np.max(logp))

  def _calcPredProbVec_Fast(self, SS, x):
    p = np.zeros(SS.K)
    nu, beta, m, kappa = self.calcPostParams(SS)
    kbeta = beta
    kbeta *= ( (kappa+1)/kappa )[:,np.newaxis]
    base = np.square(x - m)
    base /= kbeta
    base += 1
    ## logp : 2D array, size K x D
    logp = (-0.5 * (nu+1))[:,np.newaxis] * np.log(base)
    logp += (gammaln(0.5 * (nu+1)) - gammaln(0.5 * nu))[:,np.newaxis]
    logp -= 0.5 * np.log(kbeta)

    ## p : 1D array, size K
    p = np.sum(logp, axis=1)
    p -= np.max(p)
    np.exp(p, out=p)
    return p

  def _calcPredProbVec_ForLoop(self, SS, x):
    ''' For-loop version
    '''
    p = np.zeros(SS.K)
    for k in xrange(SS.K):
      nu, beta, m, kappa = self.calcPostParamsForComp(SS, k)
      kbeta = (kappa+1)/kappa * beta
      base = np.square(x - m)
      base /= kbeta
      base += 1
      p_k = np.exp(gammaln(0.5 * (nu+1)) - gammaln(0.5 * nu)) \
             * 1.0 / np.sqrt(kbeta) \
             * base ** (-0.5 * (nu+1))
      p[k] = np.prod(p_k)
    return p
  
  def _Verify_calcPredProbVec(self, SS, x):
    ''' Verify that the predictive prob vector is correct,
          by comparing 3 very different implementations
    '''
    pA = self._calcPredProbVec_Fast(SS, x)
    pB = self._calcPredProbVec_Naive(SS, x)
    pC = self._calcPredProbVec_ForLoop(SS, x)
    pA /= np.sum(pA)
    pB /= np.sum(pB)
    pC /= np.sum(pC)
    assert np.allclose(pA, pB)
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
  D = m1.size
  dvec = np.arange(1, D+1, dtype=np.float)
  return - 0.5 * D * LOGTWO * (nu1 - nu2) \
         - np.sum( gammaln( 0.5 * (nu1 + 1 - dvec) )) \
         + np.sum( gammaln( 0.5 * (nu2 + 1 - dvec) )) \
         + 0.5 * D * (np.log(kappa1) - np.log(kappa2)) \
         + 0.5 * (nu1 * logdetB1 - nu2 * logdetB2)

def createECovMatFromUserInput(D=0, Data=None, ECovMat='eye', sF=1.0):
  if Data is not None:
    assert D == Data.dim
  if ECovMat == 'eye':
    Sigma = sF * np.eye(D)
  elif ECovMat == 'covdata':
    Sigma = sF * np.cov(Data.X.T, bias=1)
  elif ECovMat == 'fromtruelabels':    
    ''' Set Cov Matrix Sigma using the true labels in empirical Bayes style
        Sigma = \sum_{c : class labels} w_c * SampleCov[ data from class c]
    '''   
    assert hasattr(Data, 'TrueLabels')
    Zvals = np.unique(Data.TrueLabels)
    Kmax = len(Zvals)
    wHat = np.zeros(Kmax)
    SampleCov = np.zeros((Kmax,D,D))
    for kLoc, kVal in enumerate(Zvals):
      mask = Data.TrueLabels == kVal
      wHat[kLoc] = np.sum(mask)
      SampleCov[kLoc] = np.cov(Data.X[mask].T, bias=1)
    wHat = wHat/np.sum(wHat)
    Sigma = 1e-8 * np.eye(D)
    for k in range(Kmax):
      Sigma += wHat[k] * SampleCov[k]
  else:
    raise ValueError('Unrecognized ECovMat procedure %s' % (ECovMat))
  return Sigma
