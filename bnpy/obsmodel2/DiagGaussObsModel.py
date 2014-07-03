'''
DiagGaussObsModel

Prior : Normal-Wishart-1D (on each dimension
* nu
* beta
* m
* kappa

EstParams 
* mu     2D array, size KxD
* sigma  2D array, size KxD

Posterior : Normal-Wishart
* nu[k]
* beta[k]
* m[k]
* kappa[k]

'''
import numpy as np
import scipy.linalg
from scipy.special import gammaln, digamma

from bnpy.suffstats import ParamBag, SuffStatBag
from bnpy.util import LOGTWO, LOGPI, LOGTWOPI, EPS
from bnpy.util import dotATA, dotATB, dotABT

from AbstractObsModel import AbstractObsModel 

class GaussObsModel(AbstractObsModel):

  def __init__(self, inferType='EM', D=0, min_covar=None, 
                     Data=None, **PriorArgs):
    ''' Initialize bare Gaussian obsmodel with Normal-Wishart prior. 
        Resulting object lacks either EstParams or Post, 
          which must be created separately.
    '''
    if Data is not None:
      self.D = Data.dim
    else:
      self.D = D
    self.K = 0
    self.inferType = inferType
    self.min_covar = min_covar
    self.createPrior(Data, **PriorArgs)
    self.Cache = dict()

  def createPrior(self, Data, nu=0, ECovMat=None, sF=1.0,
                              m=None, kappa=None):
    ''' Initialize Prior ParamBag object, with fields nu, B, m, kappa
          set according to match desired mean and expected covariance matrix.
    '''
    D = self.D
    nu = np.maximum(nu, D+2)
    if ECovMat is None or type(ECovMat) == str:
      ECovMat = createECovMatFromUserInput(D, Data, ECovMat, sF)    
    beta = np.diag(ECovMat) * (nu - D - 1)
    if m is None: 
      m = np.zeros(D)
    kappa = np.maximum(kappa, 1e-8)
    self.Prior = ParamBag(K=0, D=D)
    self.Prior.setField('nu', nu, dims=None)
    self.Prior.setField('kappa', kappa, dims=None)
    self.Prior.setField('m', m, dims=('D'))
    self.Prior.setField('beta', beta, dims=('D'))

  ######################################################### Set EstParams
  #########################################################
  def setEstParams(self, obsModel=None, SS=None, LP=None, Data=None,
                          mu=None, sigma=None,
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
      self.EstParams.setField('sigma', Sigma, dims=('K', 'D'))

  def setEstParamsFromPost(self, Post):
    ''' Convert from Post (nu, beta, m, kappa) to EstParams (mu, Sigma),
         each EstParam is set to its posterior mean.
    '''
    self.EstParams = ParamBag(K=K, D=D)    
    mu = Post.m.copy()
    sigma = Post.beta / (nu[k] - 2)
    self.EstParams.setField('mu', mu, dims=('K','D'))
    self.EstParams.setField('sigma', sigma, dims=('K','D'))

  
  ######################################################### Set Post
  #########################################################
  def setPostFactors(self, obsModel=None, SS=None, LP=None, Data=None,
                            nu=0, beta=0, m=0, kappa=0,
                            **kwargs):
    ''' Create Post ParamBag with fields nu, beta, m, kappa
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
      self.Post = ParamBag(K=K, D=mu.shape[1])
      self.Post.setField('nu', nu, dims=('K'))
      self.Post.setField('beta', beta, dims=('K', 'D'))
      self.Post.setField('m', m, dims=('K', 'D'))
      self.Post.setField('kappa', kappa, dims=('K'))

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
    beta = np.zeros( (K, D))
    beta = (nu - 2) * EstParams.sigma
    m = EstParams.mu.copy()
    kappa = self.Prior.kappa + N

    self.Post = ParamBag(K=K, D=D)
    self.Post.setField('nu', nu, dims=('K'))
    self.Post.setField('beta', beta, dims=('K', 'D'))
    self.Post.setField('m', m, dims=('K', 'D'))
    self.Post.setField('kappa', kappa, dims=('K'))

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
    SS.setField('xx', dotATB(resp, np.square(X)), dims=('K', 'D'))
    return SS 

  ########################################################### EM
  ########################################################### 
  # _________________________________________________________ E step
  def calcSoftEvMatrix_FromEstParams(self, Data):
    for k in xrange(K):
      L[:,k] = - 0.5 * self.D * LOGTWOPI \
               - 0.5 * np.sum(np.log(self.EstParams.sigma)) \
               - 0.5 * self._mahalDist_EstParam(Data.X, k)

  def _mahalDist_EstParam(self, X, k):
    Xdiff = (X - self.EstParams.mu[k])
    np.square(Xdiff, out=Xdiff)
    dist = np.sum(1.0/self.EstParams.sigma[k] * Xdiff, axis=0)
    return dist

  # _________________________________________________________  M step
  def updateEstParams_MaxLik(self, SS):
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

  def updateEstParams_MAP(self, SS):
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

  ########################################################### VB
  ########################################################### 

  def calcSoftEvMatrix_FromPost(self, Data):
    ''' Calculate soft ev matrix 

        Returns
        ------
        L : 2D array, size nObs x K
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
    Q = np.linalg.solve(self.GetCached('cholB', k), \
                        (X-self.Post.m[k]).T)
    Q *= Q
    return self.Post.nu[k] * np.sum(Q, axis=0) \
           + self.D / self.Post.kappa[k]

  def updatePost(self, SS):
    ''' Update the Post ParamBag, so each component 1, 2, ... K
          contains Normal-Wishart posterior params given Prior and SS
    '''
    self.ClearCache()
    if not hasattr(self, 'Post') or self.Post.K != SS.K:
      self.Post = ParamBag(K=SS.K, D=SS.D)

    Prior = self.Prior # use 'Prior' not 'self.Prior', improves readability
    Post = self.Post

    Post.setField('nu', Prior.nu + SS.N, dims=('K'))
    Post.setField('kappa', Prior.kappa + SS.N, dims=('K'))
    PB = Prior.B + Prior.kappa * np.outer(Prior.m, Prior.m)
    m = np.empty((SS.K, SS.D))
    B = np.empty((SS.K, SS.D, SS.D))
    for k in xrange(SS.K):
      km_x = Prior.kappa * Prior.m + SS.x[k]
      m[k] = 1.0/Post.kappa[k] * km_x
      B[k] = PB + SS.xxT[k] - 1.0/Post.kappa[k] * np.outer(km_x, km_x)
    Post.setField('m', m, dims=('K', 'D'))
    Post.setField('B', B, dims=('K', 'D', 'D'))

  def calcELBO_Memoized(self, SS, doFast=False):
    ''' Calculate obsModel's ELBO using sufficient statistics SS and Post.

        Args
        -------
        SS : bnpy SuffStatBag, contains fields for N, x, xxT
        doFast : boolean flag
                 if 1, elbo calculated assuming special terms cancel out

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
      if not doFast and SS.N[k] > 1e-9:
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
                 + 0.5 * np.inner(cDiff, self.GetCached('E_Lmu', k)) \
                 - 0.5 * dDiff * self.GetCached('E_muLmu', k)
    return elbo.sum() - 0.5 * np.sum(SS.N) * SS.D * LOGTWOPI

  def calcMergeELBO_alph(self, SS, kdel, alph):
    elbo = 0
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
      elbo -= c_Func(nu, B, m, kappa)
    return elbo

  def calcMergeELBO(self, SS, kdel, alph):
    ''' Calculate change in ELBO after a multi-way merge applied to current model/SS
    '''
    assert alph.size == SS.K
    assert np.allclose(alph[kdel], 0)
    Post = self.Post
    Prior = self.Prior
    elbo = c_Func(Post.nu[kdel],
                  Post.B[kdel],
                  Post.m[kdel],
                  Post.kappa[kdel]) \
           - c_Func(Prior.nu, 
                    Prior.B,
                    Prior.m,
                    Prior.kappa)
    
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
      elbo += c_Func(Post.nu[k],
                     Post.B[k],
                     Post.m[k],
                     Post.kappa[k]) \
              - c_Func(nu, B, m, kappa)
    return elbo

  ########################################################### Gibbs
  ########################################################### 
  def calcMargLik(self):
    pass
  
  def calcPredLik(self, xSS):
    pass

  def incrementPost(self, k, x):
    ''' Add data to the Post ParamBag, component k
    '''
    Post = self.Post
    Post.nu[k] += 1
    kappa = Post.kappa[k] + 1
    Post.B[k] += Post.kappa[k]/kappa * np.outer(x-Post.m[k], x-Post.m[k]) 
    Post.m[k] = 1/(kappa) * (Post.kappa[k] * Post.m[k] + x)
    Post.kappa[k] = kappa
    # TODO: update cached cholesky and log det with rank-one updates

  def decrementPost(self, k, x):
    ''' Remove data from the Post ParamBag, component k
    '''
    Post = self.Post
    Post.nu[k] -= 1
    kappa = Post.kappa[k] - 1
    Post.B[k] -= Post.kappa[k]/kappa * np.outer(x-Post.m[k], x-Post.m[k]) 
    Post.m[k] = 1/(kappa) * (Post.kappa[k] * Post.m[k] - x)
    Post.kappa[k] = kappa
    # TODO: update cached cholesky and log det with rank-one updates

  ########################################################### Expectations
  ########################################################### 
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
           + np.sum(np.log(0.5 * (nu + 1 - dvec)))

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
    else:
      nu = self.Post.nu[k]
      kappa = self.Post.kappa[k]
      m = self.Post.m[k]
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
         + 0.5 * D * kappa \
         + 0.5 * nu * logdetB

def c_Diff(nu1, logdetB1, m1, kappa1,
           nu2, logdetB2, m2, kappa2):
  D = m1.size
  dvec = np.arange(1, D+1, dtype=np.float)
  return -0.5 * D * LOGTWO * (nu1 - nu2) \
         - np.sum( gammaln( 0.5 * (nu1 + 1 - dvec) )) \
         + np.sum( gammaln( 0.5 * (nu2 + 1 - dvec) )) \
         + 0.5 * D * (kappa1 - kappa2) \
         + 0.5 * (nu1 * logdetB1 - nu2 * logdetB2)

def createECovMatFromUserInput(D=0, Data=None, ECovMat='eye', sF=1.0):
  if Data is not None:
    assert D == Data.dim
  if ECovMat == 'eye':
    Sigma = sF * np.eye(D)
  elif ECovMat == 'covdata':
    Sigma = sF * np.cov(Data.X.T, bias=1)
  elif ECovMat == 'fromtruelabels':    
    raise NotImplementedError('TODO')
  else:
    raise ValueError('Unrecognized ECovMat procedure %s' % (ECovMat))
  return Sigma
