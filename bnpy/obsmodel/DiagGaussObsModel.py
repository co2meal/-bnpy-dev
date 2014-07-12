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
from bnpy.util import as1D, as2D

from AbstractObsModel import AbstractObsModel 

class DiagGaussObsModel(AbstractObsModel):

  def __init__(self, inferType='EM', D=0, min_covar=None, 
                     Data=None, **PriorArgs):
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

  def createPrior(self, Data, nu=0, beta=None,
                              m=None, kappa=None,
                              ECovMat=None, sF=1.0, **kwargs):
    ''' Initialize Prior ParamBag object, with fields nu, beta, m, kappa
          set according to match desired mean and expected covariance matrix.
    '''
    D = self.D
    nu = np.maximum(nu, D+2)
    kappa = np.maximum(kappa, 1e-8)
    if beta is None:
      if ECovMat is None or type(ECovMat) == str:
        ECovMat = createECovMatFromUserInput(D, Data, ECovMat, sF)    
      beta = np.diag(ECovMat) * (nu - 2)
    else:
      if beta.ndim == 0:
        beta = np.asarray([beta], dtype=np.float)
    if m is None:
      m = np.zeros(D)
    elif m.ndim < 1:
      m = np.asarray([m], dtype=np.float)       
    self.Prior = ParamBag(K=0, D=D)
    self.Prior.setField('nu', nu, dims=None)
    self.Prior.setField('kappa', kappa, dims=None)
    self.Prior.setField('m', m, dims=('D'))
    self.Prior.setField('beta', beta, dims=('D'))

  def get_mean_for_comp(self, k):
    if hasattr(self, 'EstParams'):
      return self.EstParams.mu[k]
    else:
      return self.Post.m[k]

  def get_covar_mat_for_comp(self, k):
    if hasattr(self, 'EstParams'):
      return np.diag(self.EstParams.sigma[k])
    else:
      return self._E_CovMat(k)
    
  ######################################################### I/O Utils
  #########################################################   for humans
  def get_name(self):
    return 'DiagGauss'

  def get_info_string(self):
    return 'Gaussian with diagonal covariance.'
  
  def get_info_string_prior(self):
    msg = 'Gauss-Wishart on each pair mu, lam (each dim independent)\n'
    if self.D > 2:
      sfx = ' ...'
    else:
      sfx = ''
    S = self._E_CovMat()[:2]
    msg += 'E[ mu[k] ]     = %s%s\n' % (str(self.Prior.m[:2]), sfx)
    msg += 'E[ CovMat[k] ] = \n'
    msg += str(S) + sfx
    msg = msg.replace('\n', '\n  ')
    return msg

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
      self.K = self.EstParams.K
      return
    
    if LP is not None and Data is not None:
      SS = self.calcSummaryStats(Data, None, LP)

    if SS is not None:
      self.updateEstParams(SS)
    else:
      self.EstParams = ParamBag(K=mu.shape[0], D=mu.shape[1])
      self.EstParams.setField('mu', mu, dims=('K', 'D'))
      self.EstParams.setField('sigma', Sigma, dims=('K', 'D'))
    self.K = self.EstParams.K

  def setEstParamsFromPost(self, Post):
    ''' Convert from Post (nu, beta, m, kappa) to EstParams (mu, Sigma),
         each EstParam is set to its posterior mean.
    '''
    self.EstParams = ParamBag(K=K, D=D)    
    mu = Post.m.copy()
    sigma = Post.beta / (nu[k] - 2)
    self.EstParams.setField('mu', mu, dims=('K','D'))
    self.EstParams.setField('sigma', sigma, dims=('K','D'))
    self.K = self.EstParams.K
  
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
      beta = as2D(beta)
      if beta.shape[1] != self.D:
        beta = beta.T.copy()
      K, _ = m.shape
      self.Post = ParamBag(K=K, D=self.D)
      self.Post.setField('nu', as1D(nu), dims=('K'))
      self.Post.setField('beta', beta, dims=('K', 'D'))
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
    beta = np.zeros( (K, D))
    beta = (nu - 2) * EstParams.sigma
    m = EstParams.mu.copy()
    kappa = self.Prior.kappa + N

    self.Post = ParamBag(K=K, D=D)
    self.Post.setField('nu', nu, dims=('K'))
    self.Post.setField('beta', beta, dims=('K', 'D'))
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

    # Expected sum-of-squares for each k
    SS.setField('xx', dotATB(resp, np.square(X)), dims=('K', 'D'))
    return SS 

  ########################################################### EM
  ########################################################### 
  # _________________________________________________________ E step
  def calcLogSoftEvMatrix_FromEstParams(self, Data):
    K = self.EstParams.K
    L = np.zeros((Data.nObs, K))
    for k in xrange(K):
      L[:,k] = - 0.5 * self.D * LOGTWOPI \
               - 0.5 * np.sum(np.log(self.EstParams.sigma)) \
               - 0.5 * self._mahalDist_EstParam(Data.X, k)
    return L

  def _mahalDist_EstParam(self, X, k):
    ''' Calculate distance to every row of matrix X

        Args
        -------
        X : 2D array, size N x D

        Returns
        ------
        dist : 1D array, size N
    '''
    Xdiff = (X - self.EstParams.mu[k])
    np.square(Xdiff, out=Xdiff)
    dist = np.sum(Xdiff/self.EstParams.sigma[k], axis=1)
    return dist

  # _________________________________________________________  M step
  def updateEstParams_MaxLik(self, SS):
    self.ClearCache()
    if not hasattr(self, 'EstParams') or self.EstParams.K != SS.K:
      self.EstParams = ParamBag(K=SS.K, D=SS.D)

    mu = SS.x / SS.N[:,np.newaxis]
    sigma = self.min_covar \
            + SS.xx / SS.N[:,np.newaxis] \
            - np.square(mu)

    self.EstParams.setField('mu', mu, dims=('K', 'D'))
    self.EstParams.setField('sigma', sigma, dims=('K', 'D'))
    self.K = SS.K

  def updateEstParams_MAP(self, SS):
    self.ClearCache()
    if not hasattr(self, 'EstParams') or self.EstParams.K != SS.K:
      self.EstParams = ParamBag(K=SS.K, D=SS.D)

    Prior = self.Prior
    nu = Prior.nu + SS.N
    kappa = Prior.kappa + SS.N
    PB =  Prior.beta + Prior.kappa * np.square(Prior.m)

    m = np.empty((SS.K, SS.D))
    beta = np.empty((SS.K, SS.D))
    for k in xrange(SS.K):
      km_x = Prior.kappa * Prior.m + SS.x[k]
      m[k] = 1.0/kappa[k] * km_x
      beta[k] = PB + SS.xx[k] - 1.0/kappa[k] * np.square(km_x)
    
    mu, sigma = MAPEstParams_inplace(nu, beta, m, kappa)   
    self.EstParams.setField('mu', mu, dims=('K', 'D'))
    self.EstParams.setField('sigma', sigma, dims=('K', 'D'))
    self.K = SS.K

  ########################################################### VB
  ########################################################### 

  def calcLogSoftEvMatrix_FromPost(self, Data):
    ''' Calculate soft ev matrix 

        Returns
        ------
        L : 2D array, size nObs x K
    '''
    K = self.Post.K
    L = np.zeros((Data.nObs, K))
    for k in xrange(K):
      L[:,k] = - 0.5 * self.D * LOGTWOPI \
               + 0.5 * np.sum(self.GetCached('E_logL', k)) \
               - 0.5 * self._mahalDist_Post(Data.X, k)
    return L

  def _mahalDist_Post(self, X, k):
    ''' Calc expected mahalonobis distance from comp k to each data atom 

        Returns
        --------
        distvec : 1D array, size nObs
               distvec[n] gives E[ \Lam (x-\mu)^2 ] for comp k
    '''
    Xdiff = X - self.Post.m[k]
    np.square(Xdiff, out=Xdiff)
    dist = np.dot(Xdiff, self.Post.nu[k] / self.Post.beta[k])
    dist += self.D / self.Post.kappa[k]
    return dist

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
    PB = Prior.beta + Prior.kappa * np.square(Prior.m)
    m = np.empty((SS.K, SS.D))
    beta = np.empty((SS.K, SS.D))
    for k in xrange(SS.K):
      km_x = Prior.kappa * Prior.m + SS.x[k]
      m[k] = 1.0/Post.kappa[k] * km_x
      beta[k] = PB + SS.xx[k] - 1.0/Post.kappa[k] * np.square(km_x)
    Post.setField('m', m, dims=('K', 'D'))
    Post.setField('beta', beta, dims=('K', 'D'))
    self.K = SS.K

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
      elbo[k] = c_Diff(Prior.nu,   Prior.beta,   Prior.m,   Prior.kappa,
                       Post.nu[k], Post.beta[k], Post.m[k], Post.kappa[k],
                       )
      if not doFast:
        aDiff = SS.N[k] + Prior.nu - Post.nu[k]
        bDiff = SS.xx[k] + Prior.beta \
                          + Prior.kappa * np.square(Prior.m) \
                          - Post.beta[k] \
                          - Post.kappa[k] * np.square(Post.m[k])
        cDiff = SS.x[k] + Prior.kappa * Prior.m \
                        - Post.kappa[k] * Post.m[k]
        dDiff = SS.N[k] + Prior.kappa - Post.kappa[k]
        elbo[k] += 0.5 * aDiff * np.sum(self._E_logL(k)) \
                 - 0.5 * np.inner(bDiff, self._E_L(k)) \
                 + np.inner(cDiff, self.GetCached('E_Lmu', k)) \
                 - 0.5 * dDiff * np.sum(self.GetCached('E_muLmu', k))
    return elbo.sum() - 0.5 * np.sum(SS.N) * SS.D * LOGTWOPI

  # TODO: Merge ELBO

  ########################################################### Post
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
               logM = log p( data assigned to comp kA ) [up to constant]
    '''
    nu, beta, m, kappa = self.calcPostParamsForComp(SS, kA, kB)
    return -1 * c_Func(nu, beta, m, kappa)

  def calcPostParams(self, SS):
    ''' Calc posterior (nu, beta, m, kappa) for all comps given suff stats
    '''
    Prior = self.Prior
    nu = Prior.nu + SS.N
    kappa = Prior.kappa + SS.N
    m = (Prior.kappa * Prior.m + SS.x) / kappa[:,np.newaxis]
    beta = Prior.beta + SS.xx \
           + Prior.kappa * np.square(Prior.m) \
           - kappa[:,np.newaxis] * np.square(m)
    return nu, beta, m, kappa

  def calcPostParamsForComp(self, SS, kA, kB=None):
    if kB is None:
      SN = SS.N[kA]
      Sx = SS.x[kA]
      Sxx = SS.xx[kA]
    else:
      SN = SS.N[kA] + SS.N[kB]
      Sx = SS.x[kA] + SS.x[kB]
      Sxx = SS.xx[kA] + SS.xx[kB]
    Prior = self.Prior
    nu = Prior.nu + SN
    kappa = Prior.kappa + SN
    m = (Prior.kappa * Prior.m + Sx) / kappa
    beta = Prior.beta + Sxx \
             + Prior.kappa * np.square(Prior.m) \
             - kappa * np.square(m)
    return nu, beta, m, kappa

  ########################################################### Gibbs
  ########################################################### 
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

  def calcPredProbVec_Unnorm(self, SS, x):
    ''' Calculate K-vector of positive entries \propto p( x | SS[k] )
    '''
    return self._calcPredProbVec_Fast(SS, x)
  
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

  def _calcPredProbVec_Naive(self, SS, x):
    nu, beta, m, kappa = self.calcPostParams(SS)
    pSS = SS.copy()
    pSS.N += 1
    pSS.x += x
    pSS.xx += np.square(x)
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
    logp = (-0.5 * (nu+1))[:,np.newaxis] * np.log(base)
    logp += (gammaln(0.5 * (nu+1)) - gammaln(0.5 * nu))[:,np.newaxis]
    p = logp
    np.exp(logp, out=p)
    p /= np.sqrt(kbeta)
    return np.prod(p, axis=1)

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

  def incrementSS(self, SS, k, x):
    SS.x[k] += x
    SS.xx[k] += np.square(x)

  def decrementSS(self, SS, k, x):
    SS.x[k] -= x
    SS.xx[k] -= np.square(x)

  def incrementPost(self, k, x):
    ''' Add data to the Post ParamBag, component k
    '''
    pass

  def decrementPost(self, k, x):
    ''' Remove data from the Post ParamBag, component k
    '''
    Post = self.Post
    Post.nu -= 1
    kappa = Post.kappa[k] - 1
    Post.beta -= Post.kappa/kappa * np.square(x-Post.m)
    Post.m = 1/(kappa) * (Post.kappa * Post.m - x)
    Post.kappa = kappa

  def updateCandidatePost_inplace(self, x):
    if not hasattr(self, 'CandidatePost'):
      self.CandidatePost = Post.copy()
    else:
      self.CandidatePost.nu[:] = Post.nu
      self.CandidatePost.beta[:] = Post.beta
      self.CandidatePost.m[:] = Post.m
      self.CandidatePost.kappa[:] = Post.kappa

    CPost = self.CandidatePost
    CPost.nu += 1
    CPost.kappa += 1
    CPost.beta += Post.kappa/(CPost.kappa+1) * np.square(x-CPost.m)
    Post.m[k] += 1/(kappa) * (Post.kappa[k] * Post.m[k] - x)
  
  ########################################################### Expectations
  ########################################################### 
    
  def _E_CovMat(self, k=None):
    '''
        Returns
        --------
        E[ Sigma ] : 2D array, size DxD
    '''
    return np.diag(self._E_Cov(k))

  def _E_Cov(self, k=None):
    '''
        Returns
        --------
        E[ sigma^2 ] : 1D array, size D
    '''
    if k is None:
      nu = self.Prior.nu
      beta = self.Prior.beta
    else:
      nu = self.Post.nu[k]
      beta = self.Post.beta[k]
    return beta / (nu - 2)

  def _E_logL(self, k=None):
    ''' 
        Returns
        -------
        E_logLam : 1D array, size D
    '''
    if k is None:
      nu = self.Prior.nu
      beta = self.Prior.beta
    else:
      nu = self.Post.nu[k]
      beta = self.Post.beta[k]
    return LOGTWO - np.log(beta) + digamma(0.5*nu)

  def _E_L(self, k=None):
    ''' 
        Returns
        --------
        EL : 1D array, size D
    '''
    if k is None:
      nu = self.Prior.nu
      beta = self.Prior.beta
    else:
      nu = self.Post.nu[k]
      beta = self.Post.beta[k]
    return nu / beta
    
  def _E_Lmu(self, k=None):
    '''
        Returns
        --------
        ELmu : 1D array, size D
    '''
    if k is None:
      nu = self.Prior.nu
      beta = self.Prior.beta
      m = self.Prior.m
    else:
      nu = self.Post.nu[k]
      beta = self.Post.beta[k]
      m = self.Post.m[k]
    return (nu / beta) * m

  def _E_muLmu(self, k=None):
    ''' Calc expectation E[lam * mu^2], yielding vector with one entry per dim

        Returns
        --------
        EmuLmu : 1D array, size D
    '''
    if k is None:
      nu = self.Prior.nu
      kappa = self.Prior.kappa
      m = self.Prior.m
      beta = self.Prior.beta
    else:
      nu = self.Post.nu[k]
      kappa = self.Post.kappa[k]
      m = self.Post.m[k]
      beta = self.Post.beta[k]
    return 1.0 / kappa + (nu / beta) * (m*m)




def MAPEstParams_inplace(nu, beta, m, kappa=0):
  ''' MAP estimate parameters mu, Sigma given Normal-Wishart hyperparameters
  '''
  D = m.size
  mu = m
  sigma = beta / (nu[:,np.newaxis]+2)
  return mu, sigma

def c_Func(nu, beta, m, kappa):
  D = m.size
  c1D = - 0.5 * LOGTWOPI \
         - 0.5 * LOGTWO * nu \
         - gammaln( 0.5 * nu ) \
         + 0.5 * np.log(kappa) \
         + 0.5 * nu * np.log(beta)
  return np.sum(c1D)

def c_Diff(nu1, beta1, m1, kappa1,
           nu2, beta2, m2, kappa2):
  ''' Evaluate difference of cumulant functions c(params1) - c(params2)
      Returns
      -------
      diff : scalar real
  '''
  cDiff = - 0.5 * LOGTWO * (nu1 - nu2) \
          - gammaln(0.5 * nu1) + gammaln(0.5 * nu2) \
          + 0.5 * (np.log(kappa1) - np.log(kappa2)) \
          + 0.5 * (nu1 * np.log(beta1) - nu2 * np.log(beta2))
  return np.sum(cDiff)

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
