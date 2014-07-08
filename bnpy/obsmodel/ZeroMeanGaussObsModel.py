'''
ZeroMeanGaussObsModel

Prior : Wishart
* nu
* B

EstParams 
* Sigma

Posterior : Wishart
* nu[k]
* B[k]
'''

import numpy as np
import scipy.linalg
from scipy.special import gammaln, digamma

from bnpy.suffstats import ParamBag, SuffStatBag
from bnpy.util import LOGTWO, LOGPI, LOGTWOPI, EPS
from bnpy.util import dotATA, dotATB, dotABT
from bnpy.util import as1D, as2D, as3D

from AbstractObsModel import AbstractObsModel 

class ZeroMeanGaussObsModel(AbstractObsModel):

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
    self.Prior = ParamBag(K=0, D=D)
    self.Prior.setField('nu', nu, dims=None)
    self.Prior.setField('B', B, dims=('D','D'))

  def get_mean_for_comp(self, k):
    return np.zeros(self.D)

  def get_covar_mat_for_comp(self, k):
    if hasattr(self, 'EstParams'):
      return self.EstParams.Sigma[k]
    else:
      return self._E_CovMat(k)
    

  ######################################################### I/O Utils
  #########################################################   for humans
  def get_name(self):
    return 'ZeroMeanGauss'

  def get_info_string(self):
    return 'Gaussian with fixed zero means, full covariance.'
  
  def get_info_string_prior(self):
    msg = 'Wishart on prec matrix Lam\n'
    if self.D > 2:
      sfx = ' ...'
    else:
      sfx = ''
    S = self._E_CovMat()[:2,:2]
    msg += 'E[ CovMat[k] ] = \n'
    msg += str(S) + sfx
    msg = msg.replace('\n', '\n  ')
    return msg

  ######################################################### Set EstParams
  #########################################################
  def setEstParams(self, obsModel=None, SS=None, LP=None, Data=None,
                         Sigma=None,
                         **kwargs):
    ''' Create EstParams ParamBag with fields Sigma
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
      K = Sigma.shape[0]
      self.EstParams = ParamBag(K=K, D=self.D)
      self.EstParams.setField('Sigma', Sigma, dims=('K', 'D', 'D'))

  def setEstParamsFromPost(self, Post):
    ''' Convert from Post (nu, B) to EstParams (Sigma),
         each EstParam is set to its posterior mean.
    '''
    self.EstParams = ParamBag(K=K, D=D)    
    Sigma = Post.B / (nu[k] - D - 1)
    self.EstParams.setField('Sigma', Sigma, dims=('K','D','D'))

  
  ######################################################### Set Post
  #########################################################
  def setPostFactors(self, obsModel=None, SS=None, LP=None, Data=None,
                            nu=0, B=0,
                            **kwargs):
    ''' Create Post ParamBag with fields nu, B
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
      K = B.shape[0]
      self.Post = ParamBag(K=K, D=self.D)
      self.Post.setField('nu', as1D(nu), dims=('K'))
      self.Post.setField('B', B, dims=('K', 'D', 'D'))

  def setPostFromEstParams(self, EstParams, Data=None, N=None):
    ''' Convert from EstParams (Sigma) to Post distribution (nu, B),
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
    self.Post = ParamBag(K=K, D=D)
    self.Post.setField('nu', nu, dims=('K'))
    self.Post.setField('B', B, dims=('K', 'D', 'D'))

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

    # Expected outer-product for each k 
    sqrtResp = np.sqrt(resp)
    xxT = np.zeros( (K, self.D, self.D) )
    for k in xrange(K):
      xxT[k] = dotATA(sqrtResp[:,k][:,np.newaxis]*Data.X )
    SS.setField('xxT', xxT, dims=('K','D','D'))
    return SS 

  ########################################################### EM
  ########################################################### 
  # _________________________________________________________ E step
  def calcLogSoftEvMatrix_FromEstParams(self, Data):
    K = self.EstParams.K
    L = np.empty((Data.nObs, K))
    for k in xrange(K):
      L[:,k] = - 0.5 * self.D * LOGTWOPI \
               - 0.5 * self._logdetSigma(k)  \
               - 0.5 * self._mahalDist_EstParam(Data.X, k)
    return L

  def _mahalDist_EstParam(self, X, k):
    Q = np.linalg.solve(self.GetCached('cholSigma', k), \
                        X.T)
    Q *= Q
    return np.sum(Q, axis=0)

  def _cholSigma(self, k):
    return scipy.linalg.cholesky(self.EstParams.Sigma[k], lower=1)    

  def _logdetSigma(self, k):
    return 2 * np.sum(np.log(np.diag(self.GetCached('cholSigma', k))))

  # _________________________________________________________  M step
  def updateEstParams_MaxLik(self, SS):
    self.ClearCache()
    if not hasattr(self, 'EstParams') or self.EstParams.K != SS.K:
      self.EstParams = ParamBag(K=SS.K, D=SS.D)

    minCovMat = self.min_covar * np.eye(SS.D)
    covMat = np.tile(minCovMat, (SS.K,1,1))
    for k in xrange(SS.K):
      covMat[k] += SS.xxT[k] / SS.N[k]
    self.EstParams.setField('Sigma', covMat, dims=('K', 'D', 'D'))

  def updateEstParams_MAP(self, SS):
    self.ClearCache()
    if not hasattr(self, 'EstParams') or self.EstParams.K != SS.K:
      self.EstParams = ParamBag(K=SS.K, D=SS.D)
    Prior = self.Prior
    nu = Prior.nu + SS.N
    B = np.empty((SS.K, SS.D, SS.D))
    for k in xrange(SS.K):
      B[k] = Prior.B + SS.xxT[k]
  
    Sigma = MAPEstParams_inplace(nu, B)   
    self.EstParams.setField('Sigma', Sigma, dims=('K', 'D', 'D'))

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
                        X.T)
    Q *= Q
    return self.Post.nu[k] * np.sum(Q, axis=0)

  def updatePost(self, SS):
    ''' Update the Post ParamBag, so each component 1, 2, ... K
          contains Wishart posterior params given Prior and SS
    '''
    self.ClearCache()
    if not hasattr(self, 'Post') or self.Post.K != SS.K:
      self.Post = ParamBag(K=SS.K, D=SS.D)

    Prior = self.Prior # use 'Prior' not 'self.Prior', improves readability
    Post = self.Post

    Post.setField('nu', Prior.nu + SS.N, dims=('K'))
    B = np.empty((SS.K, SS.D, SS.D))
    for k in xrange(SS.K):
      B[k] = Prior.B + SS.xxT[k]
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
                        self.GetCached('logdetB'), self.D,
                        Post.nu[k],
                        self.GetCached('logdetB', k),
                        )
      if not doFast:
        aDiff = SS.N[k] + Prior.nu - Post.nu[k]
        bDiff = SS.xxT[k] + Prior.B \
                          - Post.B[k]
        elbo[k] += 0.5 * aDiff * self.GetCached('E_logdetL', k) \
                 - 0.5 * self._trace__E_L(bDiff, k)
    return elbo.sum() - 0.5 * np.sum(SS.N) * SS.D * LOGTWOPI


  ########################################################### Gibbs
  ########################################################### 
  def calcMargLik(self):
    pass
  
  def calcPredLik(self, xSS):
    pass

  def incrementPost(self, k, x):
    ''' Add data to the Post ParamBag, component k
    '''
    # TODO: update cached cholesky and log det with rank-one updates
    pass

  def decrementPost(self, k, x):
    ''' Remove data from the Post ParamBag, component k
    '''
    # TODO: update cached cholesky and log det with rank-one updates
    pass

  ########################################################### Expectations
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

def MAPEstParams_inplace(nu, B):
  ''' MAP estimate parameters mu, Sigma given Normal-Wishart hyperparameters
  '''
  D = B.shape[-1]
  Sigma = B
  for k in xrange(B.shape[0]):
    Sigma[k] /= (nu[k] + D + 1)
  return Sigma

def c_Func(nu, logdetB, D):
  if logdetB.ndim >= 2:
    logdetB = np.log(np.linalg.det(logdetB))
  dvec = np.arange(1, D+1, dtype=np.float)
  return - 0.5 * D * LOGTWO * nu \
         - np.sum( gammaln( 0.5 * (nu + 1 - dvec) )) \
         + 0.5 * nu * logdetB

def c_Diff(nu1, logdetB1, D, nu2, logdetB2):
  dvec = np.arange(1, D+1, dtype=np.float)
  return - 0.5 * D * LOGTWO * (nu1 - nu2) \
         - np.sum( gammaln( 0.5 * (nu1 + 1 - dvec) )) \
         + np.sum( gammaln( 0.5 * (nu2 + 1 - dvec) )) \
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
