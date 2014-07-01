'''
GaussObsModel

Prior : Dirichlet
* alpha

EstParams 
for k in 1 2, ... K:
* pi[k]

Posterior : Normal-Wishart
---------
for k = 1, 2, ... K:
  * Post.alpha[k]

'''
import numpy as np
from scipy.special import gammaln, digamma

from bnpy.suffstats import ParamBag, SuffStatBag
from bnpy.util import LOGTWO, LOGPI, LOGTWOPI, EPS
from bnpy.util import dotATA, dotATB, dotABT

from AbstractObsModel import AbstractObsModel 

class MultObsModel(AbstractObsModel):

  def __init__(self, inferType='EM', D=0, 
                     Data=None, **PriorArgs):
    ''' Initialize bare Mult obsmodel with Dirichlet prior. 
        Resulting object lacks either EstParams or Post, 
          which must be created separately.
    '''
    if Data is not None:
      self.D = Data.vocab_size
    else:
      self.D = D
    self.K = 0
    self.inferType = inferType
    self.createPrior(Data, **PriorArgs)
    self.Cache = dict()

  def createPrior(self, Data, alpha=1.0):
    ''' Initialize Prior ParamBag object, with fields nu, B, m, kappa
          set according to match desired mean and expected covariance matrix.
    '''
    D = self.D
    self.Prior = ParamBag(K=0, D=D)
    if type(alpha) == float or alpha.ndim == 0:
      alpha = alpha * np.ones(D)
    self.Prior.setField('alpha', alpha, dims=('D'))

  def setupWithAllocModel(self, allocModel):
    ''' Using the allocation model, determine the modeling scenario:
          doc  : multinomial : each atom is D-vector of integer counts
          word : categorical : each atom is a single one-of-D indicator 
    '''
    if type(allocModel) != str:
      allocModel = str(type(allocModel))
    if allocModel.lower().count('hdp') or allocModel.lower().count('admix'):
      self.DataAtomType = 'word'
    else:
      self.DataAtomType = 'doc'

  ######################################################### Set EstParams
  #########################################################
  def setEstParams(self, obsModel=None, SS=None, LP=None, Data=None,
                          pi=None,
                          **kwargs):
    ''' Create EstParams ParamBag with fields pi
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
      self.EstParams = ParamBag(K=pi.shape[0], D=pi.shape[1])
      self.EstParams.setField('pi', pi, dims=('K', 'D'))

  def setEstParamsFromPost(self, Post=None):
    ''' Convert from Post (alpha) to EstParams (pi),
         each EstParam is set to its posterior mean.
    '''
    if Post is None:
      Post = self.Post
    self.EstParams = ParamBag(K=Post.K, D=Post.D)
    pi = Post.alpha / np.sum(Post.alpha, axis=1)[:, np.newaxis]
    self.EstParams.setField('pi', pi, dims=('K','D'))
    
  
  ######################################################### Set Post
  #########################################################
  def setPostFactors(self, obsModel=None, SS=None, LP=None, Data=None,
                           alpha=None,
                            **kwargs):
    ''' Create Post ParamBag with fields (alpha)
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
      self.Post.setField('alpha', alpha, dims=('K','D'))

  def setPostFromEstParams(self, EstParams, Data=None, wc=None):
    ''' Convert from EstParams (mu, Sigma) to Post (nu, B, m, kappa),
          each posterior hyperparam is set so EstParam is the posterior mean
    '''
    K = EstParams.K
    D = EstParams.D
    if Data is not None:
      wc = Data.word_count.sum()

    alpha = wc * EstParams.pi
    self.Post = ParamBag(K=K, D=D)
    self.Post.setField('alpha', alpha, dims=('K', 'D'))

  ########################################################### Summary
  ########################################################### 

  def calcSummaryStats(self, Data, SS, LP):
    if SS is None:
      SS = SuffStatBag(K=LP['resp'].shape[1], D=Data.vocab_size)

    if self.DataAtomType == 'doc':
      DocWordMat = Data.to_sparse_docword_matrix() # D x V
      TopicWordCounts = LP['resp'].T * DocWordMat # mat-mat product

      logh = self.logh(Data)
      SS.setField('logh', logh, dims=None)
    else:
      wv = LP['resp']  # N x K
      WMat = Data.to_sparse_matrix() # V x N
      TopicWordCounts = (WMat * wv).T # matrix-matrix product

    SS.setField('WordCounts', TopicWordCounts, dims=('K','D'))
    SS.setField('N', np.sum(TopicWordCounts,axis=1), dims=('K'))

    return SS

  ########################################################### EM
  ########################################################### 
  # _________________________________________________________ E step
  def calcSoftEvMatrix_FromEstParams(self, Data):
    logpi = np.log(self.EstParams.pi)
    if self.DataAtomType == 'doc':
      WMat = Data.to_sparse_wordcount_matrix()
      return np.dot(WMat, logpi.T)
    else:
      return logpi[Data.word_id, :]
  # _________________________________________________________  M step
  def updateEstParams_MaxLik(self, SS):
    self.ClearCache()
    if not hasattr(self, 'EstParams') or self.EstParams.K != SS.K:
      self.EstParams = ParamBag(K=SS.K, D=SS.D)
    pi = SS.WordCounts / SS.WordCounts.sum(axis=1)[:,np.newaxis]
    self.EstParams.setField('pi', pi, dims=('K', 'D'))

  def updateEstParams_MAP(self, SS):
    self.ClearCache()
    if not hasattr(self, 'EstParams') or self.EstParams.K != SS.K:
      self.EstParams = ParamBag(K=SS.K, D=SS.D)
    pi = SS.WordCounts + self.Prior.alpha - 1
    pi /= pi.sum(axis=1)[:,np.newaxis]
    self.EstParams.setField('pi', pi, dims=('K', 'D'))

  ########################################################### VB
  ########################################################### 

  def calcSoftEvMatrix_FromPost(self, Data):
    ''' Calculate soft ev matrix 

        Returns
        ------
        L : 2D array, size nAtom x K
    '''
    Elogpi = self.GetCached('E_logpi', 'all') # K x V
    if self.DataAtomType == 'doc':
      WMat = Data.to_sparse_docword_matrix() # nDoc x V
      return WMat * Elogpi.T
    else:
      return Elogpi.T[Data.word_id, :]

  def updatePost(self, SS):
    ''' Update the Post ParamBag, so each component 1, 2, ... K
          contains Dirichlet posterior params given Prior and SS
    '''
    self.ClearCache()
    if not hasattr(self, 'Post') or self.Post.K != SS.K:
      self.Post = ParamBag(K=SS.K, D=SS.D)

    alpha = self.Prior.alpha + SS.WordCounts
    self.Post.setField('alpha', alpha, dims=('K', 'D'))

  def calcELBO_Memoized(self, SS, doFast=False):
    ''' Calculate obsModel's ELBO using sufficient statistics SS and Post.

        Args
        -------
        SS : bnpy SuffStatBag, contains fields for WordCounts
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
      elbo[k] = c_Diff(Prior.alpha, Post.alpha[k])
      if not doFast and SS.N[k] > 1e-9:
        pass
    if self.DataAtomType == 'doc':
      return SS.logh + np.sum(elbo)
    else:
      return np.sum(elbo)

  def logh(self, Data):
    ''' Calculate reference measure for the multinomial distribution

        Returns
        -------
        logh : scalar float, log h(Data) = \sum_{n=1}^N log [ C!/prod_d C_d!] 
    '''
    WMat = Data.to_sparse_docword_matrix().toarray()
    sumWMat = np.sum(WMat, axis=1)
    return np.sum(gammaln(sumWMat+1)) - np.sum(gammaln(WMat+1)) 

  ########################################################### Gibbs
  ########################################################### 
  def calcMargLik(self):
    pass
  
  def calcPredLik(self, xSS):
    pass

  def incrementPost(self, k, cvec):
    ''' Add data to the Post ParamBag, component k
    '''
    self.Post.alpha[k] += cvec

  def decrementPost(self, k, x):
    ''' Remove data from the Post ParamBag, component k
    '''
    self.Post.alpha[k] -= cvec

  ########################################################### Expectations
  ########################################################### 
  def _E_logpi(self, k=None):
    if k is None:
      alpha = self.Prior.alpha
      Elogpi = digamma(alpha) - digamma(np.sum(alpha))
    elif k == 'all':
      AMat = self.Post.alpha
      Elogpi = digamma(AMat) - digamma(np.sum(AMat,axis=1))[:,np.newaxis]
    else:
      Elogpi = digamma(self.Post.alpha[k]) - digamma(self.Post.alpha[k].sum())
    return Elogpi

def c_Func(alpha):
  return gammaln(np.sum(alpha)) - np.sum(gammaln(alpha))

def c_Diff(alpha1, alpha2):
  return c_Func(alpha1) - c_Func(alpha2)
