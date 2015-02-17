'''
BernObsModel

Prior : Dirichlet
* lam1
* lam0

EstParams
-------- 
for k in 1 2, ... K:
  * EstParams.phi[k]

Post
--------
for k = 1, 2, ... K:
  * Post.lam1[k]
  * Post.lam0[k]

'''
import numpy as np
from scipy.special import gammaln, digamma

from bnpy.suffstats import ParamBag, SuffStatBag
from bnpy.util import dotATA, dotATB, dotABT
from bnpy.util import as1D, as2D, as3D

from AbstractObsModel import AbstractObsModel 

class BernObsModel(AbstractObsModel):

  def __init__(self, inferType='EM', D=0,
                     Data=None, **PriorArgs):
    ''' Initialize bare Mult obsmodel with Dirichlet prior. 
        Resulting object lacks either EstParams or Post, 
          which must be created separately.
    '''
    if Data is not None:
      self.D = Data.dim
    elif D > 0:
      self.D = int(D)
    self.K = 0
    self.inferType = inferType
    self.createPrior(Data, **PriorArgs)
    self.Cache = dict()

  def createPrior(self, Data, lam1=1.0, lam0=1.0, eps_phi=1e-14, **kwargs):
    ''' Initialize Prior ParamBag object, with field 'lam'
    '''
    D = self.D
    self.eps_phi = eps_phi
    self.Prior = ParamBag(K=0, D=D)
    lam1 = np.asarray(lam1, dtype=np.float)
    lam0 = np.asarray(lam0, dtype=np.float)
    if lam1.ndim == 0:
      lam1 = lam1 * np.ones(D)
    if lam0.ndim == 0:
      lam0 = lam0 * np.ones(D)
    assert lam1.size == D
    assert lam0.size == D
    self.Prior.setField('lam1', lam1, dims=('D'))
    self.Prior.setField('lam0', lam0, dims=('D'))

  def setupWithAllocModel(self, allocModel):
    ''' Using the allocation model, determine the modeling scenario:
          doc  : multinomial : each atom is D-vector of integer counts
          word : categorical : each atom is a single one-of-D indicator 
    '''
    pass    



  ######################################################### I/O Utils
  #########################################################   for humans
  def get_name(self):
    return 'Bern'

  def get_info_string(self):
    return 'Bernoulli over %d binary attributes.' % (self.D)
  
  def get_info_string_prior(self):
    msg = 'Beta over %d attributes.\n' % (self.D)
    if self.D > 2:
      sfx = ' ...'
    else:
      sfx = ''
    msg += 'lam1 = %s%s\n' % (str(self.Prior.lam1[:2]), sfx)
    msg += 'lam0 = %s%s\n' % (str(self.Prior.lam0[:2]), sfx)
    msg = msg.replace('\n', '\n  ')
    return msg

  ######################################################### Set EstParams
  #########################################################
  def setEstParams(self, obsModel=None, SS=None, LP=None, Data=None,
                          phi=None,
                          **kwargs):
    ''' Create EstParams ParamBag with fields phi
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
      self.EstParams = ParamBag(K=phi.shape[0], D=phi.shape[1])
      self.EstParams.setField('phi', phi, dims=('K', 'D'))
    self.K = self.EstParams.K

  def setEstParamsFromPost(self, Post=None):
    ''' Convert from Post (lam) to EstParams (phi),
         each EstParam is set to its posterior mean.
    '''
    if Post is None:
      Post = self.Post
    self.EstParams = ParamBag(K=Post.K, D=Post.D)
    phi = Post.lam1 / (Post.lam1 + Post.lam0)
    self.EstParams.setField('phi', phi, dims=('K','D'))
    self.K = self.EstParams.K

  
  ######################################################### Set Post
  #########################################################
  def setPostFactors(self, obsModel=None, SS=None, LP=None, Data=None,
                           lam1=None, lam0=None, **kwargs):
    ''' Create Post ParamBag with fields (lam)
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
      lam1 = as2D(lam1)
      lam0 = as2D(lam0)
      K, D = lam1.shape
      self.Post = ParamBag(K=K, D=D)
      self.Post.setField('lam1', lam1, dims=('K','D'))
      self.Post.setField('lam0', lam0, dims=('K','D'))
    self.K = self.Post.K


  def setPostFromEstParams(self, EstParams, Data=None, nTotalTokens=1,
                                                       **kwargs):
    ''' Convert from EstParams (mu, Sigma) to Post (nu, B, m, kappa),
          each posterior hyperparam is set so EstParam is the posterior mean
    '''
    K = EstParams.K
    D = EstParams.D

    WordCounts = EstParams.phi * nTotalTokens
    lam1 = WordCounts + self.Prior.lam1
    lam0 = (1-WordCounts) + self.Prior.lam0

    self.Post = ParamBag(K=K, D=D)
    self.Post.setField('lam1', lam1, dims=('K', 'D'))
    self.Post.setField('lam0', lam0, dims=('K', 'D'))
    self.K = K

  ########################################################### Summary
  ########################################################### 

  def calcSummaryStats(self, Data, SS, LP):
    ''' Calculate summary statistics for given dataset and local parameters

        Returns
        --------
        SS : SuffStatBag object, with K components
             if DataAtomType == 'doc', 
    '''
    if SS is None:
      SS = SuffStatBag(K=LP['resp'].shape[1], D=Data.dim)

    Resp = LP['resp']  # 2D array, N x K 
    X = Data.X # 2D array, N x D
    CountON = np.dot(Resp.T, X) # matrix-matrix product, result is K x D
    CountOFF = np.dot(Resp.T, 1-X)

    SS.setField('Count1', CountON, dims=('K','D'))
    SS.setField('Count0', CountOFF, dims=('K','D'))
    return SS

  def forceSSInBounds(self, SS):
    ''' Force count vectors to remain positive

        This avoids numerical problems due to incremental add/subtract ops
        which can cause computations like 
            x = 10.
            x += 1e-15
            x -= 10
            x -= 1e-15
        to be slightly different than zero instead of exactly zero.

        Returns
        -------
        None. SS updated in-place.
    '''
    np.maximum(SS.Count1, 0, out=SS.Count1)
    np.maximum(SS.Count0, 0, out=SS.Count0)


  def incrementSS(self, SS, k, Data, docID):
    raise NotImplementedError('TODO')

  def decrementSS(self, SS, k, Data, docID):
    raise NotImplementedError('TODO')

  ########################################################### EM E step
  ###########################################################
  def calcLogSoftEvMatrix_FromEstParams(self, Data):
    ''' Calculate log soft evidence matrix for given Dataset under EstParams

        Returns
        ---------
        L : 2D array, N x K
    '''
    logphiT = np.log(self.EstParams.phi.T) # D x K matrix
    log1mphiT = np.log(1.0 - self.EstParams.phi.T) # D x K matrix
    return np.dot(Data.X, logphiT) + np.dot(1 - Data.X, log1mphiT)

  ########################################################### EM M step
  ###########################################################
  def updateEstParams_MaxLik(self, SS):
    ''' Update EstParams for all comps via maximum likelihood given suff stats

        Returns
        ---------
        None. Fields K and EstParams updated in-place.
    '''
    self.ClearCache()
    self.K = SS.K
    if not hasattr(self, 'EstParams') or self.EstParams.K != SS.K:
      self.EstParams = ParamBag(K=SS.K, D=SS.D)
    phi = SS.Count1 / (SS.Count1 + SS.Count0)
    ## prevent entries from reaching exactly 0
    np.maximum(phi, self.eps_phi, out=phi) 
    np.minimum(phi, 1.0 - self.eps_phi, out=phi) 
    self.EstParams.setField('phi', phi, dims=('K', 'D'))

  def updateEstParams_MAP(self, SS):
    ''' Update EstParams for all comps via MAP estimation given suff stats

        Returns
        ---------
        None. Fields K and EstParams updated in-place.
    '''
    self.ClearCache()
    if not hasattr(self, 'EstParams') or self.EstParams.K != SS.K:
      self.EstParams = ParamBag(K=SS.K, D=SS.D)
    phi_numer = SS.Count1 + self.Prior.lam1 - 1
    phi_denom = SS.Count1 + SS.Count0 + self.Prior.lam1 + self.Prior.lam0 - 2
    phi = phi_numer / phi_denom
    self.EstParams.setField('phi', phi, dims=('K', 'D'))


  ########################################################### Post updates
  ########################################################### 
  def updatePost(self, SS):
    ''' Update (in place) posterior params for all comps given suff stats

        Afterwards, self.Post contains Dirichlet posterior params
        updated given self.Prior and provided suff stats SS

        Returns
        ---------
        None. Fields K and Post updated in-place.
    '''
    self.ClearCache()
    if not hasattr(self, 'Post') or self.Post.K != SS.K:
      self.Post = ParamBag(K=SS.K, D=SS.D)

    lam1, lam0 = self.calcPostParams(SS)
    self.Post.setField('lam1', lam1, dims=('K', 'D'))
    self.Post.setField('lam0', lam0, dims=('K', 'D'))
    self.K = SS.K

  def calcPostParams(self, SS):
    ''' Calc updated params (lam) for all comps given suff stats

        Returns
        --------
        lam1 : 2D array, K x D
        lam0 : 2D array, K x D
    '''
    lam1 = SS.Count1 + self.Prior.lam1[np.newaxis,:]
    lam0 = SS.Count0 + self.Prior.lam0[np.newaxis,:]
    return lam1, lam0

  def calcPostParamsForComp(self, SS, kA=None, kB=None):
    ''' Calc params (lam) for specific comp, given suff stats

        These params define the common-form of the exponential family 
        Dirichlet posterior distribution over parameter vector phi

        Returns
        --------
        lam : 1D array, size D
    '''
    if kB is None:
      lam1_k = SS.Count1[kA].copy()
      lam0_k = SS.Count0[kA].copy()
    else:
      lam1_k = SS.Count1[kA] + SS.Count1[kB]
      lam0_k = SS.Count0[kA] + SS.Count0[kB]
    lam1_k += self.Prior.lam1
    lam0_k += self.Prior.lam0
    return lam1_k, lam0_k


  ########################################################### Stochastic Post
  ########################################################### update
  def updatePost_stochastic(self, SS, rho):
    ''' Stochastic update (in place) posterior for all comps given suff stats.

        Dirichlet common params used here, no need for natural form.
    '''
    assert hasattr(self, 'Post')
    assert self.Post.K == SS.K
    self.ClearCache()
    
    lam1, lam0 = self.calcPostParams(SS)
    Post = self.Post
    Post.lam1[:] = (1-rho) * Post.lam1 + rho * lam1
    Post.lam0[:] = (1-rho) * Post.lam0 + rho * lam0

  def convertPostToNatural(self):
    ''' Convert (in-place) current posterior params from common to natural form

        Beta common form is equivalent to the natural form for our purposes
    '''
    pass
    
  def convertPostToCommon(self):
    ''' Convert (in-place) current posterior params from natural to common form

        Beta common form is equivalent to the natural form for our purposes
    '''
    pass


  ########################################################### VB
  ########################################################### 
  def calcLogSoftEvMatrix_FromPost(self, Data):
    ''' Calculate expected log soft ev matrix for given dataset under posterior

        Returns
        ------
        L : 2D array, size nAtom x K
    '''
    ElogphiT = self.GetCached('E_logphiT', 'all') # D x K
    Elog1mphiT = self.GetCached('E_log1mphiT', 'all') # D x K

    # Matrix-matrix product, result is N x K
    L = np.dot(Data.X, ElogphiT) + np.dot(1.0-Data.X, Elog1mphiT)
    return L


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
    L_perComp = np.zeros(SS.K)
    Post = self.Post
    Prior = self.Prior
    if not afterMStep:
      ElogphiT = self.GetCached('E_logphiT', 'all') # D x K
      Elog1mphiT = self.GetCached('E_log1mphiT', 'all') # D x K

    for k in xrange(SS.K):
      L_perComp[k] = c_Diff(Prior.lam1, Prior.lam0,
                            Post.lam1[k], Post.lam0[k])
      if not afterMStep:
        L_perComp[k] += np.inner(SS.Count1[k] + Prior.lam1 - Post.lam1[k],
                                 ElogphiT[:, k])
        L_perComp[k] += np.inner(SS.Count0[k] + Prior.lam0 - Post.lam0[k],
                                 Elog1mphiT[:, k])
    return np.sum(L_perComp)


  def getDatasetScale(self, SS, extraSS=None):
    ''' Get scale factor for dataset, indicating number of observed scalars. 

        Used for normalizing the ELBO so it has reasonable range.

        Returns
        ---------
        s : scalar positive integer
            total number of word tokens observed in the sufficient stats
    '''
    s = SS.Count1.sum() + SS.Count0.sum()
    if extraSS is None:
      return s

    else:
      sextra = extraSS.Count1.sum() + extraSS.Count0.sum()
      return s - sextra

  ######################################################### Hard Merge
  #########################################################
  def calcHardMergeGap(self, SS, kA, kB):
    ''' Calculate change in ELBO after a hard merge applied to this model

        Returns
        ---------
        gap : scalar real, indicates change in ELBO after merge of kA, kB
    '''
    Prior = self.Prior
    cPrior = c_Func(Prior.lam1, Prior.lam0)

    Post = self.Post
    cA = c_Func(Post.lam1[kA], Post.lam0[kA])
    cB = c_Func(Post.lam1[kB], Post.lam0[kB])

    lam1, lam0 = self.calcPostParamsForComp(SS, kA, kB)
    cAB = c_Func(lam1, lam0)
    return cA + cB - cPrior - cAB


  def calcHardMergeGap_AllPairs(self, SS):
    ''' Calculate change in ELBO for all possible candidate hard merge pairs 

        Returns
        ---------
        Gap : 2D array, size K x K, upper-triangular entries non-zero
              Gap[j,k] : scalar change in ELBO after merge of k into j
    '''
    Prior = self.Prior
    cPrior = c_Func(Prior.lam1, Prior.lam0)

    Post = self.Post
    c = np.zeros(SS.K)
    for k in xrange(SS.K):
      c[k] = c_Func(Post.lam1[k], Post.lam0[k])

    Gap = np.zeros((SS.K, SS.K))
    for j in xrange(SS.K):
      for k in xrange(j+1, SS.K):
        cjk = c_Func(*self.calcPostParamsForComp(SS, j, k))
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
    return -1 * c_Func(*self.calcPostParamsForComp(SS, kA, kB))

  def calcMargLik(self, SS):
    ''' Calc log marginal likelihood combining all comps, given suff stats

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
      lam1, lam0 = self.calcPostParamsForComp(SS, k)
      logp[k] = c_Diff(Prior.lam1, Prior.lam0,
                       lam1, lam0)
    return np.sum(logp)

  ########################################################### Expectations
  ########################################################### 
  def _E_logphi(self, k=None):
    if k is None or k == 'prior':
      lam1 = self.Prior.lam1
      lam0 = self.Prior.lam0
    elif k == 'all':
      lam1 = self.Post.lam1
      lam0 = self.Post.lam0
    else:
      lam1 = self.Post.lam1[k]
      lam0 = self.Post.lam0[k]
    Elogphi = digamma(lam1) - digamma(lam1 + lam0)
    return Elogphi

  def _E_log1mphi(self, k=None):
    if k is None or k == 'prior':
      lam1 = self.Prior.lam1
      lam0 = self.Prior.lam0
    elif k == 'all':
      lam1 = self.Post.lam1
      lam0 = self.Post.lam0
    else:
      lam1 = self.Post.lam1[k]
      lam0 = self.Post.lam0[k]
    Elogphi = digamma(lam0) - digamma(lam1 + lam0)
    return Elogphi

  def _E_logphiT(self, k=None):
    ''' Calculate transpose of expected phi matrix 

        Important to make a copy of the matrix so it is C-contiguous,
        which leads to much much faster matrix operations.

        Returns
        -------
        ElogphiT : 2D array, vocab_size x K
    '''
    ElogphiT = self._E_logphi(k).T.copy()
    return ElogphiT

  def _E_log1mphiT(self, k=None):
    ''' Calculate transpose of expected 1-minus-phi matrix 

        Important to make a copy of the matrix so it is C-contiguous,
        which leads to much much faster matrix operations.

        Returns
        -------
        ElogphiT : 2D array, vocab_size x K
    '''
    ElogphiT = self._E_log1mphi(k).T.copy()
    return ElogphiT



def c_Func(lam1, lam0):
  assert lam1.ndim == lam0.ndim
  return np.sum(gammaln(lam1 + lam0) - gammaln(lam1) - gammaln(lam0))

def c_Diff(lamA1, lamA0, lamB1, lamB0):
  return c_Func(lamA1, lamA0) - c_Func(lamB1, lamB0)
