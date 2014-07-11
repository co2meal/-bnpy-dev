'''
MixModel.py
Bayesian parametric mixture model with fixed, finite number of components K

Attributes
-------
  K        : integer number of components
  alpha0   : scalar parameter of symmetric Dirichlet prior on mixture weights
'''
import numpy as np

from bnpy.allocmodel import AllocModel
from bnpy.suffstats import SuffStatBag
from bnpy.util import logsumexp, np2flatstr, flatstr2np
from bnpy.util import gammaln, digamma, EPS

class MixModel(AllocModel):

  ######################################################### Constructors
  #########################################################
  def __init__(self, inferType, priorDict=dict()):
    self.inferType = inferType
    self.set_prior(**priorDict)
    self.K = 0

  def set_prior(self, alpha0=1.0, **kwargs):
    self.alpha0 = alpha0
    if self.alpha0 < 1.0 and self.inferType == 'EM':
      raise ValueError("Cannot perform MAP inference if param alpha0 < 1")

    
  ######################################################### Accessors
  #########################################################
  def get_keys_for_memoized_local_params(self):
    ''' Return list of string names of the LP fields
        that this object needs to memoize across visits to a particular batch
    '''
    return list()

  ######################################################### Local Params
  #########################################################
  def calc_local_params(self, Data, LP, **kwargs):
    ''' Calculate local parameters for each data item and each component.    
        This is part of the E-step.
        
        Args
        -------
        Data : bnpy data object with Data.nObs observations
        LP : local param dict with fields
              E_log_soft_ev : Data.nObs x K array
                  E_log_soft_ev[n,k] = log p(data obs n | comp k)
        
        Returns
        -------
        LP : local param dict with fields
              resp : Data.nObs x K array whose rows sum to one
              resp[n,k] = posterior responsibility that comp. k has for data n
    '''
    lpr = LP['E_log_soft_ev']
    if self.inferType.count('VB') > 0:
      lpr += self.Elogw
      # Calculate exp in numerically stable manner (first subtract the max)
      #  perform this in-place so no new allocations occur
      lpr -= np.max(lpr, axis=1)[:,np.newaxis]
      np.exp(lpr, out=lpr)
      # Normalize, so rows sum to one
      lpr /= lpr.sum(axis=1)[:,np.newaxis]
    elif self.inferType == 'EM' > 0:
      lpr += np.log(self.w)
      lprPerItem = logsumexp(lpr, axis=1)
      np.exp(lpr-lprPerItem[:,np.newaxis], out=lpr)
      LP['evidence'] = lprPerItem.sum()
    LP['resp'] = lpr
    assert np.allclose(lpr.sum(axis=1), 1)
    return LP
    
  def sample_local_params(self, obsModel, Data, SS, LP, PRNG):
    '''
        for i = 1 to Data.nObs 
           sample z_i ~ p(z_i | z_-i,X)
    '''
    Z = LP['Z']
    # Iteratively sample data allocations 
    for dataindex in xrange(Data.nObs):
      x = Data.X[dataindex]

      # de-update current assignment and suff stats
      kcur = Z[dataindex]
      SS.N[kcur] -= 1
      obsModel.decrementSS(SS, kcur, x)

      # Calculate probs
      alloc_prob = self.getConditionalProbVec_Unnorm(SS)
        
      pvec = obsModel.calcPredProbVec_Unnorm(SS, x)
      pvec *= alloc_prob
      pvec /= np.sum(pvec)

      # sample new allocation
      knew = PRNG.choice(SS.K, p=pvec)

      # update with new assignment
      SS.N[knew] += 1  
      obsModel.incrementSS(SS, knew, x) 
      Z[dataindex] = knew

      if dataindex % 100 == 0:
        print '%d/%d' % (dataindex, Data.nObs)

      '''
      for k in xrange(SS.K):
        Xk = np.sum(Data.X[Z==k], axis=0)
        Sk = np.sum(np.square(Data.X[Z==k]), axis=0)
        assert np.allclose(Xk, SS.x[k])
        assert np.allclose(Sk, SS.xx[k])
      '''
    LP['Z'] = Z                      
    return (LP,SS) 
  
  def getConditionalProbVec_Unnorm( self, SS ):
    ''' Returns a K vector of positive values \propto p(z_i|z_-i)
    '''
    return SS.N + self.alpha0

  def calcMargLik(self, SS):
    ''' Calculate marginal likelihood of assignments, summed over all comps
    '''
    alphPost = self.alpha0 + SS.N
    cPrior = gammaln(SS.K * self.alpha0) - SS.K * gammaln(self.alpha0)
    cPost = gammaln(np.sum(alphPost)) - np.sum(gammaln(alphPost))
    return cPrior - cPost

  ######################################################### Suff Stats
  #########################################################
  def get_global_suff_stats(self, Data, LP, doPrecompEntropy=None, **kwargs):
    ''' Calculate the sufficient statistics for global parameter updates
        Only adds stats relevant for this allocModel. 
        Other stats are added by the obsModel.
        
        Args
        -------
        Data : bnpy data object
        LP : local param dict with fields
              resp : Data.nObs x K array,
                       where resp[n,k] = posterior resp of comp k
        doPrecompEntropy : boolean flag
                      indicates whether to precompute ELBO terms in advance
                      used for memoized learning algorithms (moVB)

        Returns
        -------
        SS : SuffStats for K components, with field
              N : vector of length-K,
                   effective number of observations assigned to each comp
    '''
    Nvec = np.sum( LP['resp'], axis=0 )
    if hasattr(Data, 'dim'):
      SS = SuffStatBag(K=Nvec.size, D=Data.dim)
    elif hasattr(Data, 'vocab_size'):
      SS = SuffStatBag(K=Nvec.size, D=Data.vocab_size)

    SS.setField('N', Nvec, dims=('K'))
    if doPrecompEntropy is not None:
      ElogqZ_vec = self.E_logqZ(LP)
      SS.setELBOTerm('ElogqZ', ElogqZ_vec, dims=('K'))
    return SS
    

  ######################################################### Global Params
  #########################################################
  def update_global_params_EM(self, SS, **kwargs):
    if np.allclose(self.alpha0, 1.0):
      w = SS.N
    else:
      w = SS.N + self.alpha0 - 1.0  # MAP estimate. Requires alpha0>1
    self.w = w / w.sum()
    self.K = SS.K
    
  def update_global_params_VB( self, SS, **kwargs):
    self.alpha = self.alpha0 + SS.N
    self.Elogw = digamma( self.alpha ) - digamma( self.alpha.sum() )
    self.K = SS.K

  def update_global_params_soVB( self, SS, rho, **kwargs):
    alphNew = self.alpha0 + SS.N
    self.alpha = rho*alphNew + (1-rho)*self.alpha
    self.Elogw = digamma( self.alpha ) - digamma( self.alpha.sum() )
    self.K = SS.K
 
  def init_global_params(self, Data, K=0, **kwargs):
    ''' Initialize global parameters "from scratch" to prep for learning.

        Will yield uniform distribution (or close to) for all K components,
        by performing a "pseudo" update in which only one observation was
        assigned to each of the K comps.

        Internal Updates
        --------
        Sets attributes w (for EM) or alpha (for VB)

        Returns
        --------
        None. 
    '''
    self.K = K
    if self.inferType == 'EM':
      self.w = 1.0/K * np.ones(K)
    else:
      self.alpha = self.alpha0 + np.ones(K)
      self.Elogw = digamma( self.alpha ) - digamma( self.alpha.sum() )

  def set_global_params(self, hmodel=None, K=None, w=None, beta=None,
                              alpha=None, nObs=10, **kwargs):
    ''' Directly set global parameters alpha to provided values
    '''
    if beta is not None:
      w = beta
    if hmodel is not None:
      self.K = hmodel.allocModel.K
      if self.inferType == 'EM':
        self.w = hmodel.allocModel.w
      else:
        self.alpha = hmodel.allocModel.alpha
        self.Elogw = digamma( self.alpha ) - digamma( self.alpha.sum() )
      return
    else:
      self.K = K
      if self.inferType == 'EM':
        self.w = w
      else:
        if w is not None:
          self.alpha = w * nObs
        elif alpha is not None:
          self.alpha = alpha
        self.Elogw = digamma( self.alpha ) - digamma( self.alpha.sum() )

  ######################################################### Evidence
  #########################################################
  def calc_evidence(self, Data, SS, LP, todict=False, **kwargs):
    if self.inferType == 'EM':
      return LP['evidence'] + self.log_pdf_dirichlet(self.w)
    elif self.inferType.count('VB') > 0:
      evW = self.E_logpW() - self.E_logqW()
      if SS.hasELBOTerm('ElogqZ'):
        ElogqZ = np.sum(SS.getELBOTerm('ElogqZ'))
      else:
        ElogqZ = np.sum(self.E_logqZ(LP))
      if SS.hasAmpFactor():
        evZ = self.E_logpZ(SS) -  SS.ampF * ElogqZ
      else:
        evZ = self.E_logpZ(SS) - ElogqZ
      return evZ + evW 
    else:
      raise NotImplementedError('Unrecognized inferType ' + self.inferType)
      
  def E_logpZ( self, SS ):
    ''' Bishop PRML eq. 10.72
    '''
    return np.inner( SS.N, self.Elogw )
    
  def E_logqZ( self, LP ):
    ''' Bishop PRML eq. 10.75
    '''
    return np.sum(LP['resp']*np.log(LP['resp']+EPS), axis=0)
    
  def E_logpW( self ):
    ''' Bishop PRML eq. 10.73
    '''
    return gammaln(self.K*self.alpha0) \
           - self.K*gammaln(self.alpha0) + (self.alpha0-1)*self.Elogw.sum()
 
  def E_logqW( self ):
    ''' Bishop PRML eq. 10.76
    '''
    return gammaln(self.alpha.sum())-gammaln(self.alpha).sum() \
             + np.inner((self.alpha-1), self.Elogw)

  def log_pdf_dirichlet(self, wvec=None, avec=None):
    ''' Return scalar log probability for Dir(wvec | avec)
    '''
    if wvec is None:
      wvec = self.w
    if avec is None:
      avec = self.alpha0*np.ones(self.K)
    logC = gammaln(np.sum(avec)) - np.sum(gammaln(avec))      
    return logC + np.sum((avec-1.0)*np.log(wvec))

  ######################################################### IO Utils
  #########################################################   for humans
  def get_info_string( self):
    ''' Returns one-line human-readable terse description of this object
    '''
    msgPattern = 'Finite mixture with K=%d. Dir prior param %.2f' 
    return msgPattern % (self.K, self.alpha0)

  ######################################################### IO Utils
  #########################################################   for machines
  def to_dict(self): 
    if self.inferType.count('VB') >0:
      return dict(alpha=self.alpha)
    elif self.inferType == 'EM':
      return dict(w=self.w)
    return dict()
  
  def from_dict(self, myDict):
    self.inferType = myDict['inferType']
    self.K = myDict['K']
    if self.inferType.count('VB') >0:
      self.alpha = myDict['alpha']
      self.Elogw = digamma( self.alpha ) - digamma( self.alpha.sum() )
    elif self.inferType == 'EM':
      self.w = myDict['w']
 
  def get_prior_dict(self):
    return dict(alpha0=self.alpha0, K=self.K)  
