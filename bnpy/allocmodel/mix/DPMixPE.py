'''
DPMixModel.py
Bayesian parametric mixture model with a unbounded number of components K

Prior Attributes
-------
gamma1 : positive real
gamma0 : positive real 

Post Attributes
-------
u : 1D array, size K
'''
import numpy as np

from bnpy.allocmodel import AllocModel
from bnpy.allocmodel.admix import OptimizerForHDPStickBreak as OptimSB
from bnpy.suffstats import SuffStatBag
from bnpy.util import NumericUtil, as1D
from bnpy.util import gammaln, digamma, EPS

class DPMixPE(AllocModel):

  ######################################################### Constructors
  #########################################################
  def __init__(self, inferType, priorDict=None):
    if inferType == 'EM':
      raise ValueError('EM not supported for DPMixPE')
    if inferType == 'GS':
      raise ValueError('GS not supported for DPMixPE')
    self.inferType = inferType
    self.set_prior(priorDict)
    self.K = 0

  def set_prior(self, PriorParamDict):
    self.gamma1 = 1.0
    self.gamma0 = np.maximum(PriorParamDict['gamma0'],
                             1 + 1e-9)

  ######################################################### Accessors
  #########################################################
  def get_active_comp_probs(self):
    ''' Get vector of probabilities for K active components (sum <= 1)
    '''
    return self.E_beta_active()

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
              resp[n,k] = posterior responsibility that 
                          comp. k has for data n                
    '''
    lpr = LP['E_log_soft_ev']
    lpr += self.E_logbeta_active()[np.newaxis, :]
    # Calculate exp in numerically stable manner (first subtract the max)
    #  perform this in-place so no new allocations occur
    NumericUtil.inplaceExpAndNormalizeRows(lpr)
    LP['resp'] = lpr
    assert np.allclose(lpr.sum(axis=1), 1)
    return LP
  
  ######################################################### Suff Stats
  #########################################################
  def get_global_suff_stats(self, Data, LP,
                             preselectroutine=None,
                             doPrecompEntropy=False, 
                             doPrecompMergeEntropy=False, mPairIDs=None):
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
        doPrecompMergeEntropy : boolean flag
                      indicates whether to precompute ELBO terms in advance
                      for all possible merges of pairs of components
                      used for optional merge moves

        Returns
        -------
        SS : SuffStats for K components, with field
              N : vector of length-K,
                   effective number of observations assigned to each comp
    '''
    Nvec = np.sum(LP['resp'], axis=0)
    if hasattr(Data, 'dim'):
      SS = SuffStatBag(K=Nvec.size, D=Data.dim)
    else:
      SS = SuffStatBag(K=Nvec.size, D=Data.vocab_size)

    SS.setField('N', Nvec, dims=('K'))
    if doPrecompEntropy:
      resp = LP['resp']
      np.maximum(resp, 1e-100, out=resp)

      ElogqZ_vec = self.E_logqZ(LP)
      SS.setELBOTerm('ElogqZ', ElogqZ_vec, dims=('K'))

    if doPrecompMergeEntropy:
      if doPrecompMergeEntropy == 2:
        ElogqZVec = NumericUtil.calcRlogR(1.0 - resp)
        SS.setMergeTerm('ElogqZ', -1*ElogqZVec, dims=('K'))
      else:
        if mPairIDs is None:
          ElogqZMat = NumericUtil.calcRlogR_allpairs(resp)
        else:
          ElogqZMat = NumericUtil.calcRlogR_specificpairs(resp, mPairIDs)
        SS.setMergeTerm('ElogqZ', ElogqZMat, dims=('K','K'))

    return SS


  ######################################################### Global Params
  #########################################################
  def update_global_params_VB( self, SS, **kwargs):
    ''' Updates global parameter for point estimate uHat
          for conventional VB learning algorithm.
        New parameters have exactly the number of components specified by SS. 
    '''
    self.K = SS.K
    g1 = self.gamma1 + SS.N
    g0 = self.gamma0 * np.ones(self.K)
    g0[:-1] += SS.N[::-1].cumsum()[::-1][1:]
    if np.any(SS.N < 0):
      np.maximum(g1, self.gamma1, out=g1)
      np.maximum(g0, self.gamma0, out=g0)
    self.uHat = (g1 - 1) / (g1 + g0 - 2)
    
  def update_global_params_soVB( self, SS, rho, **kwargs ):
    ''' Update global params (stick-breaking Beta params qalpha1, qalpha0).
        for stochastic online VB.
    '''
    raise NotImplementedError('ToDo')

  def init_global_params(self, Data, K=0, **kwargs):
    ''' Initialize global parameters "from scratch" to prep for learning.

        Will yield uniform distribution (or close to) for all K components,
        by performing a "pseudo" update in which only one observation was
        assigned to each of the K comps.

        Internal Updates
        --------
        Sets attribute uHat to viable values

        Returns
        --------
        None. 
    '''
    g1 = self.gamma1 + np.ones(K)
    g0 = self.gamma0 + np.arange(K-1, -1, -1)
    self.uHat = (g1 - 1) / (g1 + g0 - 2)
    self.K = K

  def set_global_params(self, hmodel=None, K=None, qalpha1=None, 
                              qalpha0=None, beta=None, nObs=10, **kwargs):
    ''' Directly set global point estimate to provided values
    '''
    if hmodel is not None:
      self.K = hmodel.allocModel.K
      self.uHat = hmodel.allocModel.uHat
      return

    if K is None:
      K = beta.size
    if beta.size == K:
      # expand beta to vector of size K+1, that sums to one
      rem = np.minimum(1.0/K, beta.min()/K)
      beta = np.hstack([beta, rem])
    beta = beta / beta.sum()
    assert beta.size == K+1
    self.uHat = OptimSB._beta2v(beta)
    self.K = K
    assert K == self.uHat.size
 
  ######################################################### Evidence
  #########################################################
  def calc_evidence(self, Data, SS, LP=None, todict=False, **kwargs):
    '''
    '''
    evU = self.E_logpU()
    if SS.hasELBOTerm('ElogqZ'):
      evZq = np.sum(SS.getELBOTerm('ElogqZ'))     
    else:
      evZq = np.sum(self.E_logqZ(LP))
    if SS.hasAmpFactor():
      evZ = self.E_logpZ(SS) -  SS.ampF * evZq
    else:
      evZ = self.E_logpZ(SS) - evZq
    if todict:
      raise NotImplementedError('ToDo')
    return evZ + evU
         
  def E_logpZ(self, SS):
    return np.inner(SS.N, self.E_logbeta_active()) 
    
  def E_logqZ(self, LP):
    return NumericUtil.calcRlogR(LP['resp'])

  def E_logpU( self ):
    return 0
    logBetaPDF =  (self.gamma1 - 1) * np.log(self.uHat) \
                + (self.gamma0 - 1) * np.log(1-self.uHat)
    return self.K * c_Func(self.gamma1, self.gamma0) \
           + np.sum(logBetaPDF)

  """
  def calcSoftMergeEntropyGap(self, SS, kdel, alph):
    ''' Calculate improvement in entropy after a multi-way merge.
    '''
    Halph =  -1 * np.inner(alph, np.log(alph+1e-100))
    Hplus = -1 * SS.getELBOTerm('ElogqZ')[kdel] \
               + SS.getMergeTerm('ElogqZ')[kdel]
    gap = SS.N[kdel] * Halph - Hplus
    return gap

  def calcSoftMergeGap(self, SS, kdel, alph):
    ''' Calculate improvement in allocation ELBO after a multi-way merge.
    '''
    if alph.size < SS.K:
      alph = np.hstack([alph[:kdel], 0, alph[kdel:]])
    assert alph.size == SS.K
    assert np.allclose(alph[kdel], 0)
    gap = 0
    for k in xrange(SS.K):
      if k == kdel:
        gap += c_Func(self.qalpha1[k], self.qalpha0[k]) \
               - c_Func(self.alpha1, self.alpha0)
      elif k > kdel:
        a1 = self.qalpha1[k] + alph[k] * SS.N[kdel]
        a0 = self.qalpha0[k] + np.sum(alph[k+1:]) * SS.N[kdel]
        gap += c_Func(self.qalpha1[k], self.qalpha0[k]) \
                - c_Func(a1, a0)
      elif k < kdel:
        a1 = self.qalpha1[k] + alph[k] * SS.N[kdel]
        a0 = self.qalpha0[k]  - SS.N[kdel] + np.sum(alph[k+1:]) * SS.N[kdel]
        gap += c_Func(self.qalpha1[k], self.qalpha0[k]) \
                - c_Func(a1, a0)
    return gap

  def calcSoftMergeGap_alph(self, SS, kdel, alph):
    ''' Calculate improvement in allocation ELBO after a multi-way merge,
          keeping only terms that depend on redistribution parameters alph
    '''
    if alph.size < SS.K:
      alph = np.hstack([alph[:kdel], 0, alph[kdel:]])
    assert alph.size == SS.K
    assert np.allclose(alph[kdel], 0)
    gap = 0
    for k in xrange(SS.K):
      if k == kdel:
        continue
      elif k > kdel:
        a1 = self.qalpha1[k] + alph[k] * SS.N[kdel]
        a0 = self.qalpha0[k] + np.sum(alph[k+1:]) * SS.N[kdel]
        gap -= c_Func(a1, a0)
      elif k < kdel:
        a1 = self.qalpha1[k] + alph[k] * SS.N[kdel]
        a0 = self.qalpha0[k]  - SS.N[kdel] + np.sum(alph[k+1:]) * SS.N[kdel]
        gap -= c_Func(a1, a0)
    return gap

  def calcHardMergeEntropyGap(self, SS, kA, kB):
    ''' Calc scalar improvement in entropy for hard merge of comps kA, kB
    '''
    Hmerge = SS.getMergeTerm('ElogqZ')
    Hcur   = SS.getELBOTerm('ElogqZ')
    if Hmerge.ndim == 1:
      gap = Hcur[kB] - Hmerge[kB]
    else:
      gap = - Hmerge[kA, kB] + Hcur[kA] + Hcur[kB]
    return gap

  def calcHardMergeGap(self, SS, kA, kB):
    ''' Calculate scalar improvement in ELBO for hard merge of comps kA, kB
        
        Does *not* include any entropy
    '''
    cPrior = c_Func(self.alpha1, self.alpha0)
    cB = c_Func(self.qalpha1[kB], self.qalpha0[kB])
    
    gap = cB - cPrior
    ## Add terms for changing kA to kA+kB
    gap += c_Func(self.qalpha1[kA], self.qalpha0[kA]) \
         - c_Func(self.qalpha1[kA] + SS.N[kB], self.qalpha0[kA] - SS.N[kB])

    ## Add terms for each index kA+1, kA+2, ... kB-1
    ##  where only \alpha_0 has changed
    for k in xrange(kA+1, kB):
      a1 = self.qalpha1[k]
      a0old = self.qalpha0[k]
      a0new = self.qalpha0[k] - SS.N[kB]
      gap += c_Func(a1, a0old) - c_Func(a1, a0new)
    return gap

  def calcHardMergeGap_AllPairs(self, SS):
    ''' Calc matrix of improvement in ELBO for all possible pairs of comps
    '''
    Gap = np.zeros((SS.K, SS.K))
    for kB in xrange(0, SS.K):
      for kA in xrange(0, kB):  
        Gap[kA, kB] = self.calcHardMergeGap(SS, kA, kB)
    return Gap

  def calcHardMergeGap_SpecificPairs(self, SS, PairList):
    ''' Calc matrix of improvement in ELBO for all possible pairs of comps
    '''
    Gaps = np.zeros(len(PairList))
    for ii, (kA, kB) in enumerate(PairList):
        Gaps[ii] = self.calcHardMergeGap(SS, kA, kB)
    return Gaps
  """
  ######################################################### IO Utils
  #########################################################   for humans
  def get_info_string( self):
    ''' Returns one-line human-readable terse description of this object
    '''
    msgPattern = 'DP point estimate with K=%d, Concentration gamma0= %.2f' 
    return msgPattern % (self.K, self.gamma0)

  ######################################################### IO Utils
  #########################################################   for machines
  def to_dict(self): 
    return dict(uHat=self.uHat)
    
  def from_dict(self, myDict):
    self.inferType = myDict['inferType']
    self.K = myDict['K']
    self.uHat = as1D(myDict['uHat'])
    
  def get_prior_dict(self):
    return dict(gamma1=self.gamma1,
                gamma0=self.gamma0,
                K=self.K, 
                )


  ######################################################### Expectations
  ######################################################### 
  def E_beta_active(self):
    ''' Calculate vector of component probabilities for active components

        Returns
        --------
        Ebeta : 1D array, size K
                Ebeta[k] gives expected probability for active comp k
    '''
    activeBeta = OptimSB._v2beta(self.uHat)[:-1]
    return activeBeta

  def E_logbeta_active(self):
    ''' Calculate vector of log component probabilities for active components

        Returns
        --------
        Elogbeta : 1D array, size K
                   Elogbeta[k] gives expected log probability for active comp k
    '''
    activeBeta = OptimSB._v2beta(self.uHat)[:-1]
    np.maximum(activeBeta, 1e-100, out=activeBeta)
    return np.log(activeBeta)

  def E_logbeta(self):
    ''' Calculate vector of log component probabilities (includes leftover mass)

        Returns
        --------
        Elogbeta : 1D array, size K+1
                   Elogbeta[k] gives expected log probability for active comp k
                   Elogbeta[-1] gives aggregate log prob for all inactive comps
    '''
    beta = OptimSB._v2beta(self.uHat)
    np.maximum(beta, 1e-100, out=beta)
    return np.log(beta)

def c_Func(gamma1, gamma0):
  return gammaln(gamma1+gamma0) - gammaln(gamma1) - gammaln(gamma0)
