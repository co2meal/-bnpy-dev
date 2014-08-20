'''
DPMixFull.py
Bayesian parametric mixture model with a unbounded number of components K

Attributes
-------
  K        : # of components
  gamma0   : scalar concentration hyperparameter of Dirichlet process prior
'''

import numpy as np

from bnpy.allocmodel import AllocModel
from bnpy.suffstats import SuffStatBag
from bnpy.util import NumericUtil
from bnpy.util import gammaln, digamma, EPS


def c_Beta(gamma1, gamma0):
  ''' Evaluate cumulant function of the Beta distribution
  '''
  return np.sum(gammaln(gamma1+gamma0) - gammaln(gamma1) - gammaln(gamma0))


class DPMixFull(AllocModel):

  ######################################################### Constructors
  #########################################################
  def __init__(self, inferType, priorDict=None, **priorKwargs):
    if inferType == 'EM':
      raise ValueError('EM not supported for DPMixModel')
    self.inferType = inferType
    if priorDict is None:
      self.set_prior(**priorDict)
    else:
      self.set_prior(**priorKwargs)
    self.K = 0

  def set_prior(self, gamma1=1.0, gamma0=5.0, **kwargs):
    self.gamma1 = gamma1
    self.gamma0 = gamma0
    
  def set_helper_params( self ):
    ''' Set dependent attributes given primary global params.
        For DP mixture, this means precomputing digammas.
    '''
    digammaBoth = digamma(self.qgamma0 + self.qgamma1)
    self.ElogU      = digamma(self.qgamma1) - digammaBoth
    self.Elog1mU    = digamma(self.qgamma0) - digammaBoth
		
		# Calculate expected mixture weights E[ log \beta_k ]	 
    self.Elogbeta = self.ElogU.copy() # copy allows += without modifying ElogU
    self.Elogbeta[1:] += self.Elog1mU[:-1].cumsum()
    
  ######################################################### Accessors
  #########################################################
  def get_active_comp_probs(self):
    Eu = self.qgamma1 / (self.qgamma1 + self.qgamma0)
    Ebeta = Eu.copy()
    Ebeta[1:] *= np.cumprod(1.0-Eu[:-1]) 
    return Ebeta

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
    lpr += self.Elogbeta
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
      np.minimum(resp, 1-EPS, out=resp)
      np.maximum(resp, EPS, out=resp)

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
    ''' Updates global params (stick-breaking Beta params qgamma1, qgamma0)
          for conventional VB learning algorithm.
        New parameters have exactly the number of components specified by SS. 
    '''
    self.K = SS.K
    qgamma1 = self.gamma1 + SS.N
    qgamma0 = self.gamma0 * np.ones(self.K)
    qgamma0[:-1] += SS.N[::-1].cumsum()[::-1][1:]
    self.qgamma1 = qgamma1
    self.qgamma0 = qgamma0
    self.set_helper_params()
    
  def update_global_params_soVB( self, SS, rho, **kwargs ):
    ''' Update global params (stick-breaking Beta params qgamma1, qgamma0).
        for stochastic online VB.
    '''
    assert self.K == SS.K
    qgamma1 = self.gamma1 + SS.N
    qgamma0 = self.gamma0 * np.ones( self.K )
    qgamma0[:-1] += SS.N[::-1].cumsum()[::-1][1:]
    
    self.qgamma1 = rho * qgamma1 + (1-rho) * self.qgamma1
    self.qgamma0 = rho * qgamma0 + (1-rho) * self.qgamma0
    self.set_helper_params()

  def init_global_params(self, Data, K=0, **kwargs):
    ''' Initialize global parameters "from scratch" to prep for learning.

        Will yield uniform distribution (or close to) for all K components,
        by performing a "pseudo" update in which only one observation was
        assigned to each of the K comps.

        Internal Updates
        --------
        Sets attributes qgamma1, qgamma0 (for VB) to viable values

        Returns
        --------
        None. 
    '''
    self.K = K
    Nvec = np.ones(K)
    qgamma1 = self.gamma1 + Nvec
    qgamma0 = self.gamma0 * np.ones(self.K)
    qgamma0[:-1] += Nvec[::-1].cumsum()[::-1][1:]
    self.qgamma1 = qgamma1
    self.qgamma0 = qgamma0
    self.set_helper_params()

  def set_global_params(self, hmodel=None, K=None, qgamma1=None, 
                              qgamma0=None, beta=None, nObs=10, **kwargs):
    ''' Directly set global parameters qgamma0, qgamma1 to provided values
    '''
    if hmodel is not None:
      self.K = hmodel.allocModel.K
      self.qgamma1 = hmodel.allocModel.qgamma1
      self.qgamma0 = hmodel.allocModel.qgamma0
      self.set_helper_params()
      return
    if beta is not None:
      if K is None:
        K = beta.size
      # convert to expected stick-lengths v
      import bnpy.allocmodel.admix.OptimizerForHDPStickBreak as OptimSB
      if beta.size == K:
        rem = np.minimum(0.01, 1.0/K)
        rem = np.minimum(1.0/K, beta.min()/K)
        beta = np.hstack( [beta, rem])
      beta = beta / beta.sum()
      Ev = OptimSB._beta2v(beta)
      qgamma1 = Ev * nObs
      qgamma0 = (1-Ev) * nObs

    if type(qgamma1) != np.ndarray or qgamma1.size != K or qgamma0.size != K:
      raise ValueError("Bad DP Parameters")
    self.K = K
    self.qgamma1 = qgamma1
    self.qgamma0 = qgamma0
    self.set_helper_params()
 
  ######################################################### Evidence
  #########################################################
  def calc_evidence(self, Data, SS, LP=None, todict=False, **kwargs):
    ''' Calculate ELBO
    '''
    if SS.hasELBOTerm('ElogqZ'):
      Hentropy = np.sum(SS.getELBOTerm('ElogqZ'))     
    else:
      Hentropy = np.sum(self.E_logqZ(LP))
    if SS.hasAmpFactor():
      Hentropy *= SS.ampF

    cDiff = self.ELBO_cDiff()
    slack = self.ELBO_slack(SS)
    return cDiff + slack - Hentropy
             
  def E_logqZ(self, LP):
    return NumericUtil.calcRlogR(LP['resp'])

  def ELBO_cDiff(self):
    ''' Compute difference of cumulant functions for ELBO 

        Returns
        -------
        cDiff : scalar real
    '''
    cDiff = self.K * c_Beta(self.gamma1, self.gamma0) \
                   - c_Beta(self.qgamma1, self.qgamma0) # already sums over k
    return cDiff

  def ELBO_slack(self, SS):
    ''' Compute the slack-term for ELBO
    '''
    slack = np.inner(self.gamma1 - self.qgamma1, self.ElogU) \
            + np.inner(self.gamma0 - self.qgamma0, self.Elog1mU) \
            + np.inner(SS.N, self.Elogbeta)
    return slack
    

  ######################################################### Hard Merges
  #########################################################
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
    cPrior = c_Beta(self.gamma1, self.gamma0)
    cB = c_Beta(self.qgamma1[kB], self.qgamma0[kB])
    
    gap = cB - cPrior
    ## Add terms for changing kA to kA+kB
    gap += c_Beta(self.qgamma1[kA], self.qgamma0[kA]) \
         - c_Beta(self.qgamma1[kA] + SS.N[kB], self.qgamma0[kA] - SS.N[kB])

    ## Add terms for each index kA+1, kA+2, ... kB-1
    ##  where only \gamma_0 has changed
    for k in xrange(kA+1, kB):
      a1 = self.qgamma1[k]
      a0old = self.qgamma0[k]
      a0new = self.qgamma0[k] - SS.N[kB]
      gap += c_Beta(a1, a0old) - c_Beta(a1, a0new)
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


  ######################################################### Soft Merges
  #########################################################
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
        gap += c_Beta(self.qgamma1[k], self.qgamma0[k]) \
               - c_Beta(self.gamma1, self.gamma0)
      elif k > kdel:
        a1 = self.qgamma1[k] + alph[k] * SS.N[kdel]
        a0 = self.qgamma0[k] + np.sum(alph[k+1:]) * SS.N[kdel]
        gap += c_Beta(self.qgamma1[k], self.qgamma0[k]) \
                - c_Beta(a1, a0)
      elif k < kdel:
        a1 = self.qgamma1[k] + alph[k] * SS.N[kdel]
        a0 = self.qgamma0[k]  - SS.N[kdel] + np.sum(alph[k+1:]) * SS.N[kdel]
        gap += c_Beta(self.qgamma1[k], self.qgamma0[k]) \
                - c_Beta(a1, a0)
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
        a1 = self.qgamma1[k] + alph[k] * SS.N[kdel]
        a0 = self.qgamma0[k] + np.sum(alph[k+1:]) * SS.N[kdel]
        gap -= c_Beta(a1, a0)
      elif k < kdel:
        a1 = self.qgamma1[k] + alph[k] * SS.N[kdel]
        a0 = self.qgamma0[k]  - SS.N[kdel] + np.sum(alph[k+1:]) * SS.N[kdel]
        gap -= c_Beta(a1, a0)
    return gap

  ######################################################### IO Utils
  #########################################################   for humans
  def get_info_string( self):
    ''' Returns one-line human-readable terse description of this object
    '''
    msgPattern = 'DP mixture with K=%d. Concentration gamma0= %.2f' 
    return msgPattern % (self.K, self.gamma0)

  ######################################################### IO Utils
  #########################################################   for machines
  def to_dict(self): 
    return dict(qgamma1=self.qgamma1, qgamma0=self.qgamma0)
    
  def from_dict(self, myDict):
    self.inferType = myDict['inferType']
    self.K = myDict['K']
    self.qgamma1 = myDict['qgamma1']
    self.qgamma0 = myDict['qgamma0']
    if self.qgamma0.ndim == 0:
      self.qgamma0 = self.qgamma1[np.newaxis]
    if self.qgamma0.ndim == 0:
      self.qgamma0 = self.qgamma0[np.newaxis]
    self.set_helper_params()
    
  def get_prior_dict(self):
    return dict(gamma1=self.gamma1,
                gamma0=self.gamma0,
                K=self.K, 
                )  


  ######################################################### Local Updates Gibbs
  #########################################################
  def make_hard_asgn_local_params(self, LP):
    ''' Convert soft assignments to hard assignments for provided local params

        Returns
        --------
        LP : local params dict, with new fields
             Z : 1D array, size N
                    Z[n] is an integer in range {0, 1, 2, ... K-1}
             resp : 2D array, size N x K+1 (with final column empty)
                    resp[n,k] = 1 iff Z[n] == k
    '''
    LP['Z'] = np.argmax(LP['resp'], axis=1)
    K = LP['resp'].shape[1]
    LP['resp'].fill(0)
    for k in xrange(K):
      LP['resp'][LP['Z']==k, k] = 1 
    return LP

  def removeEmptyComps_SSandLP(self, SS, LP):
    ''' Remove all parameters related to empty components from SS and LP

        Returns
        --------
        SS : bnpy SuffStatBag
        LP : dict for local params
    '''
    badks = np.flatnonzero(SS.N[:-1] < 1)
    for k in badks[::-1]:  # Remove in order, from largest index to smallest
      SS.removeComp(k)
      mask = LP['Z'] > k
      LP['Z'][mask] -= 1
    if 'resp' in LP:
      del LP['resp']
    return SS, LP

  def insertEmptyCompAtLastIndex_SSandLP(self, SS, LP):
    ''' Create empty component and insert last in order into SS 

        Returns
        --------
        SS
        LP
    '''
    SS.insertEmptyComps(1)
    return SS, LP


  def sample_local_params(self, obsModel, Data, SS, LP, PRNG, **algParams):
    ''' Sample local assignments of all data items to components
    '''
    Z = LP['Z']
    # Iteratively sample data allocations 
    for dataindex in xrange(Data.nObs):
      x = Data.X[dataindex]

      # de-update current assignment and suff stats
      kcur = Z[dataindex]
      SS.N[kcur] -= 1
      obsModel.decrementSS(SS, kcur, x)

      SS, LP = self.removeEmptyComps_SSandLP(SS, LP)

      doKeepFinalCompEmpty = SS.K < algParams['Kmax']
      if SS.N[-1] > 0 and doKeepFinalCompEmpty:
        SS, LP = self.insertEmptyCompAtLastIndex_SSandLP(SS, LP)    

      # Calculate probs
      alloc_prob = self.getConditionalProbVec_Unnorm(SS, doKeepFinalCompEmpty)
      pvec = obsModel.calcPredProbVec_Unnorm(SS, x)
      pvec *= alloc_prob
      psum = np.sum(pvec)

      if np.isnan(psum) or psum <= 0:
        print pvec
        print psum
        raise ValueError('BAD VALUES FOR PROBS!')

      pvec /= psum
      # sample new allocation
      knew = PRNG.choice(SS.K, p=pvec)

      # update with new assignment
      SS.N[knew] += 1  
      obsModel.incrementSS(SS, knew, x) 
      Z[dataindex] = knew
      
    LP['Z'] = Z                      
    print ' '.join(['%.1f' % (x) for x in SS.N])
    return LP, SS 
  
  def getConditionalProbVec_Unnorm(self, SS, doKeepFinalCompEmpty):
    ''' Returns a K vector of positive values \propto p(z_i|z_-i)
    '''
    if doKeepFinalCompEmpty:
      assert SS.N[-1] == 0
      return np.hstack([SS.N[:-1], self.gamma0])
    else:
      return np.hstack([SS.N[:-1], np.maximum(SS.N[-1], self.gamma0)])
  
  def calcMargLik(self, SS):
    ''' Calculate marginal likelihood of assignments, summed over all comps
    '''
    mask = SS.N > 0
    Nvec = SS.N[mask]
    K = Nvec.size
    return gammaln(self.gamma0) \
           + K * np.log(self.gamma0) \
           + np.sum(gammaln(Nvec)) \
           - gammaln(np.sum(Nvec) + self.gamma0)