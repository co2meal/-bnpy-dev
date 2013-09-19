'''
DPMixModel.py
Bayesian parametric mixture model with a unbounded number of components K

Attributes
-------
  K        : # of components
  alpha0   : scalar concentration hyperparameter of Dirichlet process prior
  
  qalpha0 : K-len vector of neg Beta param for variational approx to stick-break distr
  qalpha1 : K-len vector of pos Beta param for variational approx to stick-break distr
  truncType : str type of truncation on the infinite tail of the Dirichlet Process
              either 'z' (truncate on the assignments) 
                  or 'v' (truncates actual stick-breaking distribution)
'''
import numpy as np

from bnpy.allocmodel import AllocModel
from bnpy.suffstats import SuffStatDict
from bnpy.util import logsumexp, np2flatstr, flatstr2np
from bnpy.util import gammaln, digamma, EPS

class DPMixModel(AllocModel):
  
  def __init__(self, inferType, priorDict=None):
    self.inferType = inferType
    if priorDict is None:
      self.alpha0 = 1.0 # Uniform!
      self.alpha1 = 1.0
      self.truncType = 'z'
    else:
      self.set_prior(priorDict)
    self.K = 0

  ############################################################## basic accessors
  ############################################################## 
  def is_nonparametric(self):
    return True
    
  def set_helper_params( self ):
    ''' Set dependent attributes of this model given the primary global params.
        For DP mixture, these include predcomputing digammas.
    '''
    DENOM = digamma(self.qalpha0 + self.qalpha1)
    self.ElogV      = digamma(self.qalpha1) - DENOM
    self.Elog1mV    = digamma(self.qalpha0) - DENOM

    if self.truncType == 'v':
      self.qalpha1[-1] = 1
      self.qalpha0[-1] = EPS # avoid digamma(0), which is way too HUGE
      self.ElogV[-1] = 0  # log(1) => 0
      self.Elog1mV[-1] = np.log(1e-40) # log(0) => -INF, never used
		
		# Calculate expected mixture weights E[ log w_k ]	 
    self.Elogw = self.ElogV.copy() #copy so we can do += without modifying ElogV
    self.Elogw[1:] += self.Elog1mV[:-1].cumsum()
    
  def set_prior(self, PriorParamDict):
    self.alpha1 = 1.0
    self.alpha0 = PriorParamDict['alpha0']
    self.truncType = PriorParamDict['truncType']
      
  ############################################################## human readable I/O  
  ##############################################################  
  def get_info_string( self):
    ''' Returns one-line human-readable terse description of this object
    '''
    return 'Unbounded mixture with K=%d. DP conc param %.2f' % (self.K, self.alpha0)

  def get_human_global_param_string(self):
    ''' Returns human-readable numerical repr. of parameters,
          for quick inspection of correctness
    '''
    if not self.isReady():
      return ''
    raise NotImplementedError('TODO')
  
  ############################################################## MAT file I/O  
  ##############################################################  
  def to_dict(self): 
    return dict(qalpha1=self.qalpha1, qalpha0=self.qalpha0)
    
  def from_dict(self, myDict):
    self.inferType = myDict['inferType']
    self.K = myDict['K']
    self.qalpha1 = myDict['qalpha1']
    self.qalpha0 = myDict['qalpha0']
    self.set_helper_params()
    
  def get_prior_dict(self):
    return dict(alpha1=self.alpha1, alpha0=self.alpha0, K=self.K, truncType=self.truncType)  
    
  ############################################################## Suff Stat Calc   
  ##############################################################
  def get_global_suff_stats(self, Data, LP, doPrecompEntropy=False):
    ''' Calculate the sufficient statistics for global parameter updates
        Only adds stats relevant for this allocModel. Other stats added by the obsModel.
        
        Args
        -------
        Data : bnpy data object
        LP : local param dict with fields
              resp : Data.nObs x K array where resp[n,k] = posterior resp of comp k
        doPrecompEntropy : boolean flag that indicates whether to precompute the entropy of the data responsibilities (used for evaluating the evidence)
              
        Returns
        -------
        SS : SuffStatDict with K components, with field
              N : K-len vector of effective number of observations assigned to each comp
    '''
    Nvec = np.sum(LP['resp'], axis=0)
    SS = SuffStatDict(N=Nvec)
    if doPrecompEntropy:
      Hvec = np.sum( LP['resp'] * np.log(EPS+LP['resp']), axis=0 )
      SS.addPrecompEntropy(Hvec)
    return SS
    
  ############################################################# Local Param Updates   
  #############################################################
  def calc_local_params( self, Data, LP, Krange=None ):
    ''' Calculate posterior responsibilities for each data item and each component.    
        This is part of the E-step of the EM/VB algorithm.
        
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
                      resp[n,k] = posterior prob. that component k generated data n                
    '''
    lpr = self.Elogw + LP['E_log_soft_ev']
    lprPerItem = logsumexp( lpr, axis=1 )
    resp   = np.exp( lpr-lprPerItem[:,np.newaxis] )
    LP['resp'] = resp
    # Reclaim memory, don't need NxK matrix anymore
    del LP['E_log_soft_ev']
    return LP
    
  ############################################################## Param Update   
  ##############################################################
  def update_global_params_VB( self, SS, **kwargs ):
    ''' Updates global params (stick-breaking Beta params qalpha1, qalpha0)
        for conventional VB learning algorithm.
    '''
    assert self.K == SS.K
    qalpha1 = self.alpha1 + SS.N
    qalpha0 = self.alpha0 * np.ones(self.K)
    qalpha0[:-1] += SS.N[::-1].cumsum()[::-1][1:]
    self.qalpha1 = qalpha1
    self.qalpha0 = qalpha0
    self.set_helper_params()
    
  def update_global_params_soVB( self, SS, rho, **kwargs ):
    ''' Update global params (stick-breaking Beta params qalpha1, qalpha0).
        for stochastic online VB.
    '''
    assert self.K == SS.K
    qalpha1 = self.alpha1 + SS.N
    qalpha0 = self.alpha0 * np.ones( self.K )
    qalpha0[:-1] += SS.N[::-1].cumsum()[::-1][1:]
    
    self.qalpha1 = rho * qalpha1 + (1-rho) * self.qalpha1
    self.qalpha0 = rho * qalpha0 + (1-rho) * self.qalpha0
    self.set_helper_params()

  ############################################################## Evidence calc.   
  ##############################################################
  def calc_evidence(self, Data, SS, LP=None ):
    ''' Compute parts of the evidence lower bound (ELBO) of the objective function.
        Parts relevant to the DP mixture include terms involving stick-break weights V and cluster assignments Z.
    '''
    evV = self.E_logpV() - self.E_logqV()
    if SS.hasPrecompEntropy():
      evZq = np.sum(SS.getPrecompEntropy())     
    else:
      evZq = self.E_logqZ( LP )
    if SS.hasAmpFactor():
      evZ = self.E_logpZ(SS) -  SS.ampF * evZq
    else:
      evZ = self.E_logpZ(SS) - evZq
    return evZ + evV
         
  def E_logpZ(self, SS):
    '''
      E[ log p( Z | V ) ] = \sum_n E[ log p( Z[n] | V )
         = \sum_n E[ log p( Z[n]=k | w(V) ) ]
         = \sum_n \sum_k z_nk log w(V)_k
    '''
    return np.inner( SS['N'], self.Elogw ) 
    
  def E_logqZ( self, LP ):
    return np.sum( LP['resp'] *np.log(LP['resp']+EPS) )
    
  def E_logpV( self ):
    '''
      E[ log p( V | alpha ) ] = sum_{k=1}^K  E[log[ Z(alpha) Vk^(a1-1) * (1-Vk)^(a0-1) ]]
         = sum_{k=1}^K log Z(alpha)  + (a1-1) E[ logV ] + (a0-1) E[ log (1-V) ]
    '''
    logZprior = gammaln( self.alpha0 + self.alpha1 ) - gammaln(self.alpha0) - gammaln( self.alpha1 )
    logEterms  = (self.alpha1-1)*self.ElogV + (self.alpha0-1)*self.Elog1mV
    if self.truncType == 'z':
	    return self.K*logZprior + logEterms.sum()    
    elif self.truncType == 'v':
      return self.K*logZprior + logEterms[:-1].sum()

  def E_logqV( self ):
    '''
      E[ log q( V | qa ) ] = sum_{k=1}^K  E[log[ Z(qa) Vk^(ak1-1) * (1-Vk)^(ak0-1)  ]]
       = sum_{k=1}^K log Z(qa)   + (ak1-1) E[logV]  + (a0-1) E[ log(1-V) ]
    '''
    logZq = gammaln( self.qalpha0 + self.qalpha1 ) - gammaln(self.qalpha0) - gammaln( self.qalpha1 )
    logEterms  = (self.qalpha1-1)*self.ElogV + (self.qalpha0-1)*self.Elog1mV
    if self.truncType == 'z':
      return logZq.sum() + logEterms.sum()
    elif self.truncType == 'v':
      return logZq[:-1].sum() + logEterms[:-1].sum()  # entropy of deterministic draw =0
    