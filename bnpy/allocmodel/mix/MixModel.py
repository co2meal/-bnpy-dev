'''
MixModel.py
Bayesian parametric mixture model with fixed, finite number of components K

Attributes
-------
  K        : # of components
  alpha0   : scalar hyperparameter of symmetric Dirichlet prior on mix. weights

'''
import numpy as np

from bnpy.allocmodel import AllocModel
from bnpy.suffstats import SuffStatDict
from bnpy.util import logsumexp, np2flatstr, flatstr2np
from bnpy.util import gammaln, digamma, EPS

class MixModel(AllocModel):
  def __init__(self, inferType, priorDict=None):
    self.inferType = inferType
    if priorDict is None:
      self.alpha0 = 1.0 # Uniform!
    else:
      self.set_prior(priorDict)
    self.K = 0
    
  def isReady(self):
    try:
      if self.inferType == 'EM':
        return self.K > 0 and len(self.w) == self.K
      else:
        return self.K > 0 and len(self.alpha) == self.K
    except AttributeError:
      return False

  ##############################################################    
  ############################################################## set prior parameters  
  ############################################################## 
  def set_prior(self, PriorParamDict):
    self.alpha0 = PriorParamDict['alpha0']
    if self.alpha0 < 1.0 and self.inferType == 'EM':
      raise ValueError("Cannot perform MAP inference if Dir prior param alpha0 < 1")
      
  ##############################################################    
  ############################################################## human readable I/O  
  ##############################################################  
  def get_info_string( self):
    ''' Returns one-line human-readable terse description of this object
    '''
    return 'Finite mixture with K=%d. Dir prior param %.2f' % (self.K, self.alpha0)

  def get_human_global_param_string(self):
    ''' Returns human-readable numerical repr. of parameters,
          for quick inspection of correctness
    '''
    if not self.isReady():
      return ''
    if self.inferType == 'EM':
      return np2flatstr( self.w, '%3.2f' )
    else:
      return np2flatstr( np.exp(self.Elogw), '%3.2f' )

  ##############################################################    
  ############################################################## MAT file I/O  
  ##############################################################  
  def to_dict(self): 
    if self.inferType.count('VB') >0:
      return dict( alpha=self.alpha)
    elif self.inferType == 'EM':
      return dict( w=self.w)
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
    return dict( alpha0=self.alpha0, K=self.K )  

  ##############################################################    
  ############################################################## Suff Stat Calc   
  ##############################################################
  def get_global_suff_stats(self, Data, LP, doPrecompEntropy=None, **kwargs):
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
    Nvec = np.sum( LP['resp'], axis=0 )
    SS = SuffStatDict(N=Nvec)
    if doPrecompEntropy is not None:
      Elogq_vec = np.sum( LP['resp'] * np.log(EPS+LP['resp']), axis=0)
      SS.addPrecompEntropy( Elogq_vec )
    return SS
    
  ##############################################################    
  ############################################################## Local Param Updates   
  ##############################################################
  def calc_local_params(self, Data, LP):
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
    if self.inferType.count('VB') > 0:
      lpr = self.Elogw + LP['E_log_soft_ev']
    elif self.inferType == 'EM' > 0:
      lpr = np.log(self.w) + LP['E_log_soft_ev']
    lprPerItem = logsumexp( lpr, axis=1 )
    resp   = np.exp( lpr-lprPerItem[:,np.newaxis] )
    LP['resp'] = resp
    if self.inferType == 'EM':
        LP['evidence'] = lprPerItem.sum()
    # Reclaim memory, don't need NxK matrix anymore
    del LP['E_log_soft_ev']
    return LP
    
  ##############################################################    
  ############################################################## Global Param Updates   
  ##############################################################
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
    
  ##############################################################    
  ############################################################## Evidence calc.   
  ##############################################################
  def calc_evidence( self, Data, SS, LP):
    if self.inferType == 'EM':
      return LP['evidence'] + self.log_pdf_dirichlet(self.w)
        
    elif self.inferType.count('VB') >0:
      evW = self.E_logpW() - self.E_logqW()
      if SS.hasPrecompEntropy():
        ElogqZ = np.sum(SS.getPrecompEntropy())
      else:
        ElogqZ = self.E_logqZ(LP)
      if SS.hasAmpFactor():
        evZ = self.E_logpZ(SS) -  SS.ampF * ElogqZ
      else:
        evZ = self.E_logpZ(SS) - ElogqZ
      return evZ + evW 
      
  def E_logpZ( self, SS ):
    ''' Bishop PRML eq. 10.72
    '''
    return np.inner( SS.N, self.Elogw )
    
  def E_logqZ( self, LP ):
    ''' Bishop PRML eq. 10.75
    '''
    return np.sum(  LP['resp']*np.log( LP['resp']+EPS) )
    
  def E_logpW( self ):
    ''' Bishop PRML eq. 10.73
    '''
    return gammaln(self.K*self.alpha0) \
             -self.K*gammaln(self.alpha0) +(self.alpha0-1)*self.Elogw.sum()
 
  def E_logqW( self ):
    ''' Bishop PRML eq. 10.76
    '''
    return gammaln(self.alpha.sum())-gammaln(self.alpha).sum() \
             + np.inner( (self.alpha-1), self.Elogw )

  def log_pdf_dirichlet( self, wvec=None, avec=None):
    if wvec is None:
      wvec = self.w
    if avec is None:
      avec = self.alpha0*np.ones(self.K)
    logC = gammaln(np.sum(avec)) - np.sum(gammaln(avec))      
    return logC + np.sum((avec-1.0)*np.log(wvec))