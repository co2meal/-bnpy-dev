'''
  MixModel.py
     Bayesian parametric mixture model with a finite number of components K

 Parameters
 -------
   K        : # of components
   alpha0   : scalar hyperparameter of symmetric Dirichlet prior on mix. weights

'''
import numpy as np

from bnpy.allocmodel import AllocModel
from bnpy.suffstats import SuffStatCompSet
from bnpy.util import logsumexp, np2flatstr, flatstr2np
from bnpy.util import gammaln, digamma, EPS

class MixModel(AllocModel):
  def __init__(self, inferType, priorDict=None):
    self.inferType = inferType
    if priorDict is None:
      self.alpha0 = 0
    else:
      self.alpha0 = priorDict['alpha0']
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
  def get_global_suff_stats(self, Data, LP, doPrecompEntropy=None, **kwargs ):
    ''' 
    '''
    Nvec = np.sum( LP['resp'], axis=0 )
    SS = SuffStatCompSet( Nvec=Nvec )

    if doPrecompEntropy is not None:
      Elogq_vec = np.sum( LP['resp'] * np.log(EPS+LP['resp']), axis=0)
      SS.addPrecompEntropy( Elogq_vec )
    return SS
    
  ##############################################################    
  ############################################################## Local Param Updates   
  ##############################################################
  def calc_local_params( self, Data, LP ):
    ''' E-step
    '''
    if self.inferType.count('VB') > 0:
      lpr = self.Elogw + LP['E_log_soft_ev']
    elif self.inferType == 'EM' > 0:
      lpr = np.log(self.w) + LP['E_log_soft_ev']
    lprPerItem = logsumexp( lpr, axis=1 )
    resp   = np.exp( lpr-lprPerItem[:,np.newaxis] )
    LP['resp'] = resp
    assert np.allclose( resp.sum(axis=1), 1.0 )
    if self.inferType == 'EM':
        LP['evidence'] = lprPerItem.sum()
    # Don't need this memory anymore
    del LP['E_log_soft_ev']
    return LP
    
  ##############################################################    
  ############################################################## Global Param Updates   
  ##############################################################
  def update_global_params_EM( self, SS, **kwargs):
    if np.allclose(self.alpha0, 0.0):
      w = SS.Nvec
    else:
      w = SS.Nvec + self.alpha0 - 1.0  # MAP estimate. Requires alpha0>1
    self.w = w / w.sum()
    self.K = SS.K
    
  def update_global_params_VB( self, SS, **kwargs):
    self.alpha = self.alpha0 + SS.Nvec
    self.Elogw = digamma( self.alpha ) - digamma( self.alpha.sum() )
    self.K = SS.K

  def update_global_params_soVB( self, SS, rho, **kwargs):
    alphNew = self.alpha0 + SS.Nvec
    self.alpha = rho*alphNew + (1-rho)*self.alpha
    self.Elogw = digamma( self.alpha ) - digamma( self.alpha.sum() )
    self.K = SS.K
    
  ##############################################################    
  ############################################################## Evidence calc.   
  ##############################################################
  def calc_evidence( self, Data, SS, LP):
    if self.inferType == 'EM':
      if np.allclose( self.alpha0, 0.0 ):
        return LP['evidence']
      else:
        return LP['evidence'] + self.log_pdf_dirichlet(self.w)
        
    elif self.inferType.count('VB') >0:
      evW = self.E_logpW() - self.E_logqW()
      if SS.hasPrecompEntropy():
        ElogqZ = np.sum( SS.Hvec )
      else:
        ElogqZ = self.E_logqZ( LP )
      if SS.hasAmpFactor():
        evZ = self.E_logpZ( SS ) -  SS.ampF * ElogqZ
      else:
        evZ = self.E_logpZ( SS ) - ElogqZ
      return evZ + evW 
      
  def E_logpZ( self, SS ):
    ''' Bishop PRML eq. 10.72
    '''
    return np.inner( SS.Nvec, self.Elogw )
    
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