'''
  MixModel.py
     Bayesian parametric mixture model with a finite number of components K

 Author: Mike Hughes (mike@michaelchughes.com)

 Parameters
 -------
   K        : # of components
   alpha0   : scalar hyperparameter of symmetric Dirichlet prior on mix. weights

'''
from IPython import embed
import numpy as np

from ..AllocModel import AllocModel

from bnpy.util import logsumexp, np2flatstr, flatstr2np
from bnpy.util import gammaln, digamma, EPS
from bnpy.util import discrete_single_draw

class MixModel( AllocModel ):
  def __init__(self, K=3, alpha0=0.0, qType='VB', **kwargs ):
    self.qType = qType
    self.K = K
    self.alpha0 = float(alpha0)
    self.isScalar = True
    if qType == 'EM' and self.alpha0 >0 and self.alpha0 < 1:
      raise ValueError( 'MAP estimates only exist when alpha0 > 1')  
    elif qType.count('VB') > 0:
      assert self.alpha0 > 0
    
  def to_dict(self): 
    if self.qType.count('VB') >0:
      return dict( alpha=self.alpha)
    elif self.qType == 'EM':
      return dict( w=self.w )
    elif self.qType == 'CGS':
      return dict( alpha=self.alpha )
    return dict()
  
  def from_dict(self, D):
    if self.qType.count('VB') >0:
      self.alpha = D['alpha']
      self.Elogw = digamma( self.alpha ) - digamma( self.alpha.sum() )
    elif self.qType == 'EM':
      self.w = D['w']
 
  def get_prior_dict(self):
    return dict( alpha0=self.alpha0, K=self.K, qType=self.qType )
 
  def get_info_string( self):
    ''' Returns human-readable description of this object
    '''
    return 'Finite mixture with K=%d. Dir prior param %.2f' % (self.K, self.alpha0)

  def get_human_global_param_string(self):
    ''' Returns human-readable numerical repr. of parameters,
          for quick inspection of correctness
    '''
    if self.qType == 'EM':
      return np2flatstr( self.w, '%3.2f' )
    else:
      return np2flatstr( np.exp(self.Elogw), '%3.2f' )
    
  ##############################################################    
  ############################################################## Suff Stat Calc   
  ##############################################################
  def get_global_suff_stats( self, Data, SS, LP, Ntotal=None, Eflag=None, **kwargs ):
    ''' 
    '''
    SS['N'] = np.sum( LP['resp'], axis=0 )
    if Ntotal is not None:
      ampF = Ntotal/float(SS['N'].sum())
      SS['N'] = ampF*SS['N']
      SS['ampF'] = ampF
    SS['Ntotal'] = SS['N'].sum()

    if Eflag is not None:
      SS['Hz'] = np.sum( LP['resp'] * np.log(EPS+LP['resp']), axis=0 )
    return SS

  def inc_suff_stats( self, curID, SLP, SS):
    ''' Returns
        -------
        SS, kcur
    '''
    if SS is None:  
      SS = dict( N=np.zeros( self.K) )
    kcur = SLP['Z'][curID]
    SS['N'][ kcur ] += 1
    return SS, kcur
  
  def dec_suff_stats( self, curID, SLP, SS):
    '''
       Returns
       -------
       SS, kcur, delID (always None for finite mix model)
    '''
    if SS is None:  
      SS = dict( N=np.zeros( self.K) )
    kcur = SLP['Z'][curID]
    if kcur < 0:
      return SS, None, None
    assert SS['N'][kcur] > 0
    SS['N'][kcur] -= 1
    return SS, kcur, None

    
  ##############################################################    
  ############################################################## Local Param Updates   
  ##############################################################
  def calc_local_params( self, Data, LP ):
    ''' E-step
    '''
    if self.qType.count('VB') > 0:
      lpr = self.Elogw + LP['E_log_soft_ev']
    elif self.qType.count('EM') > 0:
      lpr = np.log(self.w) + LP['E_log_soft_ev']
    del LP['E_log_soft_ev']
    lprPerItem = logsumexp( lpr, axis=1 )
    resp   = np.exp( lpr-lprPerItem[:,np.newaxis] )
    LP['resp'] = resp
    assert np.allclose( resp.sum(axis=1), 1.0 )
    if self.qType == 'EM':
        LP['evidence'] = lprPerItem.sum()
    return LP
 
  ##############################################################    
  ############################################################## Sampling   
  ##############################################################
  def sample_local_from_post( self, SLP, curIDs, logSoftEvMat):
    '''
        Returns
        --------
        SLP, Kextra (always 0 for finite mix model)
    '''
    lpr = np.log(self.w) + logSoftEvMat
    lprPerItem = logsumexp( lpr, axis=1 )
    Pmat   = np.exp( lpr-lprPerItem[:,np.newaxis] )
    knew = discrete_single_draw_vectorized( Pmat )
    SLP['Z'][curIDs] = knew
    return SLP, 0

  def sample_from_pred_posterior( self, curID, SS, SLP, ps):
    '''
        Returns
        --------
        SLP, Kextra (always 0 for finite mix model)
    '''
    ps = ps * (SS['N']+self.alpha0)
    knew = discrete_single_draw( ps )
    SLP['Z'][curID] = knew
    return SLP, 0

  ##############################################################    
  ############################################################## Global Param Updates   
  ##############################################################
  def sample_global_params( self, SS, PRNG=np.random, **kwargs):
    self.w = PRNG.dirichlet( SS['N'] + self.alpha0 )

  def update_global_params_EM( self, SS, Krange=None, **kwargs):
    if self.alpha0 == 0:
      w = SS['N']
    else:
      w = SS['N'] + self.alpha0 - 1.0  # MAP estimate. Requires alpha0>1
    self.w = w/w.sum()
  
  def update_global_params_CGS(self, SS, Krange=None, **kwargs):
    self.alpha = self.alpha0 + SS['N']
    
  def update_global_params_VB( self, SS, Krange=None, **kwargs):
    self.alpha = self.alpha0 + SS['N']
    self.Elogw = digamma( self.alpha ) - digamma( self.alpha.sum() )

  def update_global_params_onlineVB( self, SS, rho, Krange=None, **kwargs):
    alphNew = self.alpha0 + SS['N']
    self.alpha = rho*alphNew + (1-rho)*self.alpha
    self.Elogw = digamma( self.alpha ) - digamma( self.alpha.sum() )
  
  ##############################################################    
  ############################################################## Evidence calc.   
  ##############################################################
  def calc_evidence( self, Data, SS, LP):
    if self.qType == 'EM':
      if np.allclose( self.alpha0, 0.0 ):
        return LP['evidence']
      else:
        return LP['evidence'] + self.log_pdf_dirichlet( self.w)
    if self.qType == 'CGS':
      return self.calc_log_marg_lik( SS, LP )
    elif self.qType.count('VB') >0:
      evW = self.E_logpW() - self.E_logqW()
      if 'Hz' in SS:
        evZq = self.E_logqZfast( SS)
      else:
        evZq = self.E_logqZ( LP )
      if 'ampF' in SS:
        evZ = self.E_logpZ( SS ) -  SS['ampF']*evZq
      else:
        evZ = self.E_logpZ( SS ) - evZq
      return evZ + evW
  
  def calc_log_marg_lik( self, SS, SLP):
    '''
        = p( SLP['Z'] | alpha0 ) 
        = \int p( 'Z' | theta) p( theta | alpha0 ) dtheta, theta ~ Dir(alpha0)
        = Phi_Dir( N + alpha0 ) / Phi_Dir( alpha0 ) 
    '''
    aPost = SS['N']+self.alpha0
    logPost = np.sum(gammaln(aPost)) - gammaln(np.sum(aPost))
    if self.isScalar:
      logPrior = self.K*gammaln(self.alpha0) - gammaln(self.K*self.alpha0)
    else:
      logPrior = np.sum( gammaln( self.alpha0)) - gammaln( np.sum(self.alpha0))
    return logPost - logPrior
             
  def E_logpZ( self, SS ):
    ''' Bishop PRML eq. 10.72
    '''
    return np.inner( SS['N'], self.Elogw )
    #return np.sum( LP['resp']* self.Elogw )

  def E_logqZfast( self, SS):
    return np.sum( SS['Hz'] )
    
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
    return logC + np.sum( (avec-1.0)*np.log(wvec) )
