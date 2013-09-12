''' AllocModel.py
'''

from IPython import embed
import numpy as np

from ..AllocModel import AllocModel

from bnpy.util import logsumexp, np2flatstr, flatstr2np
from bnpy.util import gammaln, digamma, EPS
from bnpy.util import discrete_single_draw

class DPMixModel( AllocModel ):

  def __init__(self, K=3, alpha0=5.0, truncType='z', qType='VB', Kmax=None, **kwargs):
    if qType.count('EM')>0:
      raise ValueError('DPMixModel cannot do EM. Only VB learning possible.')
    self.qType = qType
    self.K = K
    self.alpha1 = 1.0
    self.alpha0 = alpha0    
    self.truncType = truncType
    
    # q( v_k ) = Beta( qalpha1[k], qalpha0[k] )
    self.qalpha1 = np.zeros( K )
    self.qalpha0 = np.zeros( K )
    if qType.count('GS') > 0 and Kmax is not None:
      self.Kmax = Kmax
    
  def to_dict( self ):
    return dict( qalpha0=self.qalpha0, qalpha1=self.qalpha1 ) 
  
  def get_prior_dict( self ):  
      return dict( alpha0=self.alpha0, alpha1=self.alpha1, K=self.K, qType=self.qType)
  
  def from_dict( self, Dict ):
    self.qalpha1 = Dict['qalpha1']
    self.qalpha0 = Dict['qalpha0']
    self.set_helper_params()
 
  def get_info_string( self):
    ''' Returns human-readable name of this object
    '''
    return 'DP infinite mixture model with %d components. alpha0=%.2f' % (self.K, self.alpha0)

  def get_human_global_param_string(self):
    ''' Returns human-readable numerical repr. of parameters,
          for quick inspection of correctness
    '''
    return np2flatstr( np.exp(self.Elogw), '%3.2f' )
 
  def is_nonparametric(self):
    return True
    
  def set_helper_params( self ):
    DENOM = digamma( self.qalpha0 + self.qalpha1 )
    self.ElogV      = digamma( self.qalpha1 ) - DENOM
    self.Elog1mV    = digamma( self.qalpha0 ) - DENOM

    if self.truncType == 'v':
      self.qalpha1[-1] = 1
      self.qalpha0[-1] = EPS #avoid digamma(0), which is way too HUGE
      self.ElogV[-1] = 0  # log(1) => 0
      self.Elog1mV[-1] = np.log(1e-40) # log(0) => -INF, never used
		
		# Calculate expected mixture weights E[ log w_k ]	 
    self.Elogw = self.ElogV.copy() #copy so we can do += without modifying ElogV
    self.Elogw[1:] += self.Elog1mV[:-1].cumsum()
  
  ##############################################################    
  ############################################################## Remove component   
  ##############################################################
  def delete_components( self, keepIDs ):
    self.K = len(keepIDs)
    self.qalpha1 = self.qalpha1[keepIDs]  
    self.qalpha0 = self.qalpha0[keepIDs] 
    self.set_helper_params()
    
  ##############################################################    
  ############################################################## Suff Stat Calc   
  ##############################################################
  def get_global_suff_stats( self, Data, SS, LP, Krange=None, Ntotal=None, Eflag=False, Mflag=False ):
    ''' Just leave Krange alone for now.
    '''
    SS['N'] = np.sum( LP['resp'], axis=0 )
   
    if Ntotal is not None:
      ampF = Ntotal/SS['N'].sum()
      SS['N'] = ampF*SS['N']
      SS['ampF'] = ampF

    if Eflag:
      SS['Hz'] = np.sum( LP['resp'] * np.log(EPS+LP['resp']), axis=0 )

    if Mflag:
      SS['Hmerge'] = np.zeros( (self.K, self.K) )
      R = LP['resp']
      for jj in range(self.K):
        kkINDS = np.arange(jj+1, self.K)
        Rcombo = R[:,jj][:,np.newaxis]+R[:,kkINDS]
        SS['Hmerge'][jj,kkINDS] = np.sum( Rcombo*np.log(Rcombo+EPS), axis=0 )

    SS['Ntotal'] = SS['N'].sum()
    return SS
    
  def inc_suff_stats( self, curID, SLP, SS):
    if SS is None:  
      SS = dict( N=np.zeros( self.K) )
    ks = SLP['Z'][curID]
    SS['N'][ ks ] += 1
    return SS, ks
  
  def dec_suff_stats( self, curID, SLP, SS):
    if SS is None:  
      SS = dict( N=np.zeros( self.K) )
    ks = SLP['Z'][curID]
    if ks < 0:
      return SS, None, None
    elif SS['N'][ks] == 0:
      raise ValueError
    SS['N'][ ks ] -= 1
    if SS['N'][ks] > 0:
      return SS, ks, None
    else:
      #only delete if we have more than K=2 comps
      return SS, ks, ks


  ##############################################################    
  ############################################################## Local Param Updates   
  ##############################################################
  def calc_local_params( self, Data, LP, Krange=None ):
    ''' 
    '''
    lpr = self.Elogw + LP['E_log_soft_ev']
    del LP['E_log_soft_ev']
    if Krange is None:
      lprPerItem = logsumexp( lpr, axis=1 )
      resp   = np.exp( lpr-lprPerItem[:,np.newaxis] ) 
    else:
      lprPerItem = logsumexp( lpr[:,Krange], axis=1 )
      resp   = np.zeros( lpr.shape )
      resp[:,Krange]   = np.exp( lpr[:,Krange]-lprPerItem[:,np.newaxis] ) 
  
    LP['resp'] = resp
    return LP
 
  ##############################################################    
  ############################################################## Evidence calc.   
  ##############################################################
  def calc_evidence( self, Data, SS, LP=None ):
    if self.qType == 'CGS':
      self.qalpha1 = self.alpha1 + SS['N']
      self.qalpha0 = self.alpha0*np.ones( self.K )
      self.qalpha0[:-1] += SS['N'][::-1].cumsum()[::-1][1:]
      return self.calc_log_marg_lik( SS, LP )
    evV = self.E_logpV() - self.E_logqV()
    if 'Hz' in SS:
      evZq = self.E_logqZfast( SS)      
    else:
      evZq = self.E_logqZ( LP )
    if 'ampF' in SS:
      evZ = self.E_logpZ( SS ) -  SS['ampF']*evZq
    else:
      evZ = self.E_logpZ( SS ) - evZq
    return evZ + evV
  
  def calc_log_marg_lik( self, SS, SLP):
    ''' marginal likelihood of CRP partition
    '''
    Nvec = SS['N'][ SS['N'] > 0 ]
    K = len(Nvec)
    lp = K* np.log( self.alpha0 ) + gammaln( self.alpha0)
    lp -= gammaln( self.alpha0 + Nvec.sum() )
    lp += np.sum( gammaln(Nvec) )
    return lp
         
  def E_logpZ( self, SS):
    '''
      E[ log p( Z | V ) ] = \sum_n E[ log p( Z[n] | V )
         = \sum_n E[ log p( Z[n]=k | w(V) ) ]
         = \sum_n \sum_k z_nk log w(V)_k
      WARNING: Naively putting SS here instead of LP lead to big troubles
    '''
    return np.inner( SS['N'], self.Elogw ) 
    #return np.inner( np.sum(LP['resp'],axis=0), self.Elogw ) 
    
  def E_logqZ( self, LP ):
    return np.sum( LP['resp'] *np.log(LP['resp']+EPS) )
    
  def E_logqZfast( self, SS):
    if 'Hz_adjust' in SS:
      return np.sum( SS['Hz'] ) + np.sum( [v for v in SS['Hz_adjust'].values()] )
    else:
      return np.sum( SS['Hz'] )

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
    
  ##############################################################    
  ############################################################## Sampling   
  ##############################################################
  def sample_from_pred_posterior( self, curID, SS, SLP, ps):
    '''
    '''
    assert SS['N'][-1] == 0
    if SS['N'].sum() == 0:
      knew = 0 # always choose first component
    else:
      ps[:self.K-1] *= SS['N'][:self.K-1]
      if hasattr(self,'Kmax') and self.K == self.Kmax+1:
        ps[-1] = 0
      else:
        ps[-1] *= self.alpha0
      knew = discrete_single_draw( ps )
    SLP['Z'][curID] = knew
    Kextra = int( knew == self.K-1 )
    return SLP, Kextra
 
  ##############################################################    
  ############################################################## Param Update   
  ##############################################################
  def update_global_params_CGS( self, SS, Krange=None, **kwargs):
    qalpha1 = self.alpha1 + SS['N']
    qalpha0 = self.alpha0*np.ones( self.K )
    qalpha0[:-1] += SS['N'][::-1].cumsum()[::-1][1:]
    self.qalpha0 = qalpha0
    self.qalpha1 = qalpha1

    
  def update_global_params_VB( self, SS,  Krange=None,  **kwargs ):
    '''  Updates internal stick breaking weights given suff. stats
         Can optionally only update a given set of indices "Krange",
           though because of the dependent ordering of the sticks,
           it is not correct to just pick out indices at random.  
         Instead, we must update all indices from the lowest entry in Krange on up
    '''
    assert self.K == SS['N'].size
    qalpha1 = self.alpha1 + SS['N']
    qalpha0 = self.alpha0*np.ones( self.K )
    qalpha0[:-1] += SS['N'][::-1].cumsum()[::-1][1:]
    if Krange is None:
      self.qalpha1 = qalpha1
      self.qalpha0 = qalpha0
    else:
      self.qalpha1 = qalpha1
      self.qalpha0 = qalpha0
      '''
      kmin = np.min(Krange)-1
      if kmin == -1:
        self.qalpha1 = qalpha1
        self.qalpha0 = qalpha0
      else:
        Kextra = np.max(Krange)+1-self.qalpha1.size
        if Kextra > 0:
          arrExtra = np.zeros( Kextra)
          self.qalpha1 = np.append( self.qalpha1, arrExtra )
          self.qalpha0 = np.append( self.qalpha0, arrExtra )
        self.qalpha1[ kmin: ] = qalpha1[ kmin: ]
        self.qalpha0[ kmin: ] = qalpha0[ kmin: ]
      '''
    self.set_helper_params()
    
  def update_global_params_onlineVB( self, SS,  rho, Krange=None,  **kwargs ):
    '''
    '''
    assert self.K == SS['N'].size
    qalpha1 = self.alpha1 + SS['N']
    qalpha0 = self.alpha0*np.ones( self.K )
    qalpha0[:-1] += SS['N'][::-1].cumsum()[::-1][1:]
    
    if Krange is None:
      self.qalpha1 = rho*qalpha1 + (1-rho)*self.qalpha1
      self.qalpha0 = rho*qalpha0 + (1-rho)*self.qalpha0
    else:
      print Krange
      raise ValueError('TODO: Not yet implemented')      
    self.set_helper_params()
