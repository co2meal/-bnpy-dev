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

class SBM(AllocModel):
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
      return dict( PostBeta=self.PostBeta)
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
              sigmas : N x K array with posterior responsibilities over source
              rhos : N x K array with posterior responsibilities over receiver
        doPrecompEntropy : boolean flag that indicates whether to precompute the entropy of the data responsibilities (used for evaluating the evidence)

        Returns
        -------
        SS : SuffStatDict with K components, with field
              N : K-len vector of effective number of observations assigned to each comp
    '''

    sigmas = LP['sigmas']
    #rhos = LP['rhos']
    K,N = sigmas.shape 
    SS = SuffStatDict(K,N)
    # sufficient statistic for mixture component weights
    #SS['u'] = np.sum( sigmas, 1) 
    SS['u'] = np.sum( sigmas, 1) 
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
              E_log_soft_ev : Data.nObs x K x K array
                  E_log_soft_ev[e,k,l] = log p(data obs n | comp k,l)
        
        Returns
        -------
        LP : local param dict with fields
              sigmas : Data.nObs x K array whose rows sum to one
              rhos : Data.nObs x K arrawy            
    '''
    K = self.K
    N = Data.N
    if self.inferType.count('VB') > 0:
        #actual e-step check/ if rho exists, otherwise initialize that randomly
        
        '''
        if 'rhos' not in LP:
            rhos = np.zeros((K, N)) + 1
            for i in xrange(N):
                k = np.round(np.random.rand()*K)-1
                rhos[k,i] += 10.0
                rhos[:,i] = rhos[:,i] / rhos[:,i].sum()
        else:
            rhos = LP['rhos']
        '''
        sigmas = np.zeros((K,N))
        temp_s = np.zeros((K,N))
        #temp_r = np.zeros((K,N))
        X = Data.X
        E_log_pdf = LP['E_log_soft_ev']
        
        old_sigmas = sigmas
        #old_rhos = rhos
        for i in xrange(N):
            ind_i = [ii for ii, x in enumerate(X[:,0]) if x == i]
            ind_j = [jj for jj, x in enumerate(X[:,1]) if x == i]
            isconverged = False
            while not isconverged:
                # Learn Sigmas            
                for e in xrange(len(ind_i)):
                    j = X[ind_i[e],1]
                    temp_s[:,i] += np.dot( E_log_pdf[ind_i[e],:,:], sigmas[:,j]  )

                for e in xrange(len(ind_j)):
                    j = X[ind_j[e],0]
                    temp_s[:,i] += np.dot( sigmas[:,j] , E_log_pdf[ind_j[e],:,:])
                
                temp_s[:,i] += self.ElogBeta
                sigmas[:,i] = np.exp(temp_s[:,i] - logsumexp(temp_s[:,i]))
                # Learn RhosS
                '''
                for e in xrange(len(ind_j)):
                    j = X[ind_j[e],0]
                    temp_r[:,i] = np.dot(  E_log_pdf[ind_j[e],:,:], sigmas[:,j]  )
                temp_r[:,i] += self.ElogBeta
                rhos[:,i] = np.exp(temp_r[:,i]-logsumexp(temp_r[:,i]))
                '''
                #diff = np.absolute(sigmas[:,i]-old_sigmas[:,i]) + np.absolute(rhos[:,i]-old_rhos[:,i])

                #diff = np.absolute(sigmas[:,i]-old_sigmas[:,i]) 
                #if diff.sum() <= 1e-3:
                isconverged = True
            old_sigmas[:,i] = sigmas[:,i]
            #old_rhos[:,i] = rhos[:,i]
        
        LP['sigmas'] = sigmas
        #LP['rhos'] = rhos
    elif self.inferType == 'EM' > 0:
      pass
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
    self.PostBeta = self.alpha0 + SS.u
    self.ElogBeta = digamma( self.PostBeta ) - digamma( self.PostBeta.sum() )
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
      # E[p(s|beta)] + E[p(r_beta)] - q(beta)
      ps = np.dot(self.ElogBeta,LP["sigmas"]).sum()
      #pr = np.dot(self.ElogBeta,LP["rhos"]).sum()
      pB = gammaln(self.K*self.alpha0) - gammaln(self.alpha0)*self.K + ((self.alpha0-1)*self.ElogBeta).sum()
      qBeta = gammaln(SS.u.sum()) - gammaln(SS.u).sum() + ((SS.u-1)*self.ElogBeta).sum()
      #elbo_alloc = ps + pr + pB - qBeta
      elbo_alloc = ps + pB - qBeta
      return elbo_alloc
      
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