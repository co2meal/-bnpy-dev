'''
HDPPE.py
Bayesian nonparametric admixture model with unbounded number of components K

Global Parameters (shared across all documents)
--------
v   : K-length vector, point estimate for stickbreaking fractions v1, v2, ... vK
'''
import numpy as np

import OptimizerForHDPPE as OptPE
from .HDPModel import HDPModel
from ...util import gammaln

class HDPPE(HDPModel):

  ######################################################### Constructors
  #########################################################
  ''' Constructor handled by HDPModel
  '''
        
  def set_helper_params(self):
    self.Ebeta = OptPE._v2beta(self.v)

  ######################################################### Local Params
  #########################################################
  ''' Handled by HDPModel
  '''

  ######################################################### Suff Stats
  #########################################################
  ''' Handled by HDPModel
  '''

  ######################################################### Global Params
  #########################################################
  def update_global_params_VB(self, SS, **kwargs):
    ''' Update global parameters v that control topic probabilities beta
    '''
    self.K = SS.K
    v = self._find_optimal_v(SS)
    self.v = v
    assert self.v.size == self.K
    self.set_helper_params()          
  
  def update_global_params_soVB(self, SS, rho, **kwargs):
    ''' Stochastic online update for global parameters v
    '''
    raise NotImplementedError("TODO")
        
  def _find_optimal_v(self, SS):
    ''' Find optimal vector v via gradient descent
    '''
    sumLogPi = np.hstack([SS.sumLogPiActive, SS.sumLogPiUnused])
    assert sumLogPi.size == SS.K + 1

    if hasattr(self, 'v') and self.v.size == SS.K:
      initv = self.v.copy()
    else:
      initv = None
    try:
      v, f, Info = OptPE.find_optimum_multiple_tries(sumLogPi, SS.nDoc,
                                                     gamma=self.gamma,
                                                     alpha=self.alpha0,
                                                     initv=initv,
                                                     approx_grad=False)
    except ValueError as error:
      if str(error).count('FAILURE') == 0:
        raise error
      if hasattr(self, 'v') and self.v.size == self.K:
        Log.error('***** Optim failed. Remain at cur val.' + str(error))
        v = self.v
      else:
        Log.error('***** Optim failed. Set to uniform. ' + str(error))
        v = OptPE.create_initv(self.K)
    return v

  def set_global_params(self, hmodel=None, rho=None, v=None,
                              **kwargs):
    ''' Set global parameter v to provided value
    '''
    if rho is not None:
      v = rho
    if hmodel is not None:
      self.K = hmodel.allocModel.K
      self.v = hmodel.allocModel.v
      self.set_helper_params()
    elif v is not None:
      self.v = v
      self.K = v.size
      self.set_helper_params()
    else:
      self._set_global_params_from_scratch(**kwargs)

  def _set_global_params_from_scratch(self, beta=None,
                              Ebeta=None, EbetaLeftover=None, **kwargs):
    ''' Set global parameter v to match provided topic distribution
    '''
    if Ebeta is not None and EbetaLeftover is not None:
      beta = np.hstack([np.squeeze(Ebeta), np.squeeze(EbetaLeftover)])          
    elif beta is not None:
      K = beta.size
      rem = np.minimum(0.1, 1.0/(3*K))
      beta = np.hstack([np.squeeze(beta), rem])
      beta = beta/np.sum(beta)
    else:
      raise ValueError('Vector beta not specified.')
    # Convert specified beta to v
    self.v = OptPE._beta2v(beta)
    self.K = beta.size - 1    
    assert self.v.size == self.K
    self.set_helper_params()

  ######################################################### Evidence
  #########################################################  
  ''' Inherited from HDPModel
  '''

  ####################################################### ELBO terms for Z
  ''' Inherited from HDPModel
  '''
  
  ####################################################### ELBO terms for Pi
  def E_logpPi(self, SS):
    ''' Returns scalar value of E[ log p(PI | alpha0)]
    '''
    K = SS.K
    # logDirNormC : scalar norm const that applies to each iid draw pi_d
    logDirNormC = gammaln(self.gamma) - np.sum(gammaln(self.gamma*self.Ebeta))
    # logDirPDF : scalar sum over all doc's pi_d
    sumLogPi = np.hstack([SS.sumLogPiActive, SS.sumLogPiUnused])
    logDirPDF = np.inner(self.gamma * self.Ebeta - 1., sumLogPi)
    return (SS.nDoc * logDirNormC) + logDirPDF

  ####################################################### ELBO terms for V
  def E_logpV(self):
    logBetaNormC = gammaln(self.alpha0 + 1.) - gammaln(self.alpha0)
    logBetaPDF = (self.alpha0-1.) * np.sum(np.log(1-self.v))
    return self.K*logBetaNormC + logBetaPDF

  def E_logqV(self):
    ''' Returns entropy of q(v), which for a point estimate is always 0
    '''
    return 0

  ####################################################### ELBO terms merge
  ''' Inherited from HDPModel
  '''

  ######################################################### IO Utils
  #########################################################   for humans
  def get_info_string( self):
    ''' Returns human-readable name of this object
    '''
    s = 'HDP model. K=%d, alpha=%.2f, gamma=%.2f. Point estimates v.'
    return s % (self.K, self.alpha0, self.gamma)
     
  ######################################################### IO Utils
  #########################################################   for machines
  def to_dict( self ):
    return dict(v=self.v)              
  
  def from_dict(self, Dict):
    self.inferType = Dict['inferType']
    self.v = np.squeeze(np.asarray(Dict['v'], dtype=np.float64))
    self.K = self.v.size
    self.set_helper_params()

  def get_prior_dict( self ):
    return dict(K=self.K, alpha0=self.alpha0, gamma=self.gamma)

