'''
HDPModel2.py
Bayesian nonparametric admixture model with unbounded number of components K

Attributes
-------
K        : # of components
alpha0   : scalar concentration param for global-level stick-breaking params v
gamma    : scalar conc. param for document-level mixture weights pi[d]

Local Parameters (document-specific)
--------
theta : nDoc x K matrix, 
             row d has params for doc d's distribution pi[d] over the K topics
             q( pi[d] ) ~ Dir( theta[d] )
E_logPi : nDoc x K matrix
             row d has E[ log pi[d] ]
DocTopicCount : nDoc x K matrix
                  entry d,k gives the expected number of times
                              that topic k is used in document d
word_variational : nDistinctWords x K matrix
                  row i has params for word i's Discrete distr over K topics

Global Random Variables
-------------
v : 1D array, length K, 0 < v[k] < 1
    stick-breaking fractions

Global Variational Parameters
--------
q(v[k]) ~ Beta(rho[k]*omega[k], (1-rho[k]) *omega[k])
rho : 1D array, length K,  0 < rho[k] < 1
      rho[k] determines the mean of v[k]
omega : 1D array, length K, 0 < omega[k]
        omega[k] determines the variance of q(v[k])
'''
import numpy as np
import logging
Log = logging.getLogger('bnpy')

import LocalStepBagOfWords
import OptimizerForHDP2 as OptimHDP2
from HDPModel import HDPModel
from bnpy.suffstats import SuffStatBag
from ...util import NumericUtil
from ...util import digamma, gammaln
from ...util import EPS

class HDPModel2(HDPModel):

  ######################################################### Constructors
  #########################################################
  def __init__(self, inferType, priorDict=None):
    if inferType == "EM":
      raise ValueError('HDPModel cannot do EM. Only VB possible.')
    self.inferType = inferType
    self.K = 0
    self.set_prior(priorDict)
    
  def set_helper_params(self):
    ''' Set dependent attribs of this model, given the primary params U1, U0
        This includes expectations of various stickbreaking quantities
    '''
    assert self.rho.size == self.K
    assert self.omega.size == self.K
    self.Ebeta = OptimHDP2._v2beta(self.rho)
    digammaomega = digamma(self.omega)
    self.Elogv = digamma(self.rho * self.omega) - digammaomega
    self.Elog1mv = digamma((1-self.rho) * self.omega) - digammaomega

  ######################################################### Accessors
  #########################################################
  # Inherited from HDPModel


  ######################################################### Local Params
  #########################################################
  # Inherited from HDPModel

  ######################################################### Suff Stats
  #########################################################
  # Inherited from HDPModel

  ######################################################### Global Params
  #########################################################

  def update_global_params_VB(self, SS, comps=None, **kwargs):
    ''' Update global parameters that control topic probabilities
    '''
    self.K = SS.K
    rho, omega = self._find_optimum_rhoomega(SS, **kwargs)
    if comps is not None and hasattr(self, 'rho'):
      self.rho[comps] = rho[comps]
      self.omega[comps] = omega[comps]
      # TODO: when does this happen???
      if self.rho.size > self.K:
        self.rho = self.rho[:self.K]
        self.omega = self.omega[:self.K]
    else:
      self.rho = rho
      self.omega = omega
    self.set_helper_params()
        
  def update_global_params_soVB(self, SS, rho, **kwargs):
    raise NotImplementedError('TODO')

  def _find_optimum_rhoomega(self, SS, mergeCompB=None, **kwargs):
    ''' 
        Returns
        --------
        rho : 1D array, length K
        omega : 1D array, length K
    '''

    if hasattr(self, 'rho') and self.rho.size == self.K:
      initrho = self.rho
      initomega = self.omega
    elif hasattr(self, 'rho') and self.rho.size == self.K + 1 \
                              and mergeCompB is not None:
      initrho = np.delete(self.rho, mergeCompB)
      initomega = np.delete(self.omega, mergeCompB)
    else:
      # Uses default initialization in OptimizerForHDP2
      initrho = None 
      initomega = None 

    sumLogPi = np.hstack([SS.sumLogPiActive, SS.sumLogPiUnused])

    try:
      rho, omega, f, Info = OptimHDP2.find_optimum_multiple_tries(
                                        sumLogPi=sumLogPi,
                                        nDoc=SS.nDoc,
                                        gamma=self.gamma, alpha=self.alpha0,
                                        initrho=initrho, initomega=initomega)
    except ValueError as error:
      if str(error).count('FAILURE') == 0:
        raise error
      if hasattr(self, 'rho') and self.rho.size == self.K:
        Log.error('***** Optim failed. Remain at cur val.' + str(error))
        rho = self.rho
        omega = self.omega
      else:
        Log.error('***** Optim failed. Set to prior. ' + str(error))
        omega = (self.alpha0 + 1 ) * np.ones(SS.K)
        rho = 1/float(1+self.alpha0) * np.ones(SS.K)
    return rho, omega

  def set_global_params(self, hmodel=None, 
                              rho=None, omega=None, 
                              **kwargs):
    if hmodel is not None:
      self.K = hmodel.allocModel.K
      self.rho = hmodel.allocModel.rho
      self.omega = hmodel.allocModel.omega
      self.set_helper_params()
    elif rho is not None and omega is not None:
      self.rho = rho
      self.omega = omega
      self.K = omega.size
      self.set_helper_params()
    else:
      self._set_global_params_from_scratch(**kwargs)

  def _set_global_params_from_scratch(self, beta=None, nDoc=None,
                              Ebeta=None, EbetaLeftover=None, **kwargs):
    if Ebeta is not None and EbetaLeftover is not None:
      beta = np.hstack([np.squeeze(Ebeta), np.squeeze(EbetaLeftover)])
          
    elif beta is not None:
      beta = np.hstack([np.squeeze(beta), np.min(beta)/100.])
      beta = beta/np.sum(beta)
    else:
      raise ValueError('Bad parameters. Vector beta not specified.')
    self.K = beta.size - 1
    
    # Now, use the specified value of beta to find the best U1, U0
    self.rho, self.omega = self._convert_beta2rhoomega(beta, nDoc)
    assert self.rho.size == self.K
    assert self.omega.size == self.K
    self.set_helper_params()

  def _convert_beta2rhoomega(self, beta, nDoc=None):
    ''' Given a vector beta (size K+1),
          return educated guess for vectors rho, omega

        Returns
        --------
        rho : 1D array, size K
        omega : 1D array, size K
    '''
    assert abs(np.sum(beta) - 1.0) < 0.001
    rho = OptimHDP2._beta2v(beta)
    K = rho.size
    if nDoc is not None:
      omega = nDoc * np.ones(K)
    else:
      omega = 10 * np.ones(K)
    return rho, omega    
    


  ######################################################### Evidence
  #########################################################  
  def calc_evidence( self, Data, SS, LP, todict=False):
    ''' Calculate ELBO terms related to allocation model
    '''   
    E_logpV = self.E_logpV()
    E_logqV = self.E_logqV()
     
    E_logpPi = self.E_logpPi(SS)
    if SS.hasELBOTerms():
      E_logqPi = (SS.getELBOTerm('ElogqPiConst')
                  + SS.getELBOTerm('ElogqPiUnused')
                  + SS.getELBOTerm('ElogqPiActive').sum())
      E_logpZ = np.sum(SS.getELBOTerm('ElogpZ'))
      E_logqZ = np.sum(SS.getELBOTerm('ElogqZ'))
    else:
      E_logqPi = self.E_logqPi(LP)
      E_logpZ = np.sum(self.E_logpZ(Data, LP))
      E_logqZ = np.sum(self.E_logqZ(Data, LP))

    if SS.hasAmpFactor():
      E_logqPi *= SS.ampF
      E_logpZ *= SS.ampF
      E_logqZ *= SS.ampF

    if todict:
      return dict(pi_Elogp=E_logpPi, pi_Elogq=E_logqPi,
                  z_Elogp=E_logpZ, z_Elogq=E_logqZ,
                  v_Elogp=E_logpV, v_Elogq=E_logqV)

    elbo = ((E_logpPi - E_logqPi)
            + (E_logpZ - E_logqZ)
            + (E_logpV - E_logqV))
    return elbo

  ####################################################### ELBO terms for Z
  def E_logpZ( self, Data, LP):
    ''' Returns K-length vector with E[log p(Z)] for each topic k
    '''
    return np.sum(LP['DocTopicCount'] * LP['E_logPi'], axis=0)

  def E_logqZ( self, Data, LP):  
    ''' Returns K-length vector with E[ log q(Z) ] for each topic k
            r_{dwk} * E[ log r_{dwk} ]
        where z_{dw} ~ Discrete( r_dw1 , r_dw2, ... r_dwK )
    '''
    wv = LP['word_variational']
    wv += EPS # Make sure all entries > 0 before taking log
    return NumericUtil.calcRlogRdotv(wv, Data.word_count)

  def E_logqZ_memo_terms_for_merge(self, Data, LP, mPairIDs=None):
    ''' Returns KxK matrix 
    ''' 
    wv = LP['word_variational']
    wv += EPS # Make sure all entries > 0 before taking log
    if mPairIDs is None:
      ElogqZMat = NumericUtil.calcRlogRdotv_allpairs(wv, Data.word_count)
    else:
      ElogqZMat = NumericUtil.calcRlogRdotv_specificpairs(wv, 
                                                Data.word_count, mPairIDs)
    return ElogqZMat

  ####################################################### ELBO terms for Pi
  def E_logpPi(self, SS):
    ''' Returns scalar value of E[ log p(PI | alpha0)]
    '''
    K = SS.K
    kvec = K + 1 - np.arange(1, K+1)
    # logDirNormC : scalar norm const that applies to each iid draw pi_d
    logDirNormC = gammaln(self.gamma) + (K+1) * np.log(self.gamma)
    logDirNormC += np.sum(self.Elogv) + np.inner(kvec, self.Elog1mv)
    # logDirPDF : scalar sum over all doc's pi_d
    sumLogPi = np.hstack([SS.sumLogPiActive, SS.sumLogPiUnused])
    logDirPDF = np.inner(self.gamma * self.Ebeta - 1., sumLogPi)
    return (SS.nDoc * logDirNormC) + logDirPDF

  def E_logqPi(self, LP):
    ''' Returns scalar value of E[ log q(PI)],
          calculated directly from local param dict LP
    '''
    theta = LP['theta']    # nDoc x K
    utheta = LP['theta_u'] # scalar
    logDirNormC = gammaln(utheta + theta.sum(axis=1)) \
                  - (gammaln(utheta) + np.sum(gammaln(theta), axis=1))
    logDirPDF = np.sum((theta - 1.) * LP['E_logPi']) \
                  + (utheta - 1.) * np.sum(LP['E_logPi_u'])
    return np.sum(logDirNormC) + logDirPDF

  def E_logqPi_Memoized_from_LP(self, LP):
    ''' Returns three variables 
                logDirNormC (scalar),
                logqPiActive (length K)
                logqPiUnused (scalar)
                whose sum is equal to E[log q(PI)]
        when added to other results of this function from different batches,
                the sum is equal to E[log q(PI)] of the entire dataset
    '''
    theta = LP['theta']    # nDoc x K
    utheta = LP['theta_u'] # scalar
    nDoc = theta.shape[0]
    logDirNormC = np.sum(gammaln(utheta + theta.sum(axis=1)))
    piEntropyVec = np.sum( (theta - 1.0) * LP['E_logPi'], axis=0) \
                     - np.sum(gammaln(theta),axis=0)
    piEntropyUnused = (utheta - 1.0) * np.sum( LP['E_logPi_u'], axis=0) \
                     - nDoc * gammaln(utheta) 
    return logDirNormC, piEntropyVec, piEntropyUnused


  ####################################################### ELBO terms for V
  def E_logpV(self):
    logBetaNormC = gammaln(self.alpha0 + 1.) \
                      - gammaln(self.alpha0)
    logBetaPDF = (self.alpha0-1.) * np.sum(self.Elog1mv)
    return self.K*logBetaNormC + logBetaPDF

  def E_logqV(self):
    u1 = self.rho * self.omega
    u0 = (1-self.rho) * self.omega
    logBetaNormC = gammaln(self.omega) - gammaln(u1) - gammaln(u0)
    logBetaPDF = np.inner(u1 - 1., self.Elogv) \
                 + np.inner(u0 - 1., self.Elog1mv)
    return np.sum(logBetaNormC) + logBetaPDF

  ####################################################### ELBO terms merge
  def memo_elbo_terms_for_merge(self, LP):
    ''' Calculate some ELBO terms for merge proposals for current batch

        Returns
        --------
        ElogpZMat   : KxK matrix
        sumLogPiMat : KxK matrix
        ElogqPiMat  : KxK matrix
    '''
    CMat = LP['DocTopicCount']# nDoc x K
    theta = LP['theta']       # nDoc x K
    digammasumTheta = digamma(LP['theta_u'] + theta.sum(axis=1))[:, np.newaxis]

    ElogpZMat = np.zeros((self.K, self.K))
    sumLogPiMat = np.zeros((self.K, self.K))
    ElogqPiMat = np.zeros((self.K, self.K))
    for jj in range(self.K):
      M = self.K - jj - 1
      # nDoc x M matrix, theta_{dm} for each merge pair m with comp jj
      mergeTheta = theta[:,jj][:,np.newaxis] + theta[:, jj+1:]
      # nDoc x M matrix, E[log pi_m] for each merge pair m with comp jj
      mergeElogPi = digamma(mergeTheta) - digammasumTheta
      assert mergeElogPi.shape[1] == M
      # nDoc x M matrix, count for merged topic m each doc
      mergeCMat = CMat[:, jj][:,np.newaxis] + CMat[:, jj+1:]
      ElogpZMat[jj, jj+1:] = np.sum(mergeCMat * mergeElogPi, axis=0)
          
      sumLogPiMat[jj, jj+1:] = np.sum(mergeElogPi,axis=0)
      curElogqPiMat = np.sum((mergeTheta-1.) * mergeElogPi, axis=0) \
                       - np.sum(gammaln(mergeTheta),axis=0)
      assert curElogqPiMat.size == M
      ElogqPiMat[jj, jj+1:] = curElogqPiMat

    return ElogpZMat, sumLogPiMat, ElogqPiMat

  ######################################################### IO Utils
  #########################################################   for humans
    
  def get_info_string( self):
    ''' Returns human-readable name of this object'''
    s = 'HDP admixture model with K=%d comps. alpha=%.2f, gamma=%.2f'
    return s % (self.K, self.alpha0, self.gamma)
     
  ######################################################### IO Utils
  #########################################################   for machines
  def to_dict( self ):
    return dict(rho=self.rho, omega=self.omega)              
  
  def from_dict(self, Dict):
    self.inferType = Dict['inferType']
    self.K = Dict['K']
    self.rho = Dict['rho']
    self.omega = Dict['omega']
    self.set_helper_params()

  def get_prior_dict( self ):
    return dict(K=self.K, alpha0=self.alpha0, gamma=self.gamma)

