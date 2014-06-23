'''
HDPPE.py
Bayesian nonparametric admixture model with unbounded number of components K

Global Parameters (shared across all documents)
--------
v   : K-length vector, point estimate for stickbreaking fractions v1, v2, ... vK
'''
import numpy as np

import OptimizerForHDPPE as OptPE
from HDPModel import HDPModel
from bnpy.suffstats import SuffStatBag
from bnpy.util import NumericUtil
from bnpy.util import EPS, digamma, gammaln

class HDPPE2(HDPModel):

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
  def get_global_suff_stats(self, Data, LP, doPrecompEntropy=False,
                                            doPrecompMergeEntropy=False,
                                            mPairIDs=None,
                                            preselectroutine=None):
    ''' Calculate sufficient statistics
    '''
    wv = LP['word_variational'] # nDistinctWords x K
    _, K = wv.shape
    SS = SuffStatBag(K=K, D=Data.vocab_size)
    SS.setField('nDoc', Data.nDoc, dims=None)
    sumLogPi = np.sum(LP['E_logPi'], axis=0)
    SS.setField('sumLogPiActive', sumLogPi, dims='K')
    SS.setField('sumLogPiUnused', np.sum(LP['E_logPi_u'], axis=0), dims=None)

    ## Special field for better inserting empty components
    if 'digammasumTheta' in LP:
      SS.setField('sumDigammaSumTheta', np.sum(LP['digammasumTheta']),
                                        dims=None)

    if doPrecompEntropy:
      # Z terms
      SS.setELBOTerm('ElogqZ', self.E_logqZ(Data, LP), dims='K')
      # Pi terms
      # Note: no terms needed for ElogpPI
      # SS already has field sumLogPi, which is sufficient for this term
      ElogqPiC, ElogqPiA, ElogqPiU = self.E_logqPi__memoized(LP)
      SS.setELBOTerm('ElogqPiConst', ElogqPiC, dims=None)
      SS.setELBOTerm('ElogqPiActive', ElogqPiA, dims='K')
      SS.setELBOTerm('ElogqPiUnused', ElogqPiU, dims=None)

    if doPrecompMergeEntropy:
      raise NotImplementedError('TODO')

    if preselectroutine is not None:
      if preselectroutine.count('doctopiccorr') > 0:
        Tmat = LP['DocTopicCount']
        SS.setSelectionTerm('DocTopicPairMat',
                           np.dot(Tmat.T, Tmat), dims=('K','K'))
        SS.setSelectionTerm('DocTopicSum', np.sum(Tmat, axis=0), dims='K')
    return SS

  def calcSuffStatAdjustments(self, SS, Kextra):
    '''
        Returns
        -------
        AdjustInfo
        ReplaceInfo
    ''' 

    AdjustInfo = dict()
    ReplaceInfo = dict()
    Korig = SS.K

    ## Calculate corrected theta terms for new components K+1,K+2,... K+Kfresh
    ## Essentially, divide up the existing "left-over" mass among Kfresh comps
    ##   using the stick-breaking prior as the division procedure
    remEbeta = float(self.Ebeta[-1])
    newEbeta = np.zeros(Kextra)
    for k in xrange(Kextra):
      newEbeta[k] = 1.0/(1.0+self.alpha0) * remEbeta
      remEbeta = remEbeta - newEbeta[k]
    assert np.allclose(np.sum(newEbeta) + remEbeta, self.Ebeta[-1])
    newTheta = self.gamma * newEbeta
    newTheta_u = self.gamma * remEbeta

    ## Determine "adjustment" that will occur for each SS field
    ##   on a per-document basis document 
    ##   to create a sensible expanded theta parameter
    nDoc = SS.nDoc
    sumDigammaSumTheta = SS.sumDigammaSumTheta
    AdjustInfo['sumLogPiActive'] = np.zeros(SS.K + Kextra)
    AdjustInfo['sumLogPiActive'][-Kextra:] = digamma(newTheta) \
                                              - sumDigammaSumTheta / nDoc
    ReplaceInfo['sumLogPiUnused'] = digamma(newTheta_u) \
                                              - sumDigammaSumTheta / nDoc

    # Calculate adjustments to ELBO terms
    qPiActive = -1 * gammaln(newTheta)
    AdjustInfo['ElogqPiActive'] = np.hstack([np.zeros(Korig), qPiActive])
    qPiUnused = -1 * - gammaln(newTheta_u)
    ReplaceInfo['ElogqPiUnused'] = qPiUnused

    return AdjustInfo, ReplaceInfo


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
  def calc_evidence(self, Data=None, SS=None, LP=None, todict=False):
    ''' Calculate ELBO (evidence lower bound) objective
    '''
    E_logpV = self.E_logpV()
    # E_logqV = 0
      
    E_logpPi = self.E_logpPi(SS)
    if SS.hasELBOTerms():
      E_logqPi = SS.getELBOTerm('ElogqPiConst') \
                 + SS.getELBOTerm('ElogqPiUnused') \
                 + SS.getELBOTerm('ElogqPiActive').sum()
      E_logqZ = SS.getELBOTerm('ElogqZ').sum()
    else:
      E_logqPi = self.E_logqPi(LP)
      E_logqZ = self.E_logqZ(Data, LP).sum()

    if SS.hasAmpFactor():
      E_logqPi *= SS.ampF
      E_logqZ *= SS.ampF

    if todict:
      return dict(pi_Elogp=E_logpPi, pi_Elogq=E_logqPi,
                  v_Elogp=E_logpV, z_Elogp=0,
                  v_Elogq=0,       z_Elogq=E_logqZ)
    
    elbo = E_logpV + E_logpPi - E_logqPi - E_logqZ
    return elbo

  ####################################################### ELBO terms for Z
  def E_logpZ(self):
    '''
    '''
    return 0

  def E_logqZ(self, Data, LP):  
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
    ''' Returns scalar value of \sum_{d=1}^D E[log p(pi_d)]
    '''
    logDirNormC = gammaln(self.gamma) - np.sum(gammaln(self.gamma*self.Ebeta))
    return SS.nDoc * logDirNormC

  def E_logqPi(self, LP):
    ''' Returns scalar value of \sum_{d=1}^D E[log q(pi_d)]
    '''
    theta = LP['theta']    # nDoc x K
    utheta = LP['theta_u'] # scalar
    logDirNormC = gammaln(utheta + theta.sum(axis=1)) \
                  - (gammaln(utheta) + np.sum(gammaln(theta), axis=1))
    return np.sum(logDirNormC)

  def E_logqPi__memoized(self, LP):
    ''' Returns memoized form for \sum_{d=1}^D E[log q(pi_d)]
    '''
    theta = LP['theta']    # nDoc x K
    utheta = LP['theta_u'] # scalar
    nDoc = theta.shape[0]
    logNormC = np.sum(gammaln(utheta + theta.sum(axis=1)))
    logNormActive = -1 * np.sum(gammaln(theta), axis=0)
    logNormUnused = -1 * nDoc * gammaln(utheta)
    return logNormC, logNormActive, logNormUnused    

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

