'''
HDPStickBreak.py
Bayesian nonparametric admixture model with unbounded number of components K.
Uses stick-breaking construction for both global and document-level parameters.
'''
import numpy as np
import logging
Log = logging.getLogger('bnpy')

import LocalStepSBBagOfWords
import OptimizerForHDPStickBreak as OptimHDPSB

from ..AllocModel import AllocModel
from bnpy.suffstats import SuffStatBag
from ...util import NumericUtil
from ...util import digamma, gammaln
from ...util import EPS

MAXOMEGA = 1e7

class HDPStickBreak(AllocModel):

  ######################################################### Constructors
  #########################################################
  def __init__(self, inferType, priorDict=None):
    if inferType == "EM":
      raise ValueError('HDPModel cannot do EM. Only VB possible.')
    self.inferType = inferType
    self.K = 0
    self.set_prior(priorDict)
    
  def set_prior(self, priorDict):
    self.alpha0 = float(priorDict['alpha0'])
    self.gamma = float(priorDict['gamma'])

  def set_helper_params(self):
    ''' Set dependent attribs of this model, given primary parameters.
        This includes expectations of various stickbreaking quantities.
    '''
    assert self.rho.size == self.K
    assert self.omega.size == self.K
    cumprod1mrho = self._calc_cumprod1mrho()
    self.topicPrior1 = self.gamma * self.rho * cumprod1mrho
    self.topicPrior0 = self.gamma * (1-self.rho) * cumprod1mrho

    digammaomega = digamma(self.omega)
    self.Elogv = digamma(self.rho * self.omega) - digammaomega
    self.Elog1mv = digamma((1-self.rho) * self.omega) - digammaomega

  def _calc_cumprod1mrho(self, rho=None):
    if rho is None:
      rho = self.rho
    K = rho.size
    cumprod1mrho = np.ones(K)
    if K == 1:
      return cumprod1mrho
    np.cumprod(1-rho[:-1], out=cumprod1mrho[1:])
    return cumprod1mrho

  ######################################################### Accessors
  #########################################################
  def get_keys_for_memoized_local_params(self):
    ''' Return list of string names of the LP fields
        that moVB needs to memoize across visits to a particular batch
    '''
    return ['DocTopicCount']

  ######################################################### Local Params
  #########################################################
  def calc_local_params(self, Data, LP, methodLP='memo', **kwargs):
    methods = methodLP.split(',')
    if len(methods) == 1:
      if 'doInPlaceLP' not in kwargs:
        kwargs['doInPlaceLP'] = 1
      LP = LocalStepSBBagOfWords.calcLocalDocParams(Data, LP, 
                                  self.topicPrior1, self.topicPrior0,
                                  methodLP=methodLP, **kwargs)
      bestLP = self._local_update_Resp(Data, LP)
    else:
      bestLP = None
      for mID, mname in enumerate(reversed(sorted(methods))):
        initLP = dict(**LP)
        if mname == 'memo' and 'DocTopicCount' not in LP:
          continue
        
        curLP = self.calc_local_params(Data, dict(**initLP), methodLP=mname, 
                                                     doInPlaceLP=0, **kwargs) 
        curELBO = self.calcPerDocELBO(Data, curLP)
        
        if bestLP == None:
          bestELBO = curELBO
          bestLP = curLP
        else:
          # determine which docs have it better under current method
          docIDs = curELBO > bestELBO + 1e-8 # ensure difference is meaningful
          if np.sum(docIDs) > 0:
            bestELBO[docIDs] = curELBO[docIDs]
            bestLP = self._swap_LP_for_specific_docs(Data, bestLP, 
                                                           curLP, docIDs)

      bestLP['perDocELBO'] = bestELBO
    assert np.allclose( bestLP['word_variational'].sum(axis=1), 1.0)
    return bestLP

  def _local_update_Resp(self, Data, LP):
    LP['word_variational'] = LP['expEloglik']
    for d in xrange(Data.nDoc):
      start = Data.doc_range[d,0]
      stop  = Data.doc_range[d,1]
      LP['word_variational'][start:stop] *= LP['expElogpi'][d]
    LP['word_variational'] /= LP['sumRTilde'][:, np.newaxis]

    # make it safe to take logs
    np.maximum(LP['word_variational'], 1e-300, out=LP['word_variational'])
    return LP

  def _swap_LP_for_specific_docs(self, Data, LP, LP2, docIDs):
    ''' For each doc in docIDs, move relevant parameters from LP2 into LP
    '''
    for d in np.flatnonzero(docIDs):
      start = Data.doc_range[d,0]
      stop  = Data.doc_range[d,1]
      LP['word_variational'][start:stop] = LP2['word_variational'][start:stop]
    LP['DocTopicCount'][docIDs] = LP2['DocTopicCount'][docIDs]
    LP['E_logVd'][docIDs] = LP2['E_logVd'][docIDs]
    LP['E_log1-Vd'][docIDs] = LP2['E_log1-Vd'][docIDs]
    LP['E_logPi'][docIDs] = LP2['E_logPi'][docIDs]

    LP['U1'][docIDs] = LP2['U1'][docIDs]
    LP['U0'][docIDs] = LP2['U0'][docIDs]
    return LP

  ######################################################### Suff Stats
  #########################################################
  def get_global_suff_stats(self, Data, LP, doPrecompEntropy=False, 
                                            doPrecompMergeEntropy=False,
                                            preselectroutine=None,
                                            mPairIDs=None):
    ''' Calculate sufficient statistics
    '''
    wv = LP['word_variational']
    _, K = wv.shape
    SS = SuffStatBag(K=K, D=Data.vocab_size)
    SS.setField('nDoc', Data.nDoc, dims=None)
    SS.setField('sumLogVd', np.sum(LP['E_logVd'], axis=0), dims='K')
    SS.setField('sumLog1mVd', np.sum(LP['E_log1-Vd'], axis=0), dims='K')

    if doPrecompEntropy:
      # Z terms
      SS.setELBOTerm('ElogpZ', self.E_logpZ(Data, LP), dims='K')
      SS.setELBOTerm('ElogqZ', self.E_logqZ(Data, LP), dims='K')
      # Vd terms
      ElogqVd = self.E_logqVd__memoized(LP)
      SS.setELBOTerm('ElogqVd', ElogqVd, dims='K')

    if doPrecompMergeEntropy:
      raise NotImplementedError('TODO!')

    if preselectroutine is not None:
      self.add_selection_terms_to_SS(SS, LP, preselectroutine)
    return SS

  def add_selection_terms_to_SS(self, SS, preselectroutine):
    ''' Add terms to SuffStatBag for selecting good merge candidates.
    '''
    if preselectroutine.count('doctopiccorr') > 0:
      Tmat = LP['DocTopicCount']
      SS.setSelectionTerm('DocTopicPairMat',
                           np.dot(Tmat.T, Tmat), dims=('K','K'))
      SS.setSelectionTerm('DocTopicSum', np.sum(Tmat, axis=0), dims='K')
  


  def insertCompsIntoSuffStatBag(self, SS, freshSS, doUpdateModel=1):
    ''' Model-specific correction to SuffStatBag's built-in method for
          inserting components.

        Returns
        -------
        SS : SuffStatBag, with SS.K + freshSS.K components,
                          with scale consistent with SS + freshSS
        AdjustInfo: dict, with fields indicating which entries in SS
                          have been adjusted, and what the "per document" 
                            adjustment factor is
    '''
    nDoc = SS.nDoc
    AInfo, RInfo = self.calcSSAdjustmentsForExpansion(SS, freshSS.K)

    new_rho = 1.0 / (1.0 + self.alpha0) * np.ones(freshSS.K)
    new_omega = (1 + self.alpha0) * np.ones(freshSS.K)
    if doUpdateModel:
      self.K = self.K + freshSS.K
      self.rho = np.hstack([self.rho, new_rho])
      self.omega = np.hstack([self.omega, new_omega])
      self.set_helper_params()

    SS.insertComps(freshSS)
    arr = SS.sumLogVd + nDoc * AInfo['sumLogVd']
    SS.setField('sumLogVd', arr, dims='K')
    arr = SS.sumLog1mVd + nDoc * AInfo['sumLog1mVd']
    SS.setField('sumLog1mVd', arr, dims='K')
    return SS, AInfo, RInfo

  def insertEmptyCompsIntoSuffStatBag(self, SS, Kextra):
    ''' Model-specific correction to SuffStatBag's built-in method for
          inserting components.
    '''
    nDoc = SS.nDoc
    AInfo, RInfo = self.calcSSAdjustmentsForExpansion(SS, Kextra)
    SS.insertEmptyComps(Kextra)

    arr = SS.sumLogVd + nDoc * AInfo['sumLogVd']
    SS.setField('sumLogVd', arr, dims='K')
    arr = SS.sumLog1mVd + nDoc * AInfo['sumLog1mVd']
    SS.setField('sumLog1mVd', arr, dims='K')

    if SS.hasELBOTerms():
      arr = SS.getELBOTerm('ElogqVd') + nDoc * AInfo['ElogqVd']
      SS.setELBOTerm('ElogqVd', arr, dims='K')
    return SS, AInfo, RInfo


  def calcSSAdjustmentsForExpansion(self, SS, Kextra):
    ''' Calculate what should be added/replaced to SS to expand active comps

        Returns
        -------
        AdjustInfo
        ReplaceInfo
    ''' 
    # Calculate corrected theta terms for new components K+1,K+2,... K+Kfresh
    # Essentially, divide up the existing "left-over" mass among Kfresh comps
    #   using the stick-breaking prior as the division procedure
    remMass = np.prod(1 - self.rho)
    priorEv = 1.0 / (1.0 + self.alpha0)
    assert remMass > 0
    newU1 = np.zeros(Kextra)
    newU0 = np.zeros(Kextra)
    for k in xrange(Kextra):
      newU1[k] = self.gamma * priorEv * remMass
      newU0[k] = self.gamma * (1-priorEv) * remMass
      remMass = remMass * (1-priorEv)

    # Calc "adjustment" for each SS field on a per-document basis 
    #   to create a sensible expanded Udk1, Udk0 parameters
    nDoc = SS.nDoc
    sumLogVd = digamma(newU1) - digamma(newU1+newU0)
    sumLog1mVd = digamma(newU0) - digamma(newU1+newU0)
    ElogqVd = gammaln(newU1 + newU0) - gammaln(newU1) - gammaln(newU0) \
              + (newU1 - 1.0) * sumLogVd + (newU0 - 1.0) * sumLog1mVd

    sumLogVd = np.hstack([np.zeros(SS.K), sumLogVd])
    sumLog1mVd = np.hstack([np.zeros(SS.K), sumLog1mVd])
    ElogqVd = np.hstack([np.zeros(SS.K), ElogqVd])
    AdjustInfo = dict(sumLogVd=sumLogVd, sumLog1mVd=sumLog1mVd,
                      ElogqVd=ElogqVd)
    ReplaceInfo = dict()
    return AdjustInfo, ReplaceInfo

  ######################################################### Global Params
  #########################################################

  def update_global_params_VB(self, SS, comps=None, **kwargs):
    ''' Update global parameters that control topic probabilities
    '''
    self.K = SS.K
    rho, omega = self._find_optimum_rhoomega(SS, **kwargs)
    self.rho = rho
    self.omega = omega
    self.set_helper_params()
        
  def update_global_params_soVB(self, SS, rho, **kwargs):
    raise NotImplementedError('TODO')

  #________________________________________________________ init_global_params
 
  def init_global_params(self, Data, K=0, **initArgs):
    ''' Initialize global parameters "from scratch".
    '''
    self.K = K
    self.rho = self._get_rho_for_uniform_topic_probs(K)
    self.omega = (1.0 + self.alpha0) * np.ones(K)
    self.set_helper_params()

  #________________________________________________________ set_global_params
  def set_global_params(self, hmodel=None, 
                              rho=None, omega=None, 
                              **kwargs):
    if hmodel is not None:
      self.K = hmodel.allocModel.K
      if hasattr(hmodel.allocModel, 'rho'):
        self.rho = hmodel.allocModel.rho
        self.omega = hmodel.allocModel.omega
      elif hasattr(hmodel.allocModel, 'qalpha1'):
        u1 = hmodel.allocModel.qalpha1
        u0 = hmodel.allocModel.qalpha0
        self.rho = u1 / (u1 + u0)
        self.omega = u1 + u0
      else:
        raise AttributeError('Unrecognized hmodel')
      self.set_helper_params()
    elif rho is not None and omega is not None:
      self.rho = rho
      self.omega = omega
      self.K = omega.size
      self.set_helper_params()
    else:
      self._set_global_params_from_scratch(**kwargs)

  def _set_global_params_from_scratch(self, beta=None, nDoc=10,
                              Ebeta=None, EbetaLeftover=None, **kwargs):
    if Ebeta is not None and EbetaLeftover is not None:
      beta = np.hstack([np.squeeze(Ebeta), np.squeeze(EbetaLeftover)])
    elif beta is not None:
      Ktmp = beta.size
      rem = np.minimum( 0.05, 1./(Ktmp))
      beta = np.hstack([np.squeeze(beta), rem])
      beta = beta/np.sum(beta)
    else:
      raise ValueError('Bad parameters. Vector beta not specified.')
    self.K = beta.size - 1
    
    # Now, use the specified value of beta to find the best U1, U0
    self.rho, self.omega = self._convert_beta2rhoomega(beta, nDoc)
    assert self.rho.size == self.K
    assert self.omega.size == self.K
    self.set_helper_params()

  #-------------------------------------------------------- find_optimum
  def _find_optimum_rhoomega(self, SS, **kwargs):
    ''' Run numerical optimization to find optimal rho, omega parameters

        Args
        --------
        SS : bnpy SuffStatBag, with K components

        Returns
        --------
        rho : 1D array, length K
        omega : 1D array, length K
    '''
    if hasattr(self, 'rho') and self.rho.size == self.K:
      initrho = self.rho
      initomega = self.omega
    else:
      initrho = None   # default initialization
      initomega = None 

    try:
      rho, omega, f, Info = OptimHDPSB.find_optimum_multiple_tries(
                                        sumLogVd=SS.sumLogVd,
                                        sumLog1mVd=SS.sumLog1mVd,
                                        nDoc=SS.nDoc,
                                        gamma=self.gamma, alpha=self.alpha0,
                                        initrho=initrho, initomega=initomega)
    except ValueError as error:
      if str(error).count('FAILURE') == 0:
        raise error
      if hasattr(self, 'rho') and self.rho.size == self.K:
        Log.error('***** Optim failed. Remain at cur val. ' + str(error))
        rho = self.rho
        omega = self.omega
      else:
        Log.error('***** Optim failed. Set to prior. ' + str(error))
        omega = (self.alpha0 + 1 ) * np.ones(SS.K)
        rho = 1/float(1+self.alpha0) * np.ones(SS.K)

    nBad = np.sum(omega > MAXOMEGA)
    if nBad > 0:
      Log.error('***** Enforced MAXOMEGA for %d entries.' % (nBad)) 
      omega = np.minimum(omega, MAXOMEGA)
    return rho, omega



  def _convert_beta2rhoomega(self, beta, nDoc=10):
    ''' Find vectors rho, omega that are probable given beta

        Returns
        --------
        rho : 1D array, size K
        omega : 1D array, size K
    '''
    assert abs(np.sum(beta) - 1.0) < 0.001
    rho = OptimHDPSB._beta2v(beta)
    omega = nDoc * np.ones(rho.size)
    return rho, omega    

  def _get_rho_for_uniform_topic_probs(self, K=0):
    ''' Find vector rho of size K, such that E[beta_k] ~= 1/K
    '''
    rem = np.minimum( 0.05, 1.0/K)
    beta = (1.0-rem)/K * np.ones(K+1)
    beta[-1] = rem
    rho = OptimHDPSB._beta2v(beta)
    assert rho.size == K
    return rho


  ######################################################### Evidence
  #########################################################  
  def calc_evidence( self, Data, SS, LP, todict=False):
    ''' Calculate ELBO terms related to allocation model
    '''   
    E_logpV = self.E_logpV()
    E_logqV = self.E_logqV()
     
    E_logpVd = self.E_logpVd(SS)
    if SS.hasELBOTerms():
      E_logqVd = np.sum(SS.getELBOTerm('ElogqVd'))
      E_logpZ = np.sum(SS.getELBOTerm('ElogpZ'))
      E_logqZ = np.sum(SS.getELBOTerm('ElogqZ'))
    else:
      E_logqVd = self.E_logqVd(LP)
      E_logpZ = np.sum(self.E_logpZ(Data, LP))
      E_logqZ = np.sum(self.E_logqZ(Data, LP))

    if SS.hasAmpFactor():
      E_logqVd *= SS.ampF
      E_logpZ *= SS.ampF
      E_logqZ *= SS.ampF

    elbo = ((E_logpVd - E_logqVd)
            + (E_logpZ - E_logqZ)
            + (E_logpV - E_logqV))

    if todict:
      return dict(elbo=elbo, v_Elogp=E_logpV, v_Elogq=E_logqV,
                             z_Elogp=E_logpZ, z_Elogq=E_logqZ,
                             vd_Elogp=E_logpVd, vd_Elogq=E_logqVd)

    return elbo


  ####################################################### ELBO per doc 
  def calcPerDocELBO(self, Data, LP):
    ''' Returns scalar ELBO for each document
    '''
    perDocELBO = np.sum( gammaln(LP['U1'])
                       + gammaln(LP['U0'])
                       - gammaln(LP['U1'] + LP['U0']), axis=1)
    perDocELBOdata = np.zeros_like(perDocELBO)
    perDocELBOh = np.zeros_like(perDocELBO)

    for d in xrange(Data.nDoc):
      start = Data.doc_range[d,0]
      stop = Data.doc_range[d,1]
      perDocResp = LP['word_variational'][start:stop]
      perDocWC = Data.word_count[start:stop]
      perDocELBOdata[d] = np.sum( perDocWC[:,np.newaxis] * perDocResp \
                                 * LP['E_logsoftev_WordsData'][start:stop])
      perDocELBOh[d] = np.sum(NumericUtil.calcRlogRdotv(perDocResp,
                                                        perDocWC))
    #print '%.8e' % (perDocELBO.sum())
    #print '%.8e' % (perDocELBOdata.sum())
    #print '%.8e' % (perDocELBOh.sum())      
    return perDocELBO + perDocELBOdata - perDocELBOh

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
    return NumericUtil.calcRlogRdotv(wv, Data.word_count)

  def E_logqZ_memo_terms_for_merge(self, Data, LP, mPairIDs=None):
    ''' Returns KxK matrix 
    ''' 
    wv = LP['word_variational']
    if mPairIDs is None:
      ElogqZMat = NumericUtil.calcRlogRdotv_allpairs(wv, Data.word_count)
    else:
      ElogqZMat = NumericUtil.calcRlogRdotv_specificpairs(wv, 
                                                Data.word_count, mPairIDs)
    return ElogqZMat

  ####################################################### ELBO terms for Pi
  def E_logpVd(self, SS):
    ''' Returns scalar value of \sum_d \sum_k E[ log p(v_dk)]
    '''
    K = SS.K
    kvec = K + 1 - np.arange(1, K+1)
    # logDirNormC : scalar norm const that applies to each iid draw pi_d
    logDirNormC = K * np.log(self.gamma) \
                  + np.sum(self.Elogv) + np.inner(kvec, self.Elog1mv)
    logBetaPDF =   np.inner(self.topicPrior1-1, SS.sumLogVd) \
                 + np.inner(self.topicPrior0-1, SS.sumLog1mVd)
    return (SS.nDoc * logDirNormC) + logBetaPDF

  def E_logqVd(self, LP):
    ''' Returns scalar value of \sum_d \sum_k E[ log q(v_dk)]
    '''
    U1 = LP['U1']
    U0 = LP['U0']
    logDirNormC = np.sum(gammaln(U1 + U0) - gammaln(U0) - gammaln(U1))
    logDirPDF =   (U1 - 1.) * LP['E_logVd'] \
                + (U0 - 1.) * LP['E_log1-Vd']
    return logDirNormC + logDirPDF.sum()

  def E_logqVd__memoized(self, LP):
    ''' Returns memoization of \sum_d \sum_k E[ log q(v_dk)]
    '''
    U1 = LP['U1']
    U0 = LP['U0']
    logDirNormC = np.sum(gammaln(U1 + U0) - gammaln(U0) - gammaln(U1), axis=0)
    logDirPDF =   (U1 - 1.) * LP['E_logVd'] \
                + (U0 - 1.) * LP['E_log1-Vd']
    return logDirNormC + logDirPDF.sum(axis=0)


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
    logBetaPDF =   np.inner(u1 - 1., self.Elogv) \
                 + np.inner(u0 - 1., self.Elog1mv)
    return np.sum(logBetaNormC) + logBetaPDF

  ######################################################### IO Utils
  #########################################################   for humans
    
  def get_info_string( self):
    ''' Returns human-readable name of this object'''
    s = 'HDPStickBreak. K=%d, alpha=%.2f, gamma=%.2f.'
    return s % (self.K, self.alpha0, self.gamma)
     
  ######################################################### IO Utils
  #########################################################   for machines
  def to_dict( self ):
    return dict(rho=self.rho, omega=self.omega)              
  
  def from_dict(self, Dict):
    self.inferType = Dict['inferType']
    self.K = int(Dict['K'])
    self.rho = _to1D(np.asarray(Dict['rho'], dtype=np.float64))
    self.omega = _to1D(np.asarray(Dict['omega'], dtype=np.float64))  
    self.set_helper_params()

  def get_prior_dict( self ):
    return dict(K=self.K, alpha0=self.alpha0, gamma=self.gamma)

def _to1D(arr):
  if arr.ndim == 0:
    arr = arr[np.newaxis]
  elif arr.ndim > 1:
    arr = np.squeeze(arr)
  assert arr.ndim ==1
  return arr

