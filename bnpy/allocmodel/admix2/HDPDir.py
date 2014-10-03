'''
HDPDir.py
Bayesian nonparametric admixture model via the Hierarchical Dirichlet Process.
Uses a direct construction that maintains K active components,
via a Dirichlet document-level variational factor.

Attributes
-------
K : # of components
gamma : scalar positive real, global concentration 
alpha : scalar positive real, document-level concentration param

Local Model Parameters (document-specific)
--------
z :  one-of-K topic assignment indicator for tokens
     z_{dn} : binary indicator vector for assignment of token n in document d
              z_{dnk} = 1 iff assigned to topic k, 0 otherwise.

pi : document-specific stick-breaking lengths for each active topic 
     pi : 2D array, size D x K+1
Tracks remaining "leftover" mass for all (infinitely many) inactive topics
at index K+1 for variables pi

Local Variational Parameters
--------
resp :  q(z_dn) = Categorical( z_dn | resp_{dn1}, ... resp_{dnK} )

theta : 2D array, size D x K
thetaRem : scalar
q(\pi_d) = Dir( theta[d,0], ... theta[d,K], thetaRem )


Global Model Parameters (shared across all documents)
--------
rho : 1D array, size K
omega : 1D array, size K

q(u_k) = Beta(rho[k]*omega[k], (1-rho[k])*omega[k])

References
-------
TODO
Latent Dirichlet Allocation, by Blei, Ng, and Jordan
introduces a classic admixture model with Dirichlet-Mult observations.
'''
import numpy as np
import logging
Log = logging.getLogger('bnpy')

from ..AllocModel import AllocModel
from bnpy.suffstats import SuffStatBag
from ...util import digamma, gammaln
from ...util import NumericUtil, as1D

import OptimizerHDPDir as OptimHDPDir
import LocalUtil

from bnpy.util.NumericUtil import calcRlogRdotv_allpairs, calcRlogRdotv_specificpairs
from bnpy.util.NumericUtil import calcRlogR_allpairs, calcRlogR_specificpairs

class HDPDir(AllocModel):
  def __init__(self, inferType, priorDict=None):
    if inferType == 'EM':
      raise ValueError('HDPDir cannot do EM.')
    self.inferType = inferType
    self.K = 0
    if priorDict is None:
      self.set_prior()
    else:
      self.set_prior(**priorDict)

  def get_keys_for_memoized_local_params(self):
    ''' Return list of string names of the LP fields
        that moVB needs to memoize across visits to a particular batch
    '''
    return ['DocTopicCount']
      
  def get_active_comp_probs(self):
    ''' Return K vector of appearance probabilities for each of the K comps
    '''
    return self.E_beta_active()

  def E_beta_active(self):
    ''' Return vector beta of appearance probabilities for active components
    '''
    if not hasattr(self, 'Ebeta'):
      self.Ebeta = self.E_beta()
    return self.Ebeta[:-1]

  def E_beta(self):
    ''' Return vector beta of appearance probabilities

        Includes K active topics, and one entry aggregating leftover mass
    '''
    if not hasattr(self, 'Ebeta'):
      self.Ebeta = np.append(self.rho, 1.0)
      self.Ebeta[1:] *= np.cumprod(1.0 - self.rho)
    return self.Ebeta

  def alpha_E_beta(self):
    ''' Return vector of alpha * E[beta] of scaled appearance probabilities

        Includes K active topics, and one entry aggregating leftover mass
    '''
    if not hasattr(self, 'alphaEbeta'):
      self.alphaEbeta = self.alpha * self.E_beta()
    return self.alphaEbeta

  def ClearCache(self):
    if hasattr(self, 'Ebeta'):
      del self.Ebeta
    if hasattr(self, 'alphaEbeta'):
      del self.alphaEbeta

  def set_prior(self, gamma=1.0, alpha=1.0, **kwargs):
    self.alpha = float(alpha)
    self.gamma = float(gamma)

  def to_dict(self):
    return dict(rho=self.rho, omega=self.omega)              

  def from_dict(self, Dict):
    self.inferType = Dict['inferType']
    self.K = Dict['K']
    self.rho = as1D(Dict['rho'])
    self.omega = as1D(Dict['omega'])

  def get_prior_dict(self):
    return dict(alpha=self.alpha, gamma=self.gamma,
                K=self.K,
                inferType=self.inferType)
    
  def get_info_string(self):
    ''' Returns human-readable name of this object
    '''
    return 'HDP model with K=%d active comps. gamma=%.2f. alpha=%.2f' \
            % (self.K, self.gamma, self.alpha)
    

  ####################################################### VB Local Step
  ####################################################### (E-step)
  def calc_local_params(self, Data, LP, **kwargs):
    ''' Calculate document-specific quantities (E-step)
         
          Returns
          -------
          LP : local params dict, with fields
          * resp
          * theta
          * ElogPi
          * DocTopicCount
    '''
    self.alpha_E_beta() # create cached copy
    LP = LocalUtil.calcLocalParams(Data, LP, self, **kwargs)
    assert 'resp' in LP
    assert 'DocTopicCount' in LP
    return LP

  def calcLogPrActiveCompsForDoc(self, DocTopicCount_d, out):
    ''' Calculate log prob of each of the K active topics given doc-topic counts

        Note: for speed, we avoid terms in this expression that are 
        **constant** across all active topics 0, 1, 2, ... K-2, K-1.
         These are commented out below.

        Returns
        -------
        logp : 1D array, size K
        logp[k] gives log prob of topic k in provided doc, up to additive const
    '''
    np.add(DocTopicCount_d, self.alphaEbeta[:-1], out=out)
    ##digammaSum = digamma(out.sum() + self.alphaEbeta[-1])
    digamma(out, out=out)
    ##out -= digammaSum
    return out

  def calcLogPrActiveComps_Fast(self, DocTopicCount, activeDocs=None, LP=None,
                                      out=None):
    ''' Calculate log prob of each active topic for each active document
    '''
    if LP is None:
      LP = dict()

    ## alphaEbeta is 1D array, size K+1
    alphaEbeta = self.alpha * self.E_beta()

    if activeDocs is None:
      activeDocTopicCount = DocTopicCount
    else:
      activeDocTopicCount = np.take(DocTopicCount, activeDocs, axis=0)

    ## theta is 2D array, size D x K
    if 'theta' in LP:
      LP['theta'][activeDocs] = activeDocTopicCount + alphaEbeta[:-1]
    else:
      LP['theta'] = DocTopicCount + alphaEbeta[:-1]

    theta = LP['theta']
    if out is None:
      ElogPi = digamma(theta)
    else:
      ElogPi = out
      ElogPi[activeDocs] = digamma(theta[activeDocs])

    if activeDocs is None:
      digammaSumTheta = digamma(np.sum(theta,axis=1) + alphaEbeta[-1])
      ElogPi -= digammaSumTheta[:,np.newaxis]
    else:
      digammaSumTheta = digamma(np.sum(theta[activeDocs],axis=1) \
                            + alphaEbeta[-1])
      ElogPi[activeDocs] -= digammaSumTheta[:,np.newaxis]
    return ElogPi


  def updateLPGivenDocTopicCount(self, LP, DocTopicCount):
    ''' Update all local parameters, given topic counts for all docs in set.

        Returns
        --------
        LP : dict of local params, with updated fields
        * theta, thetaRem
        * ElogPi, ElogPiRem
    '''
    alphaEbeta = self.alpha * self.E_beta()

    theta = DocTopicCount + alphaEbeta[:-1]
    digammaSumTheta = digamma(theta.sum(axis=1) + alphaEbeta[-1])
    ElogPi = digamma(theta) - digammaSumTheta[:,np.newaxis]
    ElogPiRem = digamma(alphaEbeta[-1]) - digammaSumTheta
    LP['theta'] = theta
    LP['thetaRem'] = alphaEbeta[-1]
    LP['ElogPi'] = ElogPi
    LP['ElogPiRem'] = ElogPiRem
    LP['digammaSumTheta'] = digammaSumTheta
    return LP

  def initLPFromResp(self, Data, LP):
    ''' Obtain initial local params for initializing this model.
    '''
    resp = LP['resp']
    K = resp.shape[1]
    DocTopicCount = np.zeros( (Data.nDoc, K))
    for d in xrange(Data.nDoc):
      start = Data.doc_range[d]
      stop = Data.doc_range[d+1]
      if hasattr(Data, 'word_count'):
        DocTopicCount[d,:] = np.dot(Data.word_count[start:stop],
                                    resp[start:stop,:])
      else:
        DocTopicCount[d,:] = np.sum(resp[start:stop,:], axis=0)

    remMass = np.minimum(0.1, 1.0/(K*K))
    Ebeta = (1 - remMass) / K

    theta = DocTopicCount + self.alpha * Ebeta
    thetaRem = self.alpha * remMass

    digammaSumTheta = digamma(theta.sum(axis=1) + thetaRem)
    ElogPi = digamma(theta) - digammaSumTheta[:,np.newaxis]
    ElogPiRem = digamma(thetaRem) - digammaSumTheta

    LP['DocTopicCount'] = DocTopicCount
    LP['theta'] = theta
    LP['thetaRem'] = thetaRem
    LP['ElogPi'] = ElogPi
    LP['ElogPiRem'] = ElogPiRem
    return LP

  def applyHardMergePairToLP(self, LP, kA, kB):
    ''' Apply hard merge pair to provided local parameters

        Returns
        --------
        mergeLP : dict of updated local parameters
    '''
    resp = np.delete(LP['resp'], kB, axis=1)
    theta = np.delete(LP['theta'], kB, axis=1)
    DocTopicCount = np.delete(LP['DocTopicCount'], kB, axis=1)

    resp[:,kA] += LP['resp'][:, kB]
    theta[:,kA] += LP['theta'][:, kB]
    DocTopicCount[:,kA] += LP['DocTopicCount'][:, kB]

    ElogPi = np.delete(LP['ElogPi'], kB, axis=1)
    ElogPi[:, kA] = digamma(theta[:, kA]) - LP['digammaSumTheta']

    return dict(resp=resp, theta=theta, thetaRem=LP['thetaRem'],
                ElogPi=ElogPi, ElogPiRem=LP['ElogPiRem'],
                DocTopicCount=DocTopicCount,
                digammaSumTheta=LP['digammaSumTheta'])


  ####################################################### Suff Stat Calc
  ####################################################### 
  def get_global_suff_stats(self, Data, LP, doPrecompEntropy=None,
                                            doPrecompMergeEntropy=None,
                                            mPairIDs=None,
                                            preselectroutine=None,
                                            **kwargs):
    ''' Calculate sufficient statistics.
    '''
    resp = LP['resp']
    _, K = resp.shape
    SS = SuffStatBag(K=K, D=Data.get_dim())
    SS.setField('nDoc', Data.nDoc, dims=None)
    SS.setField('sumLogPi', np.sum(LP['ElogPi'], axis=0), dims='K')
    SS.setField('sumLogPiRem', np.sum(LP['ElogPiRem']), dims=None)

    if doPrecompEntropy:
      ElogqZ = self.E_logqZ(Data, LP)
      SS.setELBOTerm('ElogqZ', ElogqZ, dims='K')

      slack_NmT, slack_NmT_Rem = self.slack_NminusTheta(LP)
      SS.setELBOTerm('slackNminusTheta', slack_NmT, dims='K')
      SS.setELBOTerm('slackNminusTheta_Rem', slack_NmT_Rem, dims=None)

      glnSumTheta, glnTheta, glnThetaRem = self.c_Dir_theta__parts(LP)
      SS.setELBOTerm('gammalnSumTheta', glnSumTheta, dims=None)
      SS.setELBOTerm('gammalnTheta', glnTheta, dims='K')
      SS.setELBOTerm('gammalnTheta_Rem', glnThetaRem, dims=None)

    ## Merge Term caching
    if doPrecompMergeEntropy:
      if mPairIDs is None:
        raise NotImplementedError("TODO: all pairs for merges")
      
      ElogqZMat = self.calcElogqZForMergePairs(LP['resp'], Data, mPairIDs)
      SS.setMergeTerm('ElogqZ', ElogqZMat, dims=('K','K'))

      alphaEbeta = self.alpha_E_beta()

      sumLogPi = np.zeros((SS.K, SS.K))
      gammalnTheta = np.zeros((SS.K, SS.K))
      slack_NmT = np.zeros((SS.K, SS.K))
      for (kA, kB) in mPairIDs:
        theta_vec = LP['theta'][:, kA] + LP['theta'][:, kB]
        ElogPi_vec = digamma(theta_vec) - LP['digammaSumTheta']
        gammalnTheta[kA, kB] = np.sum(gammaln(theta_vec))
        sumLogPi[kA, kB] = np.sum(ElogPi_vec)
        ElogPi_vec *= alphaEbeta[kA] + alphaEbeta[kB]
        slack_NmT[kA, kB] = -1 * np.sum(ElogPi_vec)
      SS.setMergeTerm('gammalnTheta', gammalnTheta, dims=('K','K'))
      SS.setMergeTerm('sumLogPi', sumLogPi, dims=('K','K'))
      SS.setMergeTerm('slackNminusTheta', slack_NmT, dims=('K','K'))

      #for (kA, kB) in mPairIDs:
      #  self.verifySSForMergePair(Data, SS, LP, kA, kB)

    ## Selection terms (computes doc-topic correlation)
    if preselectroutine is not None:
      if preselectroutine.count('corr') > 0:
        Tmat = LP['DocTopicCount']
        SS.setSelectionTerm('DocTopicPairMat',
                           np.dot(Tmat.T, Tmat), dims=('K','K'))
        SS.setSelectionTerm('DocTopicSum', np.sum(Tmat, axis=0), dims='K')
    return SS

  def verifySSForMergePair(self, Data, SS, LP, kA, kB):
    mergeLP = self.applyHardMergePairToLP(LP, kA, kB)
    mergeSS = self.get_global_suff_stats(Data, mergeLP, doPrecompEntropy=1)

    sumLogPi_direct = mergeSS.sumLogPi[kA]
    sumLogPi_cached = SS.getMergeTerm('sumLogPi')[kA, kB]
    assert np.allclose(sumLogPi_direct, sumLogPi_cached)

    glnTheta_direct = mergeSS.getELBOTerm('gammalnTheta')[kA]
    glnTheta_cached = SS.getMergeTerm('gammalnTheta')[kA, kB]
    assert np.allclose(glnTheta_direct, glnTheta_cached)

    slack_direct = mergeSS.getELBOTerm('slackNminusTheta')[kA]
    slack_cached = SS.getMergeTerm('slackNminusTheta')[kA, kB]
    assert np.allclose(slack_direct, slack_cached)

    ElogqZ_direct = mergeSS.getELBOTerm('ElogqZ')[kA]
    ElogqZ_cached = SS.getMergeTerm('ElogqZ')[kA, kB]
    assert np.allclose(ElogqZ_direct, ElogqZ_cached)


  def calcElogqZForMergePairs(self, resp, Data, mPairIDs):
    ''' Calculate resp entropy terms for all candidate merge pairs

        Returns
        ---------
        ElogqZ : 2D array, size K x K
    '''
    if hasattr(Data, 'word_count'):
      if mPairIDs is None:
        ElogqZMat = calcRlogRdotv_allpairs(resp, Data.word_count)
      else:
        ElogqZMat = calcRlogRdotv_specificpairs(resp, Data.word_count, mPairIDs)
    else:
      if mPairIDs is None:
        ElogqZMat = calcRlogR_allpairs(resp)
      else:
        ElogqZMat = calcRlogR_specificpairs(resp, mPairIDs)
    return ElogqZMat

  ####################################################### VB Global Step
  #######################################################
  def update_global_params_VB(self, SS, rho=None, 
                                    mergeCompA=None, mergeCompB=None, 
                                    **kwargs):
    ''' Update global parameters.
    '''
    if mergeCompA is None:
      # Standard case:
      # Update via gradient descent.
      rho, omega = self._find_optimum_rhoomega(SS, **kwargs)
    else:
      # Special update case for merges:
      # Fast, heuristic update for rho and omega directly from existing values
      beta = OptimHDPDir.rho2beta_active(self.rho)
      beta[mergeCompA] += beta[mergeCompB]
      beta = np.delete(beta, mergeCompB, axis=0)
      rho = OptimHDPDir.beta2rho(beta, SS.K)
      omega = self.omega
      omega[mergeCompA] += omega[mergeCompB]
      omega = np.delete(omega, mergeCompB, axis=0)
    self.rho = rho
    self.omega = omega
    self.K = SS.K
    self.ClearCache()

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
    if hasattr(self, 'rho') and self.rho.size == SS.K:
      initrho = self.rho
      initomega = self.omega
    else:
      initrho = None   # default initialization
      initomega = None

    try:
      sumLogPi = np.append(SS.sumLogPi, SS.sumLogPiRem)
      rho, omega, f, Info = OptimHDPDir.find_optimum_multiple_tries(
                                        sumLogPi=sumLogPi,
                                        nDoc=SS.nDoc,
                                        gamma=self.gamma, alpha=self.alpha,
                                        initrho=initrho, initomega=initomega)
    except ValueError as error:
      if str(error).count('FAILURE') == 0:
        raise error
      if hasattr(self, 'rho') and self.rho.size == SS.K:
        Log.error('***** Optim failed. Remain at cur val. ' + str(error))
        rho = self.rho
        omega = self.omega
      else:
        Log.error('***** Optim failed. Set to default init. ' + str(error))
        omega = (1 + self.gamma) * np.ones(SS.K)
        rho = OptimHDPDir.create_initrho(K)
    return rho, omega


  ####################################################### Set Global Params
  #######################################################
  def init_global_params(self, Data, K=0, **kwargs):
    ''' Initialize rho, omega to reasonable values
    '''
    self.K = K
    self.rho = OptimHDPDir.create_initrho(K)
    self.omega = (1.0 + self.gamma) * np.ones(K)
    self.ClearCache()

  def set_global_params(self, hmodel=None, rho=None, omega=None, 
                              **kwargs):
    ''' Set rho, omega to provided values.
    '''
    if hmodel is not None:
      self.K = hmodel.allocModel.K
      if hasattr(hmodel.allocModel, 'rho'):
        self.rho = hmodel.allocModel.rho
        self.omega = hmodel.allocModel.omega
      else:
        raise AttributeError('Unrecognized hmodel')
    elif rho is not None and omega is not None:
      self.rho = rho
      self.omega = omega
      self.K = omega.size
    else:
      self._set_global_params_from_scratch(**kwargs)
    self.ClearCache()


  def _set_global_params_from_scratch(self, beta=None, probs=None,
                                            Data=None, nDoc=None, **kwargs):
    ''' Set rho, omega to values that reproduce provided appearance probs
    '''
    if nDoc is None:
      nDoc = Data.nDoc
    if nDoc is None:
      raise ValueError('Bad parameters. nDoc not specified.')
    if probs is not None:
      beta = probs / probs.sum()
    if beta is not None:
      Ktmp = beta.size
      rem = np.minimum(0.05, 1./(Ktmp))
      beta = np.hstack([np.squeeze(beta), rem])
      beta = beta/np.sum(beta)
    else:
      raise ValueError('Bad parameters. Vector beta not specified.')
    self.K = beta.size - 1
    self.rho, self.omega = self._convert_beta2rhoomega(beta, )
    assert self.rho.size == self.K
    assert self.omega.size == self.K


  def _convert_beta2rhoomega(self, beta, nDoc=10):
    ''' Find vectors rho, omega that are probable given beta

        Returns
        --------
        rho : 1D array, size K
        omega : 1D array, size K
    '''
    assert abs(np.sum(beta) - 1.0) < 0.001
    rho = OptimHDPDir.beta2rho(beta, self.K)
    omega = (nDoc + self.gamma) * np.ones(rho.size)
    return rho, omega


  ####################################################### Calc ELBO
  #######################################################
  def calc_evidence(self, Data, SS, LP, todict=0, **kwargs):
    ''' Calculate ELBO objective 
    '''
    calpha = SS.nDoc * (gammaln(self.alpha) + (SS.K+1) * np.log(self.alpha))
    U_plus_cDir_alphaBeta = self.E_logpU_logqU_plus_cDirAlphaBeta(SS)
    slack_alphaBeta, slack_alphaBeta_Rem = self.slack_alphaBeta(SS)
    if SS.hasELBOTerms():
      ElogqZ = SS.getELBOTerm('ElogqZ')
      cDir_theta = self.c_Dir_theta(SS.getELBOTerm('gammalnSumTheta'),
                                    SS.getELBOTerm('gammalnTheta'),
                                    SS.getELBOTerm('gammalnTheta_Rem'),
                                    )
      slack_NmT = SS.getELBOTerm('slackNminusTheta')
      slack_NmT_Rem = SS.getELBOTerm('slackNminusTheta_Rem')
    else:
      ElogqZ = self.E_logqZ(Data, LP)
      cDir_theta = self.c_Dir_theta(*self.c_Dir_theta__parts(LP))
      slack_NmT, slack_NmT_Rem = self.slack_NminusTheta(LP)

    if todict:
      return dict(calpha=calpha,
                  cDir_theta=-1*cDir_theta,
                  entropy=-1*np.sum(ElogqZ),
                  cDir_alphaBeta=U_plus_cDir_alphaBeta,
                  slack=np.sum(slack_NmT + slack_alphaBeta) \
                        + slack_NmT_Rem + slack_alphaBeta_Rem
                 )

    return U_plus_cDir_alphaBeta + calpha \
           - np.sum(ElogqZ) \
           - cDir_theta \
           + np.sum(slack_NmT + slack_alphaBeta) \
           + slack_NmT_Rem + slack_alphaBeta_Rem 

  def slack_alphaBeta(self, SS):
    ''' Calculate part of doc-topic slack term dependent on alpha * Ebeta

        Returns
        --------
        slack_aBeta_active : 1D array, size K
        slack_aBeta_rem : scalar
    ''' 
    alphaEbeta = self.alpha * self.E_beta()
    return alphaEbeta[:-1] * SS.sumLogPi, alphaEbeta[-1] * SS.sumLogPiRem

  def slack_NminusTheta(self, LP):
    ''' Calculate part of doc-topic slack term dependent on N[d,k] - theta[d,k]

        Returns
        -------
        slack_active : 1D array, size K
        slack_rem : scalar
    '''
    slack = LP['DocTopicCount'] - LP['theta']
    slack *=  LP['ElogPi']
    slack_active = np.sum(slack, axis=0)
    slack_rem = -1 * np.sum(LP['thetaRem'] * LP['ElogPiRem'])
    return slack_active, slack_rem

  def c_Dir_theta__parts(self, LP):
    ''' Calculate quantities needed to compute cumulant of q(pi | theta)

        Returns
        --------
        gammalnSumTheta : scalar
        gammalnTheta_active : 1D array, size K
        gammalnTheta_rem : scalar
    '''
    nDoc = LP['theta'].shape[0]
    sumTheta = np.sum(LP['theta'], axis=1) + LP['thetaRem']
    gammalnSumTheta = np.sum(gammaln(sumTheta))
    gammalnTheta_active = np.sum(gammaln(LP['theta']), axis=0)
    gammalnTheta_rem = nDoc * gammaln(LP['thetaRem'])
    return gammalnSumTheta, gammalnTheta_active, gammalnTheta_rem

  def c_Dir_theta(self, gammalnSumTheta, gammalnTheta, gammalnTheta_rem):
    ''' Calculate cumulant function for q(pi | theta)
    '''
    return gammalnSumTheta - np.sum(gammalnTheta) - gammalnTheta_rem
    
  def E_logpU_logqU_plus_cDirAlphaBeta(self, SS):
    ''' Calculate E[ log p(u) - log q(u) ]

        Returns
        ---------
        Elogstuff : real scalar
    '''
    g1 = self.rho * self.omega
    g0 = (1-self.rho) * self.omega
    digammaBoth = digamma(g1+g0)
    ElogU = digamma(g1) - digammaBoth
    Elog1mU = digamma(g0) - digammaBoth

    ONcoef = SS.nDoc + 1.0 - g1
    OFFcoef = SS.nDoc * OptimHDPDir.kvec(self.K) + self.gamma - g0

    cDiff = SS.K * c_Beta(1, self.gamma) - c_Beta(g1, g0)
    logBetaPDF = np.inner(ONcoef, ElogU) \
                 + np.inner(OFFcoef, Elog1mU)
    return cDiff + logBetaPDF

  def E_logqZ(self, Data, LP):
    ''' Calculate E[ log q(z)] for each active topic

        Returns
        -------
        ElogqZ : 1D array, size K
    '''
    if hasattr(Data, 'word_count'):
      return NumericUtil.calcRlogRdotv(LP['resp'], Data.word_count)
    else:
      return NumericUtil.calcRlogR(LP['resp'])

  ######################################################### OLD calc_evidence
  #########################################################
  # To be used only to verify the current objective

  def zzz_calc_evidence(self, Data, SS, LP, **kwargs):
    ''' Calculate ELBO objective 
    '''
    calpha = SS.nDoc * (gammaln(self.alpha) + (SS.K+1) * np.log(self.alpha))
    UandcPi_global = self.E_logpU_logqU_c(SS)
    Pi_global = self.E_logpPi__global(SS)
    if SS.hasELBOTerms():
      ElogqZ = SS.getELBOTerm('ElogqZ')
      VPi_local = SS.getELBOTerm('VPilocal')
    else:
      ElogqZ = self.E_logqZ(Data, LP)
      VPi_local = self.E_logpPiZ_logqPi(Data, LP)
    elbo = calpha + UandcPi_global + Pi_global + VPi_local - np.sum(ElogqZ)
    return dict(calpha=calpha,
                UandcPi_global=UandcPi_global,
                Pi_global=Pi_global,
                ZPi_local=VPi_local,
                ElogqZ=ElogqZ,
                elbo=elbo)

  def E_logpPi__global(self, SS):
    ''' Calculate the part of E[ log p(v) ] that depends on global topic probs
        DEPRECATED
        Returns
        --------
        Elogstuff : real scalar
    ''' 
    alphaEbeta = self.alpha * self.E_beta()
    return np.inner(alphaEbeta[:-1], SS.sumLogPi) \
           + alphaEbeta[-1] * SS.sumLogPiRem

  def E_logpPiZ_logqPi(self, Data, LP):
    ''' Calculate E[ log p(v) + log p(z) - log q(v) ]
        DEPRECATED

        Returns
        -------
        Elogstuff : real scalar
    '''
    cDiff = -1 * c_Dir(LP['theta'], LP['thetaRem'])
    logDirPDF = np.sum((LP['DocTopicCount'] - LP['theta']) * LP['ElogPi']) \
                 - np.sum(LP['thetaRem'] * LP['ElogPiRem'])
    return cDiff + np.sum(logDirPDF)

  def E_logpU_logqU_c(self, SS):
    ''' Calculate E[ log p(u) - log q(u) ]
        DEPRECATED

        Returns
        ---------
        Elogstuff : real scalar
    '''
    g1 = self.rho * self.omega
    g0 = (1-self.rho) * self.omega
    digammaBoth = digamma(g1+g0)
    ElogU = digamma(g1) - digammaBoth
    Elog1mU = digamma(g0) - digammaBoth

    ONcoef = SS.nDoc + 1.0 - g1
    OFFcoef = SS.nDoc * OptimHDPDir.kvec(self.K) + self.gamma - g0

    cDiff = SS.K * c_Beta(1, self.gamma) - c_Beta(g1, g0)
    logBetaPDF = np.inner(ONcoef, ElogU) \
                 + np.inner(OFFcoef, Elog1mU)
    return cDiff + logBetaPDF




def c_Beta(a1, a0):
  ''' Evaluate cumulant function of the Beta distribution

      When input is vectorized, we compute sum over all entries.

      Returns
      -------
      c : scalar real
  '''
  return np.sum(gammaln(a1 + a0)) - np.sum(gammaln(a1)) - np.sum(gammaln(a0))  



def c_Dir(AMat, arem):
  ''' Evaluate cumulant function of the Dir distribution

      When input is vectorized, we compute sum over all entries.

      Returns
      -------
      c : scalar real
  '''
  D = AMat.shape[0]
  return  np.sum(gammaln(np.sum(AMat,axis=1)+arem)) \
          - np.sum(gammaln(AMat)) \
          - D * np.sum(gammaln(arem))

def c_Dir__big(AMat, arem):
  AMatBig = np.hstack([AMat, arem*np.ones(AMat.shape[0])[:,np.newaxis]])
  return np.sum(gammaln(np.sum(AMatBig,axis=1))) - np.sum(gammaln(AMatBig))

def c_Dir__slow(AMat, arem):
  c = 0
  for d in xrange(AMat.shape[0]):
    avec = np.hstack([AMat[d], arem])
    c += gammaln(np.sum(avec)) - np.sum(gammaln(avec))
  return c
