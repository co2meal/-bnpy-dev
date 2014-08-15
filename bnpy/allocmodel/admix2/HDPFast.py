'''
HDPFast.py
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

from ..AllocModel import AllocModel
from bnpy.suffstats import SuffStatBag
from ...util import digamma, gammaln
from ...util import NumericUtil, as1D

import OptimizerHDPFast as OptimFast
import LocalUtil

class HDPFast(AllocModel):
  def __init__(self, inferType, priorDict=None):
    if inferType == 'EM':
      raise ValueError('HDPFastFixed cannot do EM.')
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

  def ClearCache(self):
    if hasattr(self, 'Ebeta'):
      del self.Ebeta

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
    LP = LocalUtil.calcLocalParams(Data, LP, self, **kwargs)
    assert 'resp' in LP
    assert 'DocTopicCount' in LP
    return LP

  def calcLogPrActiveCompsForDoc(self, DocTopicCount_d):
    ''' Calculate log prob of each of the K active topics given doc-topic counts

        Returns
        -------
        logp : 1D array, size K
               logp[k] gives probability of topic k in provided doc
    '''
    theta = self.alpha * self.E_beta()
    theta[:-1] += DocTopicCount_d
    ElogPi = digamma(theta[:-1]) - digamma(theta.sum()) 
    return ElogPi

  def calcLogPrActiveComps_Fast(self, DocTopicCount, activeDocs=None, LP=dict(),
                                      out=None):
    ''' Calculate log prob of each active topic for each active document
    '''
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
    LP['DocTopicCount'] = DocTopicCount
    return LP

  ####################################################### Suff Stat Calc
  ####################################################### 
  def get_global_suff_stats(self, Data, LP, doPrecompEntropy=None, **kwargs):
    ''' Calculate sufficient statistics.
    '''
    resp = LP['resp']
    _, K = resp.shape
    SS = SuffStatBag(K=K, D=Data.get_dim(), A=Data.nDoc)
    SS.setField('nDoc', Data.nDoc, dims=None)
    SS.setField('DocTopicCount', LP['DocTopicCount'], dims=('A', 'K'))
    if doPrecompEntropy:
      ElogqZ = self.E_logqZ(Data, LP)
      SS.setELBOTerm('ElogqZ', ElogqZ, dims='K')

    return SS

  ####################################################### VB Global Step
  #######################################################
  def update_global_params_VB(self, SS, rho=None, **kwargs):
    ''' Update global parameters.
    '''
    rho, omega = self._find_optimum_rhoomega(SS, **kwargs)
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
      rho, omega, f, Info = OptimFast.find_optimum_multiple_tries(
                                        DocTopicCount=SS.DocTopicCount,
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
        rho = OptimFast.create_initrho(K)
    return rho, omega


  ####################################################### Set Global Params
  #######################################################
  def init_global_params(self, Data, K=0, **kwargs):
    ''' Initialize rho, omega to reasonable values
    '''
    self.K = K
    self.rho = OptimFast.create_initrho(K)
    self.omega = (Data.nDoc/K + self.gamma) * np.ones(K)
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


  def _set_global_params_from_scratch(self, beta=None, topic_prior=None,
                                            Data=None, **kwargs):
    ''' Set rho, omega to values that reproduce provided appearance probs
    '''
    if topic_prior is not None:
      beta = topic_prior / topic_prior.sum()
    if beta is not None:
      Ktmp = beta.size
      rem = np.minimum(0.05, 1./(Ktmp))
      beta = np.hstack([np.squeeze(beta), rem])
      beta = beta/np.sum(beta)
    else:
      raise ValueError('Bad parameters. Vector beta not specified.')
    self.K = beta.size - 1
    self.rho, self.omega = self._convert_beta2rhoomega(beta, Data.nDoc)
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
    rho = OptimFast.beta2rho(beta, self.K)
    omega = (nDoc + self.gamma) * np.ones(rho.size)
    return rho, omega

  ####################################################### Calc ELBO
  #######################################################
  def calc_evidence(self, Data, SS, LP, **kwargs):
    ''' Calculate ELBO objective 
    '''
    UandcPi_global = self.E_logpU_logqU_c(SS)
    qcPi = self.q_cPi(LP)
    if SS.hasELBOTerms():
      ElogqZ = SS.getELBOTerm('ElogqZ')
    else:
      ElogqZ = self.E_logqZ(Data, LP)
    return UandcPi_global - np.sum(ElogqZ) - qcPi

  def q_cPi(self, LP):
    ''' Calculate c_Dir( DocTopicCount[d,:] + alpha E[beta] ) for each doc d

        Returns
        --------
        Elogstuff : real scalar
    ''' 
    alphaEbeta = self.alpha * self.E_beta()
    theta = LP['DocTopicCount'] + alphaEbeta[:-1]
    thetaRem = alphaEbeta[-1]
    return c_Dir(theta, thetaRem)

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

  def E_logpU_logqU_c(self, SS):
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
    OFFcoef = SS.nDoc * OptimFast.kvec(self.K) + self.gamma - g0

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
