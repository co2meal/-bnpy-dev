'''
LDA.py
Bayesian parametric admixture model with a finite number of components K

Attributes
-------
K        : # of components
alpha   : scalar symmetric Dirichlet prior on topic appearance probabilities

Local Model Parameters (document-specific)
--------
z :  one-of-K topic assignment indicator for tokens
     z_{dn} : binary indicator vector for assignment of token n in document d
              z_{dnk} = 1 iff assigned to topic k, 0 otherwise.

pi : document-specific probabilities for each active topic 
     pi_d : 1D array, size K

Local Variational Parameters
--------
resp :  q(z_dn) = Categorical( z_dn | resp_{dn1}, ... resp_{dnK} )
theta : q(pi) = \prod_d q(\pi_d | \theta_d)
        q(pi_d) = Dirichlet( \theta_{d1}, ... \theta_{dK} )

Global Model Parameters (shared across all documents)
--------
None. No global structure is used except the prior parameter gamma0.
All structure is document-specific.

References
-------
Latent Dirichlet Allocation, by Blei, Ng, and Jordan
introduces a classic admixture model with Dirichlet-Mult observations.
'''
import numpy as np

from ..AllocModel import AllocModel
from bnpy.suffstats import SuffStatBag
from ...util import digamma, gammaln
from ...util import NumericUtil

import LocalUtil

class LDA(AllocModel):
  def __init__(self, inferType, priorDict=None):
    if inferType == 'EM':
      raise ValueError('LDA cannot do EM.')
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
    return np.ones(self.K) / float(self.K)
   
  def set_prior(self, alpha=1.0, **kwargs):
    self.alpha = float(alpha)

  def to_dict(self):
    return dict()              

  def from_dict(self, Dict):
    self.inferType = Dict['inferType']
    self.K = Dict['K']

  def get_prior_dict(self):
    return dict(alpha=self.alpha, 
                K=self.K,
                inferType=self.inferType)
    
  def get_info_string(self):
    ''' Returns human-readable name of this object
    '''
    return 'Finite LDA model with K=%d comps. alpha=%.2f' \
            % (self.K, self.alpha)
    

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
    assert 'theta' in LP
    assert 'DocTopicCount' in LP
    return LP

  def calcLogPrActiveCompsForDoc(self, DocTopicCount_d, out):
    ''' Calculate log prob of each of the K active topics given doc-topic counts

        Returns
        -------
        logp : 1D array, size K
               logp[k] gives probability of topic k in provided doc
    '''
    np.add(DocTopicCount_d, self.alpha, out=out)
    digammaSum = digamma(out.sum())
    digamma(out, out=out)
    out -= digammaSum
    return out

  def calcLogPrActiveComps_Fast(self, DocTopicCount, activeDocs=None, LP=None,
                                      out=None):
    ''' Calculate log prob of each active topic for each active document
    '''
    if out is None:
      theta = DocTopicCount + self.alpha
      ElogPi = digamma(theta)
      digammaThetaSum = digamma(theta.sum(axis=1))
      ElogPi -= digammaThetaSum[:,np.newaxis]
    elif activeDocs is None:
      np.add(DocTopicCount, self.alpha, out=out)
      digammaThetaSum = digamma(out.sum(axis=1))
      digamma(out, out=out)
      ElogPi = out
      ElogPi -= digammaThetaSum[:,np.newaxis]
    else:
      out[activeDocs] = DocTopicCount[activeDocs] + self.alpha
      digammaThetaSum = digamma(out[activeDocs].sum(axis=1))
      out[activeDocs] = digamma(out[activeDocs])
      ElogPi = out
      ElogPi[activeDocs] -= digammaThetaSum[:,np.newaxis]
    return ElogPi

  def updateLPGivenDocTopicCount(self, LP, DocTopicCount):
    ''' Update all local parameters, given topic counts for all docs in set.

        Returns
        --------
        LP : dict of local params, with updated fields
        * theta
        * ElogPi
    '''
    theta = DocTopicCount + self.alpha
    LP['theta'] = theta
    LP['ElogPi'] = digamma(theta) \
                   - digamma(theta.sum(axis=1))[:,np.newaxis]
    return LP

  ####################################################### Suff Stat Calc
  ####################################################### 
  def get_global_suff_stats(self, Data, LP, doPrecompEntropy=None, **kwargs):
    ''' Calculate sufficient statistics.
        Admixture models have no suff stats for allocation   
    '''
    resp = LP['resp']
    _, K = resp.shape
    SS = SuffStatBag(K=K, D=Data.get_dim())
    SS.setField('nDoc', Data.nDoc, dims=None)

    if doPrecompEntropy:
      ElogqZ = self.E_logqZ(Data, LP)
      Erest = self.E_logpPiZ_logqPi(Data, LP)
      SS.setELBOTerm('ElogqZ', ElogqZ, dims='K')
      SS.setELBOTerm('Erest', Erest, dims=None)
    return SS

  ####################################################### VB Global Step
  #######################################################
  def update_global_params(self, SS, rho=None, **kwargs):
    ''' Update global parameters.
    '''
    self.K = SS.K
        
  def set_global_params(self, K=0, **kwargs):
    self.K = K

  def init_global_params(self, Data, K=0, **kwargs):
    self.K = K

  ####################################################### Calc ELBO
  #######################################################   
  def calc_evidence(self, Data, SS, LP, **kwargs):
    ''' Calculate ELBO objective 
    '''
    if SS.hasELBOTerms():
      ElogqZ = SS.getELBOTerm('ElogqZ')
      Erest = SS.getELBOTerm('Erest')
    else:
      ElogqZ = self.E_logqZ(Data, LP)
      Erest = self.E_logpPiZ_logqPi(Data, LP)
    return Erest - np.sum(ElogqZ)

  def E_logqZ(self, Data, LP):
    ''' Calculate E[ log q(z)]
    '''
    if hasattr(Data, 'word_count'):
      return NumericUtil.calcRlogRdotv(LP['resp'], Data.word_count)
    else:
      return NumericUtil.calcRlogR(LP['resp'])

  def E_logpPiZ_logqPi(self, Data, LP):
    ''' Calculate E[ log p(pi) + log p(z) - log q(pi)  ]
    '''
    cDiff = Data.nDoc * c_Func(self.alpha, self.K) - c_Func(LP['theta'])
    logDirPDF = LP['DocTopicCount'] + self.alpha - LP['theta']
    logDirPDF *= LP['ElogPi']
    return cDiff + np.sum(logDirPDF)

def c_Func(avec, K=0):
  ''' Evaluate cumulant function of the Dirichlet distribution

      Returns
      -------
      c : scalar real
  '''
  if type(avec) == float or avec.ndim == 0:
    assert K > 0
    avec = avec * np.ones(K)
    return gammaln(np.sum(avec)) - np.sum(gammaln(avec))
  elif avec.ndim == 1:
    return gammaln(np.sum(avec)) - np.sum(gammaln(avec))  
  else:
    return np.sum(gammaln(np.sum(avec, axis=1))) - np.sum(gammaln(avec))
