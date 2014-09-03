'''
HDPTightPE.py
Bayesian nonparametric admixture model via the Hierarchical Dirichlet Process.
Uses a direct construction that maintains K active components.

Tight local (document-specific) factor.
Point estimate for global factor (not proper variational distribution).

Attributes
-------
K : # of components
gamma : scalar positive real, global concentration 
alpha : scalar positive real, document-level concentration param

References
-------
TODO
Latent Dirichlet Allocation, by Blei, Ng, and Jordan
introduces a classic admixture model with Dirichlet-Mult observations.
'''
import numpy as np

from HDPFast import HDPFast, c_Beta, c_Dir

from bnpy.suffstats import SuffStatBag
from ...util import digamma, gammaln
from ...util import NumericUtil, as1D

import OptimizerHDPTightPE as Optim
import LocalUtil

class HDPTightPE(HDPFast):

  def E_beta_active(self):
    ''' Return vector beta of appearance probabilities for all active topics
    '''
    return Optim.rho2beta_active(self.rho)

  def to_dict(self):
    return dict(rho=self.rho)

  def from_dict(self, Dict):
    self.inferType = Dict['inferType']
    self.K = Dict['K']
    self.rho = as1D(Dict['rho'])

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

  ### Inherited from HDPFast
  # def calcLogPrActiveCompsForDoc(self, DocTopicCount_d)
  # def updateLPGivenDocTopicCount(self, LP, DocTopicCount)
  # def initLPFromResp(self, Data, LP)

  ####################################################### Suff Stat Calc
  ####################################################### 
  ### Inherited from HDPFast

  ####################################################### Global Update
  ####################################################### 
  def update_global_params_VB(self, SS, rho=None, **kwargs):
    ''' Update global parameters.
    '''
    self.rho = self._find_optimum_rho(SS, **kwargs)
    self.K = SS.K
    self.ClearCache()

  def _find_optimum_rho(self, SS, **kwargs):
    ''' Run numerical optimization to find optimal rho point estimate

        Args
        --------
        SS : bnpy SuffStatBag, with K components

        Returns
        --------
        rho : 1D array, length K
    '''
    if hasattr(self, 'rho') and self.rho.size == SS.K:
      initrho = self.rho
    else:
      initrho = None

    try:
      rho, f, Info = Optim.find_optimum_multiple_tries(
                                        DocTopicCount=SS.DocTopicCount,
                                        nDoc=SS.nDoc,
                                        gamma=self.gamma, alpha=self.alpha,
                                        initrho=initrho)

    except ValueError as error:
      if str(error).count('FAILURE') == 0:
        raise error
      if initrho is not None:
        Log.error('***** Optim failed. Remain at cur val. ' + str(error))
        self.rho = initrho
      else:
        Log.error('***** Optim failed. Set to default init. ' + str(error))
        rho = Optim.create_initrho(K)
    return rho

  ####################################################### Set Global Params
  #######################################################
  def init_global_params(self, Data, K=0, **kwargs):
    self.K = K
    self.rho = Optim.create_initrho(K)
    self.ClearCache()

  def set_global_params(self, hmodel=None, 
                              rho=None, beta=None, 
                              **kwargs):
    if hmodel is not None:
      self.K = hmodel.allocModel.K
      if hasattr(hmodel.allocModel, 'rho'):
        self.rho = hmodel.allocModel.rho
      elif hasattr(hmodel.allocModel, 'rho'):
        self.rho = hmodel.allocModel.rho
      else:
        raise AttributeError('Unrecognized hmodel')
    elif rho is not None:
      self.rho = rho
      self.K = rho.size
    else:
      self._set_global_params_from_scratch(**kwargs)
    self.ClearCache()

  def _set_global_params_from_scratch(self, beta=None, topic_prior=None,
                                            **kwargs):
    ''' Set rho to values that reproduce provided appearance probs
    '''
    if topic_prior is not None:
      beta = topic_prior / np.sum(topic_prior)
    if beta is not None:
      Ktmp = beta.size
      rem = np.minimum(0.05, 1./(Ktmp))
      beta = np.hstack([np.squeeze(beta), rem])
      beta = beta/np.sum(beta)
    else:
      raise ValueError('Bad parameters. Vector beta not specified.')
    self.K = beta.size - 1
    self.rho = self._convert_beta2rho(beta)
    assert self.rho.size == self.K

  def _convert_beta2rho(self, beta):
    ''' Find stick-lengths rho that best recreate provided appearance probs beta

        Returns
        --------
        rho : 1D array, size K
    '''
    assert abs(np.sum(beta) - 1.0) < 0.001
    return Optim.beta2rho(beta, self.K)


  ####################################################### Calc ELBO
  #######################################################
  def calc_evidence(self, Data, SS, LP, **kwargs):
    ''' Calculate ELBO objective 
    '''
    U_global = self.E_logpU()
    c_Dir_Pi_p = SS.nDoc * c_Dir(self.alpha_E_beta())
    c_Dir_Pi_q = self.c_Dir_DocTopicCount(SS)
    if SS.hasELBOTerms():
      ElogqZ = SS.getELBOTerm('ElogqZ')
    else:
      ElogqZ = self.E_logqZ(Data, LP)
    return U_global + c_Dir_Pi_p - c_Dir_Pi_q - np.sum(ElogqZ)

  def E_logpU(self):
    ''' Calculate E[ log p(u) ]
    '''
    Elog1mU = np.log(1-self.rho)
    return self.K * c_Beta(1, self.gamma) \
           + np.sum((self.gamma - 1) * Elog1mU)

  def c_Dir_DocTopicCount(self, SS):
    ''' Calculate c_Dir( DocTopicCount[d,:] + alpha E[beta] ) for each doc d

        Returns
        --------
        Elogstuff : real scalar
    ''' 
    alphaEbeta = self.alpha * self.E_beta()
    theta = SS.DocTopicCount + alphaEbeta[:-1]
    thetaRem = alphaEbeta[-1]
    return c_Dir(theta, thetaRem)