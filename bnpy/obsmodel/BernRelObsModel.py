''' 
BagOfWordsObsModel.py
'''
import numpy as np
import copy
from scipy.special import digamma, gammaln

from ..util import np2flatstr, EPS
from ..distr import BetaDistr
from ObsModel import ObsModel


class BernRelObsModel(ObsModel):
    
  ######################################################### Constructors
  #########################################################
  def __init__(self, inferType, obsPrior=None):
    self.inferType = inferType
    self.obsPrior = obsPrior
    self.comp = list()

  @classmethod
  def CreateWithPrior(cls, inferType, priorArgDict, Data):
    if inferType == 'EM':
      raise NotImplementedError('TODO')
    else:
      obsPrior = BetaDistr.InitFromData(priorArgDict, Data)
    return cls(inferType, obsPrior)

  @classmethod
  def CreateWithAllComps(cls, oDict, obsPrior, compDictList):
    ''' Create MultObsModel, all K component Distr objects,
         and the prior Distr object in one call
    '''
    if oDict['inferType'] == 'EM':
      raise NotImplementedError("TODO")

    self = cls(oDict['inferType'], obsPrior=obsPrior)
    self.K = len(compDictList)
    self.comp = [None for k in range(self.K)]
    for k in xrange(self.K):
      self.comp[k] = BetaDistr(**compDictList[k])
    return self

  ######################################################### Accessors
  #########################################################  
  def Elogphi(self):
    digLam = np.empty(self.Lam.shape)
    digamma(self.Lam, out=digLam)
    digLam -= digamma(self.LamSum)[:,np.newaxis]
    return digLam

  def ElogphiOLD(self):
    return digamma(self.Lam) - digamma(self.LamSum)[:,np.newaxis]

  ######################################################### Local Params
  #########################################################   E-step
  def calc_local_params(self, Data, LP, **kwargs):
    ''' Calculate local parameters (E-step)

        Returns
        -------
        LP : bnpy local parameter dict, with updated fields
        E_logsoftev_WordsData : nDistinctWords x K matrix, where
                               entry n,k = log p(word n | topic k)
    '''
    if self.inferType == 'EM':
      raise NotImplementedError('TODO')
    else:
      LP['E_logsoftev_WordsData'] = self.Elogphi.T[Data.word_id,:]
    return LP

  ######################################################### Suff Stats
  #########################################################
  def get_global_suff_stats(self, Data, SS, LP, **kwargs):
    ''' Calculate and return sufficient statistics.

        Returns
        -------
        SS : bnpy SuffStatDict object, with updated fields
                WordCounts : K x VocabSize matrix
                  WordCounts[k,v] = # times vocab word v seen with topic k
    '''
    wv = LP['word_variational']
    WMat = Data.to_sparse_matrix()
    TopicWordCounts = (WMat * wv).T

    SS.setField('WordCounts', TopicWordCounts, dims=('K','D'))
    SS.setField('N', np.sum(TopicWordCounts,axis=1), dims=('K'))
    return SS

  ######################################################### Global Params
  #########################################################   M-step

  def update_obs_params_EM(self, SS, **kwargs):
    raise NotImplementedError("TODO")

  def update_obs_params_VB(self, SS, mergeCompA=None, **kwargs):
    if mergeCompA is None:
      self.Lam[mergeCompA] = SS.WordCounts[mergeCompA] + self.obsPrior.lamvec    
    else:
      self.Lam = SS.WordCounts + self.obsPrior.lamvec    

  def update_obs_params_soVB( self, SS, rho, **kwargs):
    raise NotImplementedError("TODO")  

  def set_global_params(self, hmodel=None, lamA=None, lamB=None, theta=None, **kwargs):
    ''' Set global params to provided values

        Params
        --------
        sb : K x K matrix, each entry represents
        topics[k,v] = probability of word v under topic k
    '''
    if hmodel is not None:
        self.K = hmodel.obsModel.K
        self.comp = copy.deepcopy(hmodel.obsModel.comp)
        return
    self.K = theta.shape[1]
    self.comp = list()

    for k in range(self.K):
        # Scale up Etopics to lamvec, a V-len vector of positive entries,
        #   such that (1) E[phi] is still Etopics, and
        #             (2) lamvec = obsPrior.lamvec + [some suff stats]
        #   where (2) means that lamvec is a feasible posterior value
        ii = np.argmin(topics[k,:])
        lamvec = self.obsPrior.lamvec[ii]/topics[k,ii] * topics[k,:]
        # Cut-off values that are way way too big
        if np.any( lamvec > 1e9):
          lamvec = np.minimum(lamvec, 1e9)
          print "WARNING: mucking with lamvec"
        self.comp.append(BetaDistr(lamA, lamB))


  ######################################################### Evidence
  #########################################################
  def calc_evidence(self, Data, SS, LP):
    elbo_pData = self.Elogp_Words(SS)
    elbo_pLam  = self.Elogp_Lam()
    elbo_qLam  = self.Elogq_Lam()
    return elbo_pData + elbo_pLam - elbo_qLam

  def Elogp_Edges(self, SS):
    '''
    '''
    return np.sum(SS.WordCounts * self.Elogphi())

  def Elogp_Lam(self):
    '''
    '''
    logNormC = -1 * self.obsPrior.get_log_norm_const()
    logDirPDF = np.dot(self.Elogphi(), self.obsPrior.lamvec - 1.0)
    return np.sum(logDirPDF + logNormC)

  def Elogq_Lam(self):
    '''
    '''
    logNormC = np.sum(gammaln(self.Lam)) - np.sum(gammaln(self.LamSum))
    Elogphi = self.Elogphi() 
    logDirPDF = np.sum(Elogphi * (1.0 - self.Lam))
    return -1.0 * (logNormC + logDirPDF)


  ######################################################### I/O Utils
  #########################################################  for humans
  def get_info_string(self):
    return 'Multinomial distribution'
  
  def get_info_string_prior(self):
    if self.obsPrior is None:
      return 'None'
    else:
      lamvec = self.obsPrior.lamvec
      if np.allclose(lamvec, lamvec[0]):
        return 'Symmetric Dirichlet, lambda=%.2f' % (lamvec[0])
      else:
        return 'Dirichlet, lambda %s' % (np2flatstr(lamvec[:3]))

  ######################################################### I/O Utils
  #########################################################  for machines
  def get_prior_dict( self ):
    return self.obsPrior.to_dict()
 
