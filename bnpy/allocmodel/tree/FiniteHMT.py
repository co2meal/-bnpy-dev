import numpy as np
import matplotlib.pyplot as plt

from bnpy.allocmodel import AllocModel
from bnpy.suffstats import SuffStatBag
from bnpy.allocmodel.tree import QuadTreeUtil as HMTUtil
from bnpy.allocmodel.tree import HMTViterbi
from bnpy.util import digamma, gammaln

class FiniteHMT(AllocModel):

  ######################################################### Constructors
  #########################################################

  def __init__(self, inferType, priorDict=dict()):
    self.inferType = inferType
    self.K = 0
    self.initPi = None
    self.transPi = None
    self.initTheta = None
    self.transTheta = None
    self.maxBranch = 4
    self.set_prior(**priorDict)

  def set_prior(self, initAlpha=0.1, transAlpha=0.1, **kwargs):
    self.initAlpha = initAlpha
    self.transAlpha = transAlpha

  def get_active_comp_probs(self):
    if self.inferType == 'EM':
      pi = self.transPi.sum(axis=0) / self.maxBranch
      return pi.mean(axis=0)
    else:
      theta = self.transTheta.sum(axis=0) / self.maxBranch
      EPiMat = self.theta / self.theta.sum(axis=1)[:,np.newaxis]
      return EPiMat.mean(axis=0)

  ######################################################### Local Params
  #########################################################

  def calc_local_params(self, Data, LP, **kwargs):
    logSoftEv = LP['E_log_soft_ev']
    K = logSoftEv.shape[1]
    if self.inferType.count('VB') > 0:
      expELogInit = np.exp(digamma(self.initTheta) - 
                           digamma(np.sum(self.initTheta)))
      expELogTrans = np.empty( (self.maxBranch, K, K) )
      for b in xrange(self.maxBranch):
        digammasumvec = digamma(np.sum(self.transTheta[b,:,:], axis = 1))
        expELogTrans[b,:,:] = np.exp(digamma(self.transTheta[b,:,:]) 
                                     - digammasumvec[:,np.newaxis])
      initParam = expELogInit
      transParam = expELogTrans
    elif self.inferType == 'EM' > 0:
      #encoding = HMTViterbi.ViterbiAlg(self.initPi, self.transPi, logSoftEv)
      #plt.scatter(Data.X[:,0], Data.X[:,1], c=encoding, alpha=.7)
      #plt.show()
      #if self.image % 5 == 0:
      #  plt.savefig('/home/mterzihan/Desktop/denoising/trial/%d.png' % self.image)
      #self.image = self.image+1
      initParam = self.initPi
      transParam = self.transPi
    logMargPr = np.empty(Data.nDoc)
    resp = np.empty((Data.nObs, K))
    respPair = np.empty((Data.nObs, K, K))
    respPair[0].fill(0)
    for n in xrange(Data.nDoc):
      start = Data.doc_range[n]
      stop = Data.doc_range[n+1]
      logSoftEv_n = logSoftEv[start:stop]
      treeResp, treeRespPair, treeLogMargPr = HMTUtil.SumProductAlg_QuadTree(initParam, transParam, logSoftEv_n)
      resp[start:stop] = treeResp
      respPair[start:stop] = treeRespPair
      logMargPr[n] = treeLogMargPr
    LP['evidence'] = np.sum(logMargPr)
    LP['resp'] = resp
    LP['respPair'] = respPair

    return LP

  ######################################################### Suff Stats
  #########################################################

  def get_global_suff_stats( self, Data, LP , doPrecompEntropy=None, **kwargs):   
    resp = LP['resp']
    respPair = LP['respPair']
    K = resp.shape[1]
    SS = SuffStatBag(K=K, D=Data.dim)

    ## Count usage of each state across all tokens
    N = np.sum(resp, axis = 0)
    SS.setField('N', N, dims=('K'))

    ## Count usage of each state at "root" of each tree
    startLocIDs = Data.doc_range[:-1]
    firstStateResp = np.sum(resp[startLocIDs], axis = 0)
    SS.setField('FirstStateCount', firstStateResp, dims=('K'))

    ## Count pair-wise co-occurances for all state pairs
    # We loop over each tree (aka "document")
    #  identify which nodes belong to each branch, & sum over these
    # TODO: make this faster? avoid loop over docs.
    for b in xrange(self.maxBranch):
      bkey = 'PairCounts' + str(b)
      bPairCounts = np.zeros((self.K, self.K))
      for docID in xrange(Data.nDoc):
        bmask = range(Data.doc_range[docID] + b + 1,
                      Data.doc_range[docID+1],
                      self.maxBranch)
        bPairCounts += np.sum(respPair[bmask], axis=0)
      SS.setField(bkey, bPairCounts, dims=('K', 'K'))

    if doPrecompEntropy is not None:
      entropy = self.elbo_entropy(Data, LP)
      SS.setELBOTerm('Elogqz', entropy, dims = (()))

    return SS

  ######################################################### Global Params
  #########################################################
  def update_global_params_EM( self, SS, **kwargs ):
    self.K = SS.K

    self.initPi = (SS.FirstStateCount) / (SS.FirstStateCount.sum())
    
    for b in xrange(self.maxBranch):
      PairCounts = getattr(SS._Fields, 'PairCounts'+str(b))
      normFactor = np.sum(PairCounts, axis = 1)
      self.transPi[b,:,:] = PairCounts / normFactor[:,np.newaxis]

  def update_global_params_VB( self, SS, **kwargs ):
    self.initTheta = self.initAlpha + SS.FirstStateCount
    for b in xrange(self.maxBranch):
      bkey = 'PairCounts' + str(b)
      self.transTheta[b,:,:] = self.transAlpha + getattr(SS._Fields, bkey)
    self.K = SS.K

  def init_global_params(self, Data, K=0, **kwargs):
    self.K = K
    if self.inferType == 'EM':
      self.initPi = 1.0/K * np.ones(K)
      self.transPi = np.empty((self.maxBranch, K, K))
      for b in xrange(self.maxBranch):
        self.transPi[b,:,:] = np.ones(K)[:,np.newaxis]/K * np.ones((K,K))
    else:
      self.initTheta = self.initAlpha + np.ones(K)
      self.transTheta = self.transAlpha + np.ones((self.maxBranch,K,K))

  def set_global_params(self, trueParams=None, hmodel=None, K=None, initPi=None, transPi=None, maxBranch=None,**kwargs):
    if hmodel is not None:
      self.K = hmodel.allocModel.K
      if maxBranch is None:
        self.maxBranch = 4
      else:
        self.maxBranch = maxBranch
      if self.inferType == 'EM':
        self.initPi = hmodel.allocModel.initPi
        self.transPi = hmodel.allocModel.transPi
      elif self.inferType == 'VB':
        self.initTheta = hmodel.allocModel.initTheta
        self.transTheta = hmodel.allocModel.transTheta
    elif trueParams is not None:
      self.mu = trueParams[mu]
      self.Sigma = trueParams[Sigma]
      if self.inferType == 'EM':
        self.initPi = trueParams[initPi]
        self.transPi = trueParams[transPi]
      elif self.inferType == 'VB':
        self.initTheta = trueParams[initTheta]
        self.transTheta = trueParams[transTheta]
    else:
      self.K = K
      self.maxBranch = maxBranch
      if self.inferType == 'EM':
        self.initPi = initPi
        self.transPi = transPi
      elif self.inferType == 'VB':
        self.initTheta = initTheta
        self.transTheta = transTheta

  def calc_evidence(self, Data, SS, LP, todict=False, **kwargs):
    if self.inferType == 'EM':
      return LP['evidence']
    elif self.inferType.count('VB') > 0:
      if SS.hasELBOTerm('Elogqz'):
        entropy = SS.getELBOTerm('Elogqz')
      else:
        entropy = self.elbo_entropy(Data, LP)

      return entropy + self.elbo_alloc__forloop()
    else :
      emsg = 'Unrecognized inferType: ' + self.inferType
      raise NotImplementedError(emsg)

  def elbo_entropy(self, Data, LP):
    return HMTUtil.calcEntropyFromResp(LP['resp'], LP['respPair'], Data)

  def elbo_alloc__forloop(self):
    ''' Calculate term in objective related to transition parameters.

        Math
        --------
        E[ log p(z) + log p(pi) - log q(pi) ]

        Returns
        --------
        scalar
    '''
    cDirAlpha_init = gammaln(self.K * self.initAlpha) \
                     - self.K * gammaln(self.initAlpha)
    cDirTheta_init = gammaln(np.sum(self.initTheta)) \
                     - np.sum(gammaln(self.initTheta))
    cDirAlpha_trans = 0
    cDirTheta_trans = 0
    for b in xrange(self.maxBranch):
      for j in xrange(self.K):
        cDirAlpha_trans += gammaln(self.K * self.transAlpha) \
                          - self.K * gammaln(self.transAlpha)
        cDirTheta_trans += gammaln(np.sum(self.transTheta[b,j,:])) \
                          - np.sum(gammaln(self.transTheta[b,j,:]))
    return cDirAlpha_init - cDirTheta_init \
           + cDirAlpha_trans - cDirTheta_trans


  def from_dict(self, myDict):
    self.inferType = myDict['inferType']
    self.K = myDict['K']
    if self.inferType == 'VB':
      self.initTheta = myDict['initTheta']
      self.transTheta = myDict['transTheta']
    elif self.inferType == 'EM':
      self.initPit = myDict['initPi']
      self.transPi = myDict['transPi']

  def to_dict(self):
    if self.inferType == 'EM':
      return dict(initPi = self.initPi, transPi = self.transPi)
    elif self.inferType.count('VB') > 0:
      return dict(initTheta=self.initTheta, 
            transTheta=self.transTheta)

  def get_prior_dict(self):
    return dict(initAlpha=self.initAlpha, transAlpha=self.transAlpha, K=self.K)

  def get_active_comp_probs(self):
    ''' Return K vector of appearance probabilities for each of the K comps
    '''
    return np.ones(self.K) / float(self.K)
