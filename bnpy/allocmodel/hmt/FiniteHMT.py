import numpy as np

from bnpy.allocmodel import AllocModel
from bnpy.suffstats import SuffStatBag
from bnpy.allocmodel.tree import HMTUtil

class FiniteHMT(AllocModel):

	######################################################### Constructors
 	#########################################################

 	def __init__(self, inferType):
 		self.inferType = inferType
 		self.K = 0
 		self.initPi = None
 		self.transPi = None
 		self.initAlpha = 0.0
 		self.maxBranch = 0

 	def set_prior(self, initAlpha):
 		self.initAlpha = initAlpha

 	######################################################### Local Params
	#########################################################

	def calc_local_params(self, Data, LP, **kwargs):
		lpr = LP['E_log_soft_ev']
		if self.inferType.count('VB') > 0:
      print 'inferType VB yet not supported for FiniteHMT'
    elif self.inferType == 'EM' > 0:
      if self.initPi is None:
        self.initPi = np.ones(self.K)
        self.initPi /= self.K
      if self.transPi is None:
        self.transPi = np.ones((self.maxBranch, self.K, self.K))
        for b in xrange(self.maxBranch):
          for k in xrange(self.K):
            self.transPi[b,k,:] /= self.K
      gamma, psi, logMargPrSeq = HMTUtil.SumProductAlg_QuadTree(self.initPi, self.transPi, lpr)
      LP.update({'gamma':gamma})
      LP.update({'psi':psi})
      LP.update({'resp':gamma})
      LP.update({'evidence':logMargPrSeq})

      return LP

  ######################################################### Suff Stats
	#########################################################

	def get_global_suff_stats( self, Data, SS, LP ):
		if ('gamma' not in LP) or ('psi' not in LP):
      self.K = LP['resp'].shape[1]
      gamma = np.ones((Data.nObs, self.K)) / self.K
      psi = np.ones((Data,nObs, self.K, self.K)) / (self.K * self.K)
      LP.update({'gamma':gamma})
      LP.update({'psi':psi})
          
    gamma = LP['gamma']
    psi = LP['psi']
    
    gamma1 = gamma[0,:]
    N = np.sum(gamma, axis = 1)
    for b in xrange(self.maxBranch):
      psiSums = np.sum(psi[Data.mask[b],:,:], axis = 0)
      SS.setField('psiSums'+str(b), psiSums, dims=('K','K'))

    SS = SuffStatBag(K = self.K , D = Data.dim)
    SS.setField('gamma1', gamma1, dims=('K'))
    SS.setField('N', N, dims=('K'))

    return SS

  ######################################################### Global Params
	#########################################################
	def update_global_params_EM( self, SS, **kwargs ):
    self.K = SS.K

    if (self.initPi is None) or (self.transPi is None):
      self.initPi = np.ones(self.K)
      self.transPi = np.ones((self.maxBranch, self.K, self.K))

    self.initPi = (SS.gamma1 + self.initAlpha) / (SS.gamma1.sum() + self.K * self.initAlpha)

    for b in xrange(self.maxBranch):
      normFactor = np.sum(SS.psiSums, axis = 1)
      psiSums = getattr(SS._Fields, 'psiSums'+str(b))
      for k in xrange(SS.K):
        self.transPi[b,k,:] = psiSums[k,:] / normFactor[k]

  def set_global_params(self, hmodel=None, K=None, initPi=None, transPi=None, maxBranch=None,**kwargs):
    if hmodel is not None:
      self.K = hmodel.allocModel.K
      self.initPi = hmodel.allocModel.initPi
      self.transPi = hmodel.allocModel.transPi
      if maxBranch is None:
        self.maxBranch = 4
      else:
        self.maxBranch = maxBranch
    else:
      self.K = K
      self.initPi = initPi
      self.transPi = transPi
      self.maxBranch = maxBranch

  def calc_evidence(self, Data, SS, LP):
    if self.inferType == 'EM':
      return LP['evidence']

  def to_dict(self):
    if self.inferType == 'EM':
      return dict(initPi = self.initPi, transPi = self.transPi)

  def get_branch(child_index):
    '''Find on which branch of its parent this child lies
    '''
    if (child_index%4 == 0):
      return 4
    else:
      return (child_index%4 - 1)