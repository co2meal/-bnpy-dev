'''
MMSB.py

'''

import numpy as np
from scipy.cluster.vq import kmeans2

from bnpy.allocmodel import AllocModel
from bnpy.suffstats import SuffStatBag
from bnpy.util import gammaln, digamma, EPS
from bnpy.allocmodel.topics.HDPTopicModel import c_Beta, c_Dir

import cProfile

class FiniteMMSB(AllocModel):
  ######################################################### Constructors
  #########################################################
  def __init__(self, inferType, priorDict=dict()):
    if inferType.count('EM') > 0:
      raise NotImplementedError('EM not implemented for FiniteMMSB (yet)')

    self.inferType = inferType
    self.set_prior(**priorDict)
    self.K = 0

    #Variational parameter for pi
    self.theta = None

    self.estZ = [0]

  def set_prior(self, alpha = .1):
    self.alpha = alpha


  def get_active_comp_probs(self):
    print 'TODO TODO TODO'
    
  def getSSDims(self):
    '''Called during obsmodel.setupWithAllocModel to determine the dimensions
       of the statistics computed by the obsmodel.
       
       Overridden from the default of ('K',), as we need E_log_soft_ev to be
       dimension E x K x K
    '''
    return ('K', 'K',)


  ######################################################### Local Params
  #########################################################
  def calc_local_params(self, Data, LP, **kwargs):

    if self.inferType.count('EM') > 0:
      pass

    pr = cProfile.Profile()

    N = Data.nNodes
    K = self.K
    D = Data.dim
    ElogPi = digamma(self.theta) - \
             digamma(np.sum(self.theta, axis=1))[:,np.newaxis]
     

    if Data.isSparse: # sparse binary data. Too expensive to actually allocate
      sumSource = np.zeros((N,K))# phi, so all SS are computed in a loop here.
      sumReceiver = np.zeros((N,K))
      entropy = np.zeros((K,K))
      counts = np.zeros((K,K,2))
      phi_ij = np.zeros((K,K))
      
      logSoftEv = LP['E_log_soft_ev']
      assert logSoftEv.shape == (K,K,2)

      pr.enable()
      
      for i in xrange(Data.nNodes):
        if i % 20 == 0:
          print i
        for j in xrange(Data.nNodes):
          if i == j:
            continue
          if (i,j) in Data.edgeSet: # TODO : ACCOUNT FOR DIRECTEDNESS
            y_ij = 1
          else:
            y_ij = 0
          np.exp(ElogPi[i,:,np.newaxis] + ElogPi[j,np.newaxis,:] + \
                 logSoftEv[:,:,y_ij], out=phi_ij)
          phi_ij /= np.sum(phi_ij)

          #np.sum([ElogPi[i,:,np.newaxis], ElogPi[j,np.newaxis,:]], out=phi_ij)
          #np.exp(phi_ij + logSoftEv[:,:,y_ij], out=phi_ij)
          
          entropy += phi_ij * np.log(phi_ij + EPS)
          counts[:,:,y_ij] += phi_ij
          sumSource[i,:] += np.sum(phi_ij, axis=1)
          sumReceiver[j,:] += np.sum(phi_ij, axis=0)
      LP['sumSource'] = sumSource
      LP['sumReceiver'] = sumReceiver
      LP['Count1'] = counts[:,:,1]
      LP['Count0'] = counts[:,:,0]
      LP['entropy'] = entropy

    else:
      logSoftEv = LP['E_log_soft_ev'] # E x K x K
      logSoftEv[np.where(Data.sourceID == Data.destID),:,:] = 0
      logSoftEv = np.reshape(logSoftEv, (N, N, K, K))

      resp = np.zeros((N,N,K,K))
      # resp[i,j,l,m] = ElogPi[i,l] + ElogPi[j,m] + logSoftEv[i,j,l,m]
      resp = ElogPi[:,np.newaxis,:,np.newaxis] + \
             ElogPi[np.newaxis,:,np.newaxis,:] + logSoftEv
      np.exp(resp, out=resp)
      resp /= np.sum(resp, axis=(2,3))[:,:,np.newaxis,np.newaxis]
      resp[np.diag_indices(N)] = 0
      LP['resp'] = resp.reshape((N**2,K,K))
      LP['squareResp'] = resp

    pr.disable()
    from IPython import embed; embed()
    return LP

    
  ######################################################### Suff Stats
  #########################################################
    
  def get_global_suff_stats(self, Data, LP, doPrecompEntropy=None, **kwargs):

    if 'resp' in LP:
      K = LP['resp'].shape[-1]
    else:
      K = LP['sumSource'].shape[1]
    SS = SuffStatBag(K=K, D=Data.dim, N=Data.nNodes)

    # sumSource[i,l] = \sum_j E_q[s_{ijl}]
    if 'sumSource' in LP:
      sumSource = LP['sumSource']
    else:
      sumSource = np.sum(LP['squareResp'], axis=(1,3))
    SS.setField('sumSource', sumSource, dims=('N','K'))
      
    # sumReceiver[i,l] = \sum_j E_q[r_{jil}]
    if 'sumReceiver' in LP:
      sumReceiver = LP['sumReceiver']
    else:
      sumReceiver = np.sum(LP['squareResp'], axis=(0,2))
    SS.setField('sumReceiver', sumReceiver, dims=('N','K'))

    if 'resp' in LP:
      Npair = np.sum(LP['resp'], axis=0)
    else:
      Npair = LP['Count1'] + LP['Count0']
      #SS.setField('Count1', LP['Count1'], dims=('K','K'))
      #SS.setField('Count0', LP['Count0'], dims=('K','K'))
    SS.setField('N', Npair, dims=('K','K'))
    SS.setField('Npair', Npair, dims=('K','K'))
    
    return SS

  def forceSSInBounds(self, SS):
    ''' Force SS.respPairSums and firstStateResp to be >= 0.  This avoids
        numerical issues in moVB (where SS "chunks" are added and subtracted)
          such as:
            x = 10
            x += 1e-15
            x -= 10
            x -= 1e-15
          resulting in x < 0.

          Returns
          -------
          Nothing.  SS is updated in-place.
    '''
    np.maximum(SS.sumSource, 0, out=SS.sumSource)
    np.maximum(SS.sumReceiver, 0, out=SS.sumReceiver)
    

  

  ######################################################### Global Params
  #########################################################
  def update_global_params_VB(self, SS, **kwargs):
    self.theta = self.alpha + SS.sumSource + SS.sumReceiver
    self.calc_estZ()

  def calc_estZ(self):
    self.estZ = np.argmax(self.theta, axis=1)
      
  def init_global_params(self, Data, K=0, initname=None, **kwargs):
    N = Data.nNodes
    self.K = K
    if initname == 'kmeansRelational':
      X = np.reshape(Data.X, (N,N))
      centroids, labels = kmeans2(data=X, k=K, minit='points')
      self.theta = np.ones((N,K))*self.alpha
      for n in xrange(N):
        self.theta[n,labels[n]] += N-1
    elif initname == 'prior':
      self.theta = np.random.gamma(100, .01, size=(Data.nNodes,K))
    else:
      if self.inferType == 'EM':
        pass
      else:
        self.theta = self.alpha + np.ones((Data.nNodes,K))

  def set_global_params(self, hmodel=None, K=None, **kwargs):
    if hmodel is not None:
      self.K = hmodel.allocModel.K
      if self.inferType == 'EM':
        raise NotImplemetedError('EM not implemented (yet) for FiniteMMSB')
      elif self.inferType.count('VB') > 0:
        self.theta = hmodel.allocModel.theta

    else:
      self.K = K
      if self.inferType == 'EM':
        raise NotImplemetedError('EM not implemented (yet) for FiniteMMSB')
      elif self.inferType.count('VB') > 0:
        self.theta = theta

  ######################################################### Evidence
  #########################################################
  def calc_evidence(self, Data, SS, LP, **kwargs):
    alloc = self.elbo_alloc_no_slack(Data, LP)
    entropy = self.elbo_entropy(LP)
    return alloc + entropy

  def elbo_alloc_no_slack(self, Data, LP):
    N = Data.nNodes
    K = self.K
    p_cDir = N * (gammaln(K*self.alpha) - K*gammaln(self.alpha))
    q_cDir = np.sum(gammaln(np.sum(self.theta, axis=1))) - \
             np.sum(gammaln(self.theta))

    return p_cDir - q_cDir

  def elbo_entropy(self, LP):
    if 'entropy' in LP:
      return -np.sum(LP['entropy'])
    return -np.sum(LP['squareResp']*np.log(LP['squareResp']+EPS))
  



  ######################################################### IO Utils
  #########################################################   for machines
  def to_dict(self):
    return dict(theta=self.theta, estZ=self.estZ)

  def from_dict(self, myDict):
    self.inferType = myDict['inferType']
    self.K = myDict['K']
    self.theta = myDict['theta']

  def get_prior_dict(self):
    return dict(alpha=self.alpha)
