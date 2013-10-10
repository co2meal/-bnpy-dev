'''
'''
import itertools
import numpy as np

from ..distr import DirichletDistr
from ..distr import MultinomialDistr

from ..util import LOGTWO, LOGPI, LOGTWOPI, EPS
from ..util import np2flatstr, dotATA, dotATB, dotABT
from ..util import MVgammaln, MVdigamma

from ObsCompSet import ObsCompSet

class MultObsModel( ObsCompSet ):

  def __init__( self, inferType, W=None, obsPrior=None,):
    self.inferType = inferType
    self.obsPrior = obsPrior
    self.W = None #vocabulary size
    self.comp = list()

  @classmethod
  def InitFromData(cls, inferType, priorArgDict, Data):
    ''' Create GaussObCompSet and its prior distr in one call
        The resulting object then needs to be initialized via init_global_params,
        otherwise it has no components and can't be used in learn algs.
    '''
    D = Data.D
    if inferType == 'VB':
      # Defines the Dirichlet (topic x word) that is initialized to ensure it fits data dimensions
      obsPrior = DirichletDistr.InitFromData(priorArgDict, Data)
      return cls(inferType, D, obsPrior)

  def get_info_string(self):
    return 'Multinomial distribution'
  
  def get_info_string_prior(self):
    if self.obsPrior is None:
      return 'None'
    else:
      return 'Dirichlet'

  def get_human_global_param_string(self, fmtStr='%3.2f'):
    if self.inferType == 'EM':
      return '\n'.join( [np2flatstr(self.obsPrior[k].phi, fmtStr) for k in xrange(self.K)] )
    else:
      return '\n'.join( [np2flatstr(self.obsPrior[k].lamvec/self.obsPrior[k].lamsum, fmtStr) for k in xrange(self.K)] )

  def set_obs_dims( self, Data):
    self.D = Data['nVocab']
    if self.obsPrior is not None:
      self.obsPrior.set_dims( self.D )

  def save_params( self, filename):
    pass

  ################################################################## Suff stats
  def get_global_suff_stats( self, Data, SS, LP ):
    ''' Suff Stats
    '''
    # Grab topic x word sufficient statistics
    resp = LP['resp']
    word_count = Data.word_count
    V = Data.V
    total_obs,K = resp.shape
    lambda_kw = np.zeros( (K, V) )
    # Loop through documents
    ii = 0
    for d in xrange(len(word_count)):
        for word_id, word_freq in word_count[d].iteritems():
            lambda_kw[:, word_id] += resp[ii,:] * word_freq  
            ii += 1
    # Return K x V matrix of sufficient stats (topic x word)
    SS.lambda_kw = lambda_kw
    return SS

  def update_obs_params_EM( self, SS, **kwargs):
    phiHat = SS['TermCount']
    phiHat = phiHat/( EPS+ phiHat.sum(axis=1)[:,np.newaxis] )
    for k in xrange( self.K ):
      self.obsPrior[k] = MultinomialDistr( phiHat[k] )

  def update_obs_params_VB( self, SS, Krange, **kwargs):
    for k in Krange:
      self.comp[k] = self.obsPrior.get_post_distr( SS, k )

  def update_obs_params_VB_stochastic( self, SS, rho, Ntotal, **kwargs):
    ampF = Ntotal/SS['Ntotal']
    for k in xrange( self.K):
      postDistr = self.obsPrior.getPosteriorDistr( ampF*SS['TermCount'][k] )
      if self.obsPrior[k] is None:
        self.obsPrior[k] = postDistr
      else:
        self.obsPrior[k].rho_update( rho, postDistr )
      
  #########################################################  Soft Evidence Fcns  
  def calc_local_params( self, Data, LP):
    if self.inferType == 'EM':
      LP['E_log_soft_ev'] = self.log_soft_ev_mat( Data )
    else:
      LP['E_log_soft_ev'] = self.E_log_soft_ev_mat( Data )
    return LP

  def log_soft_ev_mat( self, Data ):
    ''' E-step update,  for EM-type
    '''
    lpr = np.empty( (Data['nObs'], self.K) )
    for k in xrange( self.K ):
      lpr[:,k] = self.obsPrior[k].log_pdf( Data )
    return lpr
      
  def E_log_soft_ev_mat( self, Data ):
    ''' E-step update, for word tokens
    '''
    word_count = Data.word_count
    lpr = np.empty( (Data.nObsTotal, self.K) )
    lambda_kw = np.zeros((self.K, Data.V))
    for k in xrange(self.K):
        lambda_kw[k,:] = self.comp[k].Elogphi # returns topic by word matrix expectations
    ii = 0
    for d in xrange( Data.D ):
       for (word_id,count) in enumerate(word_count[d]):
           lpr[ii,:] = lambda_kw[:,word_id]
           ii += 1
    return lpr
  
  #########################################################  Evidence Bound Fcns  
  def calc_evidence( self, Data, SS, LP):
    if self.inferType == 'EM':
      return 0 # handled by alloc model
    # Calculate p(w | z, lambda) + p(lambda) - q(lambda)
    pw = self.E_logpX( LP, SS) 
    po = self.E_logpPhi()
    qo = self.E_logqPhi()
    lb_obs = pw + po - qo
    print "pw: " + str(pw)
    print "po: " + str(po)
    print "qo: " + str(qo)
    return lb_obs
  
  def E_logpX( self, LP, SS ):
    ''' E_{q(Z), q(Phi)} [ log p(X) ]
    '''
    K,W = SS.lambda_kw.shape
    doc_topic_weights = LP['doc_topic_weights']
    elambda_kw = np.zeros( (K, W) )
    for k in xrange( K ):
        elambda_kw[k,:] = self.comp[k].Elogphi
        
    lpX = np.dot(doc_topic_weights, elambda_kw)
    return lpX.sum()
    
  def E_logpPhi( self ):
    lp = self.obsPrior.get_log_norm_const()*np.ones( self.K)
    for k in xrange( self.K):
      lp[k] += np.sum( (self.obsPrior.lamvec - 1)*self.comp[k].Elogphi )
    return lp.sum()

  def get_prior_dict( self ):
    PDict = self.obsPrior.to_dict()
    return PDict
  
  def E_logqPhi( self ):
    ''' Return negative entropy!
    '''    
    lp = np.zeros( self.K)
    for k in xrange( self.K):
      lp[k] = self.comp[k].get_entropy()
    return -1*lp.sum()
