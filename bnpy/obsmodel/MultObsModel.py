'''
'''
import itertools
import numpy as np

from ..distr import DirichletDistr
from ..distr import MultinomialDistr

from ..util import LOGTWO, LOGPI, LOGTWOPI, EPS
from ..util import np2flatstr, dotATA, dotATB, dotABT
from ..util import MVgammaln, MVdigamma

EPS = np.finfo(float).eps

class MultObsModel( object ):

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
    if self.qType == 'EM':
      return '\n'.join( [np2flatstr(self.qobsDistr[k].phi, fmtStr) for k in xrange(self.K)] )
    else:
      return '\n'.join( [np2flatstr(self.qobsDistr[k].lamvec/self.qobsDistr[k].lamsum, fmtStr) for k in xrange(self.K)] )

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
    resp = LP['resp']
    try:      
      SS['TermCount'] = np.dot( resp.T, Data['X'] )
      return SS
    except KeyError:
      TermCountMat  = np.zeros( (self.K, self.D) )      
      for docID in xrange( Data['nGroup'] ):
        docResp = LP['resp'][ Data['GroupIDs'][docID][0]:Data['GroupIDs'][docID][1] ]
        TermCountMat[:, Data['wordIDs_perGroup'][docID] ]  += docResp.T
        assert np.allclose( docResp.sum(axis=1), Data['wordCounts_perGroup'][docID] )
        #TermCountMat[:, Data['wordIDs_perGroup'][docID] ]  += Data['wordCounts_perGroup'][docID]*docResp.T
      SS['TermCount'] = TermCountMat
    return SS

  ################################################################## Param updates
  def update_global_params( self, SS, rho=None, Ntotal=None, **kwargs):
    ''' M-step update
    '''
    if self.qType == 'EM':
        self.update_obs_params_EM( SS)
    elif self.qType.count('VB')>0:
      if rho is None:
        self.update_obs_params_VB( SS )
      else:
        self.update_obs_params_VB_stochastic( SS, rho, Ntotal )

  def update_obs_params_EM( self, SS, **kwargs):
    phiHat = SS['TermCount']
    phiHat = phiHat/( EPS+ phiHat.sum(axis=1)[:,np.newaxis] )
    for k in xrange( self.K ):
      self.qobsDistr[k] = MultinomialDistr( phiHat[k] )

  def update_obs_params_VB( self, SS, **kwargs):
    for k in xrange( self.K):
      self.qobsDistr[k] = self.obsPrior.getPosteriorDistr( SS['TermCount'][k] )

  def update_obs_params_VB_stochastic( self, SS, rho, Ntotal, **kwargs):
    ampF = Ntotal/SS['Ntotal']
    for k in xrange( self.K):
      postDistr = self.obsPrior.getPosteriorDistr( ampF*SS['TermCount'][k] )
      if self.qobsDistr[k] is None:
        self.qobsDistr[k] = postDistr
      else:
        self.qobsDistr[k].rho_update( rho, postDistr )
      
  #########################################################  Soft Evidence Fcns  
  def calc_local_params( self, Data, LP):
    if self.qType == 'EM':
      LP['E_log_soft_ev'] = self.log_soft_ev_mat( Data )
    else:
      LP['E_log_soft_ev'] = self.E_log_soft_ev_mat( Data )
    return LP

  def log_soft_ev_mat( self, Data ):
    ''' E-step update,  for EM-type
    '''
    lpr = np.empty( (Data['nObs'], self.K) )
    for k in xrange( self.K ):
      lpr[:,k] = self.qobsDistr[k].log_pdf( Data )
    return lpr
      
  def E_log_soft_ev_mat( self, Data ):
    ''' E-step update, for VB-type
    '''
    lpr = np.empty( (Data['nObs'], self.K) )
    for k in xrange( self.K ):
      lpr[:,k] = self.qobsDistr[k].E_log_pdf( Data )
    return lpr
  
  #########################################################  Evidence Bound Fcns  
  def calc_evidence( self, Data, SS, LP):
    if self.qType == 'EM':
      return 0 # handled by alloc model
    return self.E_logpX( LP, SS) + self.E_logpPhi() - self.E_logqPhi()
  
  def E_logpX( self, LP, SS ):
    ''' E_{q(Z), q(Phi)} [ log p(X) ]
    '''
    lpX = np.zeros( self.K )
    for k in xrange( self.K ):
      lpX[k] = np.sum( SS['TermCount'][k] * self.qobsDistr[k].Elogphi )
    return lpX.sum()
    
  def E_logpPhi( self ):
    lp = self.obsPrior.get_log_norm_const()*np.ones( self.K)
    for k in xrange( self.K):
      lp[k] += np.sum( (self.obsPrior.lamvec - 1)*self.qobsDistr[k].Elogphi )
    return lp.sum()
          
  def E_logqPhi( self ):
    ''' Return negative entropy!
    '''    
    lp = np.zeros( self.K)
    for k in xrange( self.K):
      lp[k] = self.qobsDistr[k].get_entropy()
    return -1*lp.sum()
