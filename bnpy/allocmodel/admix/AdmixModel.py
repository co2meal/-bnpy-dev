'''
  MixModel.py
     Bayesian parametric admixture model with a finite number of components K

  Provides code for performing variational Bayesian inference,
     using a mean-field approximation.
     
 Author: Mike Hughes (mike@michaelchughes.com)

 Parameters
 -------
    K        : # of components
    alpha0   : scalar hyperparameter of symmetric Dirichlet prior on mix. weights

 References
 -------
   Latent Dirichlet Allocation, by Blei, Ng, and Jordan
      introduces a classic admixture model with Dirichlet-Mult observations.
'''
from IPython import embed
import numpy as np

from ..AllocModel import AllocModel
from bnpy.suffstats import SuffStatDict
from ...util import digamma, gammaln, logsumexp
from ...util import EPS, np2flatstr

class AdmixModel( AllocModel ):

  def __init__( self, inferType, priorDict=None):
    if inferType == "EM":
      raise ValueError('AdmixModel cannot do EM. Only VB learning possible.')
    self.inferType = inferType
    self.K = 0
    if priorDict is None:
      self.alpha0 = 1.0 # Uniform!
    else:
      self.set_prior(priorDict)

  def set_prior(self, PriorParamDict):
    self.alpha0 = PriorParamDict['alpha0']
    
  def to_dict( self ):
    return dict()  	    	
  
  def from_dict(self, Dict):
    pass
      	
  def get_prior_dict( self ):
    return dict( alpha0=self.alpha0, K=self.K, qType=self.qType )
    
  def get_info_string( self):
    ''' Returns human-readable name of this object
    '''
    return 'Finite admixture model with %d components | alpha=%.2f' % (self.K, self.alpha0)

  def get_human_global_param_string(self):
    ''' Returns human-readable numerical repr. of parameters,
          for quick inspection of correctness
    '''
    mystr = ''
    for rowID in xrange(3):
      mystr += np2flatstr( np.exp(self.Elogw[rowID]), '%3.2f') + '\n'
    return mystr
 
  def is_nonparametric(self):
    return False 

  def need_prev_local_params(self):
    return True

  ##############################################################    
  ############################################################## Suff Stat Calc   
  ##############################################################
  def get_global_suff_stats( self, Data, LP, doPrecompEntropy=None, **kwargs):
    ''' Just count expected # assigned to each cluster across all Docs, as usual
    '''
    phi = LP['phi']
    K,total_obs = phi.shape
    word_count = Data.word_count
    groupid = Data.groupid
    D = Data.D
    Nvec = np.zeros( (K,D) )
    # Loop through documents
    for d in xrange(D):
        start,stop = groupid[d]
        # get document-level sufficient statistics 
        Nvec[:, d] = np.dot( phi[:,start:stop], word_count[d].values() ) 
    SS = SuffStatDict(N=Nvec)
    SS.K = K
    return SS
     
    
  ##############################################################    
  ############################################################## Local Param Updates   
  ##############################################################
  def calc_local_params( self, Data, LP ):
    ''' E-step
          alternate between these updates until convergence
             q(Z)  (posterior on topic-token assignment)
         and q(W)  (posterior on Doc-topic distribution)
    '''
    try:
      LP['N_perDoc']
    except KeyError:
      LP['N_perDoc'] = np.zeros( (Data['nDoc'],self.K) )

    DocIDs = Data['DocIDs']
    nDocs = Data['nDoc']
    prevVec = None
    for rep in xrange( 4 ):
      LP = self.get_doc_theta( Data, LP)
      LP = self.get_word_phi( Data, LP)
      for gg in range( nDocs ):
        DocResp = LP['resp'][ DocIDs[gg][0]:DocIDs[gg][1] ]
        LP['N_perDoc'][gg] = np.sum( DocResp, axis=0 )

      curVec = LP['alpha_perDoc'].flatten()
      if prevVec is not None and np.allclose( prevVec, curVec ):
        break
      prevVec = curVec
    return LP
    
  def get_doc_theta( self, Data, LP):
    DocIDs = Data['DocIDs']
    alpha_perDoc = self.alpha0 + LP['N_perDoc']
    LP['alpha_perDoc'] = alpha_perDoc
    LP['Elogw_perDoc'] = digamma( alpha_perDoc ) \
                             - digamma( alpha_perDoc.sum(axis=1) )[:,np.newaxis]
    # Added this line to aid human inspection. self.Elogw is never used except to print status
    self.Elogw = LP['Elogw_perDoc']
    return LP
    
  def get_word_phi( self, Data, LP):
    DocIDs = Data['DocIDs']
    lpr = LP['E_log_soft_ev'].copy() # so we can do += later
    for gg in xrange( len(DocIDs) ):
      lpr[ DocIDs[gg][0]:DocIDs[gg][1] ] += LP['Elogw_perDoc'][gg]
    lprPerItem = logsumexp( lpr, axis=1 )
    resp   = np.exp( lpr-lprPerItem[:,np.newaxis] )
    resp   /= resp.sum( axis=1)[:,np.newaxis] # row normalize
    assert np.allclose( resp.sum(axis=1), 1)
    if 'wordIDs_perDoc' in Data:
      for gg in xrange(len(DocIDs)):
        resp[ DocIDs[gg][0]:DocIDs[gg][1] ] *= Data['wordCounts_perDoc'][gg][:,np.newaxis]
    LP['resp'] = resp
    return LP


  ##############################################################    
  ############################################################## Global param updates   
  ##############################################################
  def update_global_params( self, SS, rho=None, **kwargs ):
    '''Admixtures have no global allocation params! 
         Mixture weights are Doc/document specific.
    '''
    pass
    
  ##############################################################    
  ############################################################## Evidence calc.   
  ##############################################################
  def calc_evidence( self, Data, SS, LP ):
    DocIDs = Data['DocIDs']
    if 'wordCounts_perDoc' in Data:
      respNorm = LP['resp'] / LP['resp'].sum(axis=1)[:,np.newaxis]
    else:
      respNorm = None
    if 'ampG' in SS:
      evW = SS['ampG']*self.E_logpW( LP) - SS['ampG']*self.E_logqW(LP)
    else:
      evW = self.E_logpW( LP) - self.E_logqW(LP)
    if 'ampG' in SS:
      evZ = SS['ampG']*self.E_logpZ( DocIDs, LP ) - SS['ampF']*self.E_logqZ( DocIDs, LP, respNorm )
    else:
      evZ = self.E_logpZ( DocIDs, LP ) - self.E_logqZ( DocIDs, LP, respNorm )
    return evZ + evW

  def E_logpZ( self, DocIDs, LP ):
    ElogpZ = 0
    for gg in xrange( len(DocIDs) ):
      ElogpZ += np.sum( LP['resp'][ DocIDs[gg][0]:DocIDs[gg][1] ] * LP['Elogw_perDoc'][gg] )
    return ElogpZ
    
  def E_logqZ( self, DocIDs, LP, respNorm=None ):
    if respNorm is None:
      ElogqZ = np.sum( LP['resp'] * np.log(EPS+LP['resp'] ) )
    else:
      ElogqZ = np.sum( LP['resp'] * np.log(EPS+respNorm ) )
    return ElogqZ    

  def E_logpW( self, LP ):
    nDoc = len(LP['alpha_perDoc'])
    ElogpW = gammaln(self.K*self.alpha0)-self.K*gammaln(self.alpha0)    
    ElogpW *= nDoc  # same prior over each Doc of data!
    for gg in xrange( nDoc ):
      ElogpW += (self.alpha0-1)*LP['Elogw_perDoc'][gg].sum()
    return ElogpW
 
  def E_logqW( self, LP ):
    ElogqW = 0
    for gg in xrange( len(LP['alpha_perDoc']) ):
      a_gg = LP['alpha_perDoc'][gg]
      ElogqW +=  gammaln(  a_gg.sum()) - gammaln(  a_gg ).sum() \
                  + np.inner(  a_gg -1,  LP['Elogw_perDoc'][gg] )
    return ElogqW

  ##############################################################    
  ############################################################## Sampling   
  ##############################################################
  def sample_from_pred_posterior( self ):
    pass
