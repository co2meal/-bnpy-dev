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
    return dict( alpha0=self.alpha0, K=self.K, inferType=self.inferType )
    
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
    resp = LP['resp']
    total_obs,K = resp.shape
    word_count = Data.word_count
    groupid = Data.groupid
    D = Data.D
    Nvec = np.zeros( (D, K) )
    # Loop through documents
    for d in xrange(D):
        start,stop = groupid[d]
        # get document-level sufficient statistics 
        Nvec[d,:] = np.dot( word_count[d].values(), resp[start:stop,:] ) 
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
      LP['doc_topic_weights']
    except KeyError:
      total_obs, K = LP['E_log_soft_ev'].shape
      LP['doc_topic_weights'] = np.zeros( (Data.D, K) )

    groupid = Data.groupid
    D = len(groupid)
    #DocIDs = Data['DocIDs']
    #nDocs = Data['nDoc']
    prevVec = None
    for rep in xrange( 4 ):
      LP = self.get_doc_theta( Data, LP)
      LP = self.get_word_phi( Data, LP)
      #for gg in range( nDocs ):
        #DocResp = LP['resp'][ DocIDs[gg][0]:DocIDs[gg][1] ]
        #LP['doc_topic_weights'][gg] = np.sum( DocResp, axis=0 )
        
      for d in xrange( D ):
        start,stop = groupid[d]
        #doc_topic_weights = Freq of unique word counts for document d x responsibilities
        LP['doc_topic_weights'][d,:] = np.dot( Data.word_count[d].values(), LP['resp'][start:stop,:] )  
      curVec = LP['theta'].flatten()
      if prevVec is not None and np.allclose( prevVec, curVec ):
        break
      prevVec = curVec
    
    return LP
    
  def get_doc_theta( self, Data, LP):
    
    theta = self.alpha0 + LP['doc_topic_weights']
    LP['theta'] = theta
    LP['ElogTheta'] = digamma( theta ) \
                             - digamma( theta.sum(axis=1) )[:,np.newaxis]
    # Added this line to aid human inspection. self.Elogw is never used except to print status
    self.ElogTheta = LP['ElogTheta']
    return LP
    
  def get_word_phi( self, Data, LP):
    #DocIDs = Data['DocIDs']
    groupid = Data.groupid
    lpr = LP['E_log_soft_ev'].copy() # so we can do += later
    # Loop through documents and add expectations of document i and topics 1:K
    # to all relevant observations
    for d in xrange( Data.D ):
        start,stop = groupid[d]
        lpr[start:stop, :] += LP['ElogTheta'][d,:]
    lprPerItem = logsumexp( lpr, axis=1 )
    resp   = np.exp( lpr-lprPerItem[:,np.newaxis] )
    resp   /= resp.sum( axis=1)[:,np.newaxis] # row normalize
    assert np.allclose( resp.sum(axis=1), 1)
    '''
    ii = 0
    for d in xrange(Data.D):
        start,stop = groupid[d]
        for (word_id, word_freq) in enumerate(word_count[d]):
            resp[ :, ii] *= Data['wordCounts_perDoc'][d][:,np.newaxis]
            ii += 1
    '''
    LP['resp'] = resp
    LP['resp_unorm'] = lpr
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
    #DocIDs = Data['DocIDs']
    # assume for now that DocIDs is an index that contains all documents
    DocIDs = range(Data.D)
    respNorm = None
    '''
    if 'wordCounts_perDoc' in Data:
      respNorm = LP['resp'] / LP['resp'].sum(axis=1)[:,np.newaxis]
    else:
      respNorm = None
    '''
    if 'ampG' in SS:
      evW = SS['ampG']*self.E_logpW( LP) - SS['ampG']*self.E_logqW(LP)
    else:
      pPI = self.E_logpTheta( Data, LP ) # evidence of 
      qPI = self.E_logqTheta( Data, LP ) # entropy of ...
    if 'ampG' in SS:
      evZ = SS['ampG']*self.E_logpZ( DocIDs, LP ) - SS['ampF']*self.E_logqZ( DocIDs, LP, respNorm )
    else:
      #evZ = self.E_logpZ( DocIDs, LP ) - self.E_logqZ( DocIDs, LP, respNorm )
      pz = self.E_logpZ( Data, LP )
      qz = self.E_logqZ( Data, LP )
    lb_alloc = pPI + pz - qPI - qz
    print "pz: " + str(pz)
    print "qz: " + str(qz)
    print "pPI: " + str(pPI)
    print "qPI: " + str(qPI)
    return lb_alloc

  def E_logpZ( self, Data, LP ):
    #for gg in xrange( len(DocIDs) ):
      #ElogpZ += np.sum( LP['resp'][ DocIDs[gg][0]:DocIDs[gg][1] ] * LP['ElogTheta'][gg] )
      
    # p(z | pi)
    ElogpZ = LP["doc_topic_weights"] * LP["ElogTheta"]
    
    return ElogpZ.sum()
    
  def E_logqZ( self, Data, LP):  
    ElogqZ = np.dot(np.log(EPS+LP['resp']), LP["doc_topic_weights"].T)    
    return ElogqZ.sum()    

  def E_logpTheta( self, Data, LP ):
    D,K = LP['theta'].shape
    ElogpW = gammaln(K*self.alpha0)-K*gammaln(self.alpha0)    
    ElogpW *= D  # same prior over each Doc of data!
    for gg in xrange( D ):
      ElogpW += (self.alpha0-1)*LP['ElogTheta'][gg].sum()
    return ElogpW
 
  def E_logqTheta( self, Data, LP ):
    ElogqW = 0
    for gg in xrange( len(LP['theta']) ):
      a_gg = LP['theta'][gg]
      ElogqW +=  gammaln(  a_gg.sum()) - gammaln(  a_gg ).sum() \
                  + np.inner(  a_gg -1,  LP['ElogTheta'][gg] )
    return ElogqW

  ##############################################################    
  ############################################################## Sampling   
  ##############################################################
  def sample_from_pred_posterior( self ):
    pass
