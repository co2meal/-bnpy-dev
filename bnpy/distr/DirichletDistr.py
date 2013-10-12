'''
DirichletDistr.py

Dirichlet distribution in D-dimensions
    
Attributes
-------
lamvec  :  Dx1 vector of non-negative numbers
                lamvec[d] >= 0 for all d
  
'''
import numpy as np
import scipy.linalg
from scipy.special import digamma, gammaln

class DirichletDistr(object):

  @classmethod
  def InitFromData( cls, argDict, Data):
    raise NotImplementedError("TODO after Data format for topic models")
      
  def __init__(self, lamvec=None):
    self.lamvec = lamvec
    if lamvec is not None:
      self.D = lamvec.size
      self.set_helpers()

  def set_helpers(self):
    self.lamsum = self.lamvec.sum()
    self.digammalamvec = digamma(self.lamvec)
    self.digammalamsum = digamma(self.lamsum)
    self.Elogphi   = self.digammalamvec - self.digammalamsum

  ############################################################## Param updates  
  ##############################################################
  def get_post_distr( self, SS ):
    ''' Create new Distr object with posterior params
    '''
    return DirichletDistr(SS.Nvec + self.lamvec)
    
  def post_update( self, SS ):
    ''' Posterior update of internal params given data
    '''
    self.lamvec += SS.Nvec
    self.set_helpers()

  def post_update_soVB( self, rho, starD):
    ''' Stochastic online update of internal params
    '''
    self.lamvec = rho*starD.lamvec + (1.0-rho)*self.lamvec
    self.set_helpers()


  ############################################################## Norm Constants  
  ##############################################################
  @classmethod
  def calc_log_norm_const(cls, lamvec):
    return gammaln(lamvec) - np.sum(gammaln(lamvec))
  
  def get_log_norm_const(self):
    ''' Returns log( Z ), where
         PDF(x) :=  1/Z(theta) f( x | theta )
    '''
    return gammaln( self.lamsum ) - np.sum(gammaln(self.lamvec ))
  
  def get_log_norm_const_from_stats(self):
    ''' Returns log( Znew ), where
            Znew = log norm const of post distr given suff stats
    '''
    pass
    
  def get_entropy( self ):
    ''' Returns entropy of this distribution 
          H[ p(x) ] = -1*\int p(x|theta) log p(x|theta) dx
    '''
    H = -1*self.get_log_norm_const()
    H -= np.sum((self.lamvec-1)*self.Elogphi)
    return H
    
  ############################################################## Conditional Probs.  
  ##############################################################    
  def E_log_pdf(self, Data):
    ''' Returns E[ log p( x | theta ) ] under q(theta) <- this distr
    '''
    raise NotImplementedError("TODO")    
     
  ############################################################## Samplers
  ##############################################################
  def sample(self,numSamples=1):
    ''' Returns samples from self
    '''
    return np.random.dirichlet(self.lamvec, size=numSamples)   

  ############################################################## I/O  
  ##############################################################
  def to_dict(self):
    return dict(lamvec=self.lamvec)
    
  def from_dict(self, PDict):
    self.lamvec = PDict['lamvec']
    self.set_helpers()

    

  def E_log_pdf( self, Data ):
    '''
    '''
    try:
      Data['wordIDs_perGroup'][0]
      return self.log_pdf_from_list( Data )
    except KeyError:
      return np.dot( Data['X'], self.Elogphi )

  def log_pdf_from_list( self, Data ):
    lpr = np.zeros( Data['nObs'] )
    GroupIDs = Data['GroupIDs']
    for docID in xrange( Data['nGroup'] ):
      lpr[ GroupIDs[docID][0]:GroupIDs[docID][1] ] = self.Elogphi[:, Data['wordIDs_perGroup'][docID] ].T
    return lpr
