''' Distr.py : 

Generic exponential family probability distribution object
'''

class Distr( object ):

  @classmethod
  def InitFromData( cls, argDict, Data):
    pass
    
  def __init__( self, *args, **kwargs):
    pass

  ##############################################################    
  ############################################################## Param updates  
  ##############################################################
  def get_post_distr( self, SS ):
    ''' Create new Distr object with posterior params
    '''
    pass
    
  def post_update( self, SS ):
    ''' Posterior update of internal params given data
    '''
    pass

  def post_update_soVB( self, rho, *args ):
    ''' Stochastic online update of internal params
    '''
    pass

  ##############################################################    
  ############################################################## Norm Constants  
  ##############################################################
  def get_log_norm_const(self):
    ''' Returns log( Z ), where
         PDF(x) :=  1/Z(theta) f( x | theta )
    '''
    pass
  
  def get_log_norm_const_from_stats(self):
    ''' Returns log( Znew ), where
            Znew = log norm const of post distr given suff stats
    '''
    pass
    
  def get_entropy( self ):
    ''' Returns entropy of this distribution 
          H[ p(x) ] = -1*\int p(x|theta) log p(x|theta) dx
    '''
    pass
    
    
  ##############################################################    
  ############################################################## Conditional Probs.  
  ##############################################################
  def log_pdf( self ):
    ''' Returns log p( x | theta )
    '''
    pass
    
  def E_log_pdf( self ):
    ''' Returns E[ log p( x | theta ) ] under q(theta) <- this distr
    '''
    pass
    
    

  ##############################################################    
  ############################################################## I/O  
  ##############################################################
  def from_string( self, mystr ):
    pass

  def to_string(self):
    pass

  def to_dict(self):
    pass
    
  def from_dict(self):
    pass