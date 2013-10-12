'''
Distr.py 

Generic exponential family probability distribution object
'''

class Distr( object ):

  @classmethod
  def InitFromData( cls, argDict, Data):
    pass
    
  def __init__( self, *args, **kwargs):
    pass

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
    
    
  ############################################################## E step cond probs.  
  ##############################################################
  def log_pdf( self ):
    ''' Returns log p( x | theta )
    '''
    pass
    
  def E_log_pdf( self ):
    ''' Returns E[ log p( x | theta ) ] under q(theta) <- this distr
    '''
    pass
    
    
  ############################################################## Samplers
  ##############################################################
  def sample(self):
    ''' Returns samples from Distr
    '''
    pass

  
  ############################################################## Exp Fam Accessors  
  ##############################################################
    
    
  ############################################################## I/O  
  ##############################################################
  def to_dict(self):
    pass
    
  def from_dict(self, pDict):
    pass