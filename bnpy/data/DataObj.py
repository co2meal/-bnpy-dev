'''
DataObj.py

General abstract base class for all data objects,
whether they are full datasets or iterators over small batches of data

Attributes
-------
nObs : 
'''

class DataObj(object):
  @classmethod
  def read_from_mat(self, matfilepath):
    ''' Constructor for building data object from disk
    '''
    pass
  
  def __init__(self, *args, **kwargs):
    ''' Constructor for building data object from scratch in memory
    '''
    pass
    
  def get_text_summary(self):
    ''' Returns string with human-readable description of this dataset 
        e.g. source, author/creator, etc.
    '''
    if hasattr(self, 'summary'):
      return self.summary
    return 'Generic %s Dataset' % (self.__class__.__name__)
    
  def summarize_num_observations(self):
    ''' Returns string summary of number of observations in this data object
    '''
    pass
    
  def select_subset_by_mask(self, *args, **kwargs):
    ''' Returns DataObj of the same type, containing a subset of self's data
    '''
    pass
    
