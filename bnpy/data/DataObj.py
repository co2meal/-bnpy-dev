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
    
  def get_short_name(self):
    ''' Returns string with short name (at most 10 char) of this data object,
          with no spaces and only alpha-numeric characters.
        Useful for creating filepaths specific for this data object.
    '''
    return "UnknownCustomData"

  def get_text_summary(self, **kwargs):
    ''' Returns string with human-readable description of this dataset 
        e.g. source, author/creator, etc.
    '''
    s = 'DataType: %s. Size: %d' % (self.__class__.__name__, self.get_size())
    return s

  def get_size(self, **kwargs):
    ''' Returns int specifying the 'size' of this object.

        Usually, this means the number of data units.
    '''
    pass

  def select_subset_by_mask(self, *args, **kwargs):
    ''' Returns DataObj of the same type, containing a subset of self's data
    '''
    pass
    
  def add_data(self, DataObj):
    ''' Updates (in-place) the dataset to include provided data
    '''
    pass