'''
DataObj.py

General abstract base class for all data objects,
whether they are full datasets or iterators over small batches of data

Attributes
-------
nObs : 
'''
from DataIterator import DataIterator

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
    

  ######################################################### To DataIterator
  ######################################################### 
  def to_iterator(self, **kwargs):
    ''' Create an iterator for streamed processing of subsets of this data object.

        Args
        ------
        nBatch
        nLap
    
        Returns
        ------
        I : DataIterator object.
    '''
    return DataIterator(self, **kwargs)

  ######################################################### Human I/O
  ######################################################### 
  def get_short_name(self):
    ''' Returns string with human-readable name viable for system file paths.

        Useful for creating filepaths specific for this data object.
    '''
    if hasattr(self, 'name'):
      return self.name
    return "UnknownData"

  def get_text_summary(self, **kwargs):
    ''' Returns string with human-readable description of this dataset 
        e.g. source, author/creator, etc.
    '''
    s = 'DataType: %s. Size: %d' % (self.__class__.__name__, self.get_size())
    return s

  ######################################################### Accessors
  ######################################################### 
  def get_size(self, **kwargs):
    ''' Returns int count of active, in-memory data units for this Data object.
    '''
    pass

  def get_total_size(self, **kwargs):
    ''' Returns int count of all data units associated with current dataset.
    '''
    pass

  ######################################################### Create Data from Subset
  ######################################################### 
  def select_subset_by_mask(self, *args, **kwargs):
    ''' Returns DataObj of the same type, containing a subset of self's data
    '''
    pass
    

  ######################################################### Append Data
  ######################################################### 
  def add_data(self, DataObj):
    ''' Updates (in-place) this dataset to append all data in provided Data object
    '''
    pass
