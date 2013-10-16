'''
XData.py

Data object for holding a dense matrix X of real numbers,
where each row of X is a single observation.

This object guarantees underlying numpy array representation is best for math ops.
This means byteorder and C-contiguity are standardized.
'''
from .DataObj import DataObj

class WordsData(DataObj):
  
    @classmethod
    def read_from_mat(cls, matfilepath, nObsTotal=None, **kwargs):
        ''' Static Constructor for building an instance of XData from disk
        '''
        import scipy.io
        InDict = scipy.io.loadmat( matfilepath, **kwargs)
        if 'doc_word_count' not in InDict:
            raise KeyError('Stored matfile needs to have data in field named WC and DW')
        return cls( InDict['doc_word_count'], InDict['doc_word_id'], nObsTotal )
  
    def __init__(self, word_dict):
        ''' Constructor for building an instance of XData given an array
        Ensures array is 2-dimensional with proper byteorder, contiguity, and ownership
        '''
        self.WC = word_dict["WC"]
        self.DOC_ID = word_dict["DOC_ID"]
        self.nObsTotal = word_dict["nObs"]
        self.nWords = word_dict["nWords"]
        self.nDocs = word_dict["nDocs"]
        if 'true_tw' in word_dict:
            self.true_tw = word_dict["true_tw"]
            self.true_td = word_dict["true_td"]
            self.true_K = word_dict["true_K"]
        #self.set_dependent_params(nObsTotal=nObsTotal)
        #self.check_dims()
'''    
  #########################################################  internal methods
  ######################################################### 
    
  def set_dependent_params( self, nObsTotal=None): 
    
    self.D = len(self.word_id)
    self.nObs = self.X.shape[0]
    self.dim = self.X.shape[1]
    if nObsTotal is None:
      self.nObsTotal = self.nObs
    else:
      self.nObsTotal = nObsTotal
    
  def check_dims( self ):
    assert self.X.ndim == 2
    assert self.X.flags.c_contiguous
    assert self.X.flags.owndata
    assert self.X.flags.aligned
    assert self.X.flags.writeable
    
  #########################################################  DataObj operations
  ######################################################### 
  def select_subset_by_mask(self, mask):
    #Creates new XData object by selecting certain rows (observations)
    #Ensures the nObsTotal attribute is the same.
    return WordsData(self.X[mask], nObsTotal=self.nObsTotal)

  #########################################################  I/O methods
  ######################################################### 
  def __str__(self):
    np.set_printoptions(precision=5)
    return self.X.__str__()
    
  def summarize_num_observations(self):
    return '  num obs: %d' % (self.nObsTotal)
'''    