'''
WordsData.py

Data object that represents word counts across a collection of documents.

Terminology
-------
* Vocab : The finite collection of possible words.  
          {apple, berry, cardamom, fruit, pear, walnut}
        We assume this set has a fixed ordering, so each word is associated 
        with a particular integer in the set 0, 1, ... VocabSize-1
           0: apple        3: fruit
           1: berry        4: pear
           2: cardamom     5: walnut

* VocabWordID : an integer that indexes into the Vocab collection
        for example, the word "fruit" above has a VocabWordID of 3

* Document : a collection of words, observed together from the same source
        For example: 
            "apple, berry, berry, pear, pear, pear, walnut"

* nDoc : number of documents in the whole dataset

* nTotalWords : number of total words in a document
        The example document has 6 tokens, so nTotalTokens=6.
     
* DistinctWords : set of vocab words that appear at least ONCE in a document
        The example document has *distinct* words
            "apple, berry, pear, walnut"

* nDistinctWords : number of distinct tokens in a document
        The example document has nDistinctWords=4

* wordCount : counts *how many* times each distinct word appears in a document
        The example document has the following counts
             apple: 1, berry: 2, pear: 3, walnut: 1

Data Structure / Representation
-------
Each document may be represented with two vectors:
  wordID    : nDistinctWords-length vector (integers 0, 1, ... VocabSize)
                  entry n gives vocab word ID for the n-th distinct word
  wordCount : nDistinctWords-length vector (integers 0, 1, ... )
                  entry n gives the count for the n-th distinct word

  For the example document, we have the following encoding
      wordID    = [0 1 4 5]
      wordCount = [1 2 3 1]

The entire document collection is represented by concatenating these building-blocks 

  wordID    = [wordID(Doc 1)    wordID(Doc 2) ... wordID(Doc #nDoc)]
  wordCount = [wordCount(Doc 1) wordCount(Doc 2) ... wordCount(Doc #nDoc)]
'''
from .DataObj import DataObj
from collections import defaultdict
import numpy as np

class WordsData(DataObj):

    @classmethod
    def read_from_mat(cls, matfilepath, nObsTotal=None, **kwargs):
        ''' Creates an instance of WordsData from Matlab matfile
        '''
        import scipy.io
        InDict = scipy.io.loadmat(matfilepath, **kwargs)
        if 'word_id' not in InDict:
            raise KeyError('Stored matfile needs to have field "word_id"')
        return cls(**InDict)

    def __init__(self, word_id=None, word_count=None, doc_range=None,
                 vocab_size=0, vocab_dict=None, 
                 true_tw=None, true_td=None, true_K=None, **kwargs):
        ''' Constructor for WordsData

            Args
            -------
            word_id : nDistinctWords-length vector 
                      entry i gives VocabWordID for distinct word i in corpus
            word_count : nDistinctWords-length vector
                      entry i gives count for word_id[i] in that document
            doc_range : nDoc x 2 matrix
                      doc_range[d,:] gives (start,stop) for document d
                      where start/stop index rows in word_id,word_count
            vocab_size : integer size of set of possible vocabulary words
        '''
        self.word_id = np.asarray(np.squeeze(word_id), dtype=np.uint32)
        self.word_count = np.asarray(np.squeeze(word_count), dtype=np.float32)
        self.doc_range = doc_range
        self.vocab_size = int(vocab_size)
        self.set_dependent_params()
        self.verify_dimensions()

        if true_tw is not None: # if generated from toy data, save to Data object
            self.true_tw = true_tw
            self.true_td = true_td
            self.true_K = true_K
        
        # check if data contains a dictionary of vocabulary words
        if vocab_dict is not None:
            self.vocab_dict = vocab_dict
        else:
            print "Warning: Data doesn't contain the vocabulary dictionary, a qualitative assessment will be difficult"


    def verify_dimensions(self):
        ''' Basic runtime checks to make sure things look good
        '''
        # Make sure both are 1D vectors.  
        # 2D vectors with shape (nDistinctWords,1) will screw up indexing!
        assert self.word_id.ndim == 1
        assert self.word_count.ndim == 1

    def set_dependent_params( self, nObsTotal=None):
        ''' Sets dependent parameters so that we don't have to store too many stuff
        '''
        self.nDocTotal,_ = self.doc_range.shape
        self.nDoc = self.nDocTotal
        self.nObsTotal = len(self.word_id)
        self.nObs = self.nObsTotal
        
    def select_subset_by_mask(self, mask):
        '''
            Selects subset of documents defined by mask
             and returns a WordsData object representing that subset

            Returns
            --------
            WordsData object, where
                nDoc = number of documents in the subset (=len(mask))
                nObs = nDistinctWords in the subset of docs
                nObsTotal, nDocTotal define size of entire dataset (not subset)
        '''
        subset_size = ( (self.doc_range[mask,1]) - self.doc_range[mask,0] ).sum()
        new_word_id = np.zeros( subset_size )
        new_word_count = np.zeros( subset_size )
        new_doc_range = np.zeros( (len(mask),2) )
        ii = 0
        
        # Create new word_id, word_count, and doc_range
        for d in xrange(len(mask)):
            start,stop = self.doc_range[mask[d],:]
            num_distinct = stop-start
            new_stop = ii + num_distinct
            new_word_count[ii:new_stop] = self.word_count[start:stop]
            new_word_id[ii:new_stop] = self.word_id[start:stop]
            new_doc_range[d,:] = [ii,new_stop]
            ii += num_distinct
            
        myDict = defaultdict()
        myDict["doc_range"] = new_doc_range
        myDict["word_id"] = new_word_id
        myDict["word_count"] = new_word_count
        myDict["nDocTotal"] = self.nDocTotal
        myDict["nDoc"] = len(mask)
        myDict["nObsTotal"] = self.nObsTotal
        myDict["nObs"] = int(subset_size)
        myDict["vocab_size"] = self.vocab_size
        
        if hasattr(self,'true_tw'):
            myDict["true_tw"] = self.true_tw
            myDict["true_td"] = self.true_td
            myDict["true_K"] = self.true_K
        
        return WordsData(**myDict)