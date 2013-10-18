'''
WordsData.py

Data object that represents word counts across a collection of documents.

Terminology
-------

'''
from .DataObj import DataObj
from collections import defaultdict
import numpy as np

class WordsData(DataObj):
  
    @classmethod
    def read_from_mat(cls, matfilepath, nObsTotal=None, **kwargs):
        ''' (not yet implemented) Static Constructor for building an instance of Words Data from disk 
        '''
        import scipy.io
        InDict = scipy.io.loadmat( matfilepath, **kwargs)
        if 'word_id' not in InDict:
            raise KeyError('Stored matfile needs to have data in field named word_id and doc_range')
        return cls( InDict['word_id'], InDict['doc_range'], nObsTotal )
  
    def __init__(self, word_id, word_count, doc_range, nDocTotal, nDoc, nObsTotal, nObs, vocab_size, true_tw=None, true_td=None, true_K=None):
        
        # Variable definitions can be found here
        self.word_id = word_id # list of distinct word ids within a document across the entire document corpus
        self.word_count = word_count # each row defines the count of that word in a given document associated with word_id
        self.doc_range = doc_range # no. of documents x 2 (col1 = start row of word_id, col2 = stop row of word_id)
        self.nDocTotal = nDocTotal # no. of total documents in the entire corpus
        self.nObsTotal = nObsTotal # no. of unique word_tokens in corpus across all documents
        self.nDoc = nDoc # if using a minibatch, specifies the number of documents for that minibatch
        self.nObs = nObs # if using a minibatch, specifies the distinct number of unique vocabulary words within a document for that minibatch
        self.vocab_size = vocab_size # no. of unique vocabulary words across the entire document corpus (not document specific)
        
        if true_tw is not None: # if generated from toy data, save to Data object
            self.true_tw = true_tw
            self.true_td = true_td
            self.true_K = true_K

        
    def select_subset_by_mask(self, mask):
        # Selects a subset of documents defined by mask and recreates the relevant fields
        # In particular, note that nDoc and nObs now refers to the total subset of relevant documents and distinct tokens respectively
        
        # Since we are masking by documents, we need to figure out the size of our new minibatch
        # with respect to the number of unique word tokens
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
