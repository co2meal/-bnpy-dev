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
import scipy.sparse
import sqlite3

class WordsData(DataObj):

    @classmethod
    def makeRandomData(cls, nDoc=10, nWordsPerDoc=10, vocab_size=12, **kwargs):
        ''' Creates an instance of WordsData from Matlab matfile
        '''
        PRNG = np.random.RandomState(0)
        word_id = list()
        word_count = list()
        doc_range = np.zeros((nDoc, 2))
        for dd in range(nDoc):
            wID = PRNG.choice(vocab_size, size=nWordsPerDoc, replace=False)
            wCount = PRNG.choice(np.arange(1,5),size=nWordsPerDoc, replace=True)
            word_id.extend(wID)
            word_count.extend(wCount)
            start = nWordsPerDoc * dd
            doc_range[dd,:] = [start, start + nWordsPerDoc]
        myDict = dict(word_id=word_id, word_count=word_count)
        myDict['doc_range'] = doc_range
        myDict['vocab_size'] = vocab_size
        return cls(**myDict)

    @classmethod
    def read_from_db(cls, dbpath, sqlquery, nDoc=None, nDocTotal=None, vocab_size=None, **kwargs):
        ''' Creates an instance of WordsData from the database
        '''
        # Connect to sqlite database and retrieve results as string
        conn = sqlite3.connect(dbpath)
        conn.text_factory = str
        result = conn.execute(sqlquery)
        doc_data = result.fetchall()
        
        word_id = list()
        word_count = list()
        doc_range = np.zeros( (nDoc,2) )
        ii = 0
        for d in xrange( len(doc_data) ):
            doc_range[d,0] = ii
            # make sure we subtract 1 for word_ids since python indexes by 0
            temp_word_id = [(int(n)-1) for n in doc_data[d][1].split()]
            word_id.extend(temp_word_id)
            word_count.extend([int(n) for n in doc_data[d][2].split()])
            nUniqueWords = len(temp_word_id)
            doc_range[d,1] = ii + nUniqueWords
            ii += nUniqueWords + 1
        
        nObs = len(word_id)
        nObsTotal = nObs
        myDict = dict(word_id = word_id, word_count=word_count, doc_range=doc_range, 
                      nDoc=nDoc, nDocTotal=nDocTotal, nObs=nObs, nObsTotal = nObsTotal,
                      vocab_size=vocab_size, db_pull=True)
        
        return cls(**myDict)

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
                 true_tw=None, true_td=None, true_K=None, 
                 nDoc=None, nDocTotal=None, nObs=None,
                 nObsTotal=None, db_pull=False, **kwargs):
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
        
        # If we're not pulling from the database
        if db_pull is False:
            self.set_dependent_params()
            self.verify_dimensions()
        else: # Set to full database corpus
            self.nDocTotal = nDocTotal
            self.nObsTotal = nObsTotal
            self.nDoc = nDoc
            self.nObs = nObs

        if true_tw is not None: # if generated from toy data, save to Data object
            self.true_tw = true_tw
            self.true_td = true_td
            self.true_K = true_K
        
        # check if data contains a dictionary of vocabulary words
        if vocab_dict is not None:
            self.vocab_dict = vocab_dict


    def to_sparse_matrix(self):
        ''' Create sparse matrix that represents this dataset
            of size V x nDistinctWords
            where only one entry in each column is non-zero
        '''
        if hasattr(self, "__sparseMat__"):
            return self.__sparseMat__
        
        nDW = self.word_id.size
        infoTuple = (self.word_count, self.word_id, np.arange(nDW+1))
        shape = (self.vocab_size,nDW)
        self.__sparseMat__ = scipy.sparse.csc_matrix(infoTuple, shape=shape)
        return self.__sparseMat__

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