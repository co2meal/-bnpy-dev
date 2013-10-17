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

class WordsData(DataObj):
  
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
    
  #########################################################  DataObj operations
  ######################################################### 
    def select_subset_by_mask(self, mask):
      raise NotImplementedError("TODO")

  #########################################################  I/O methods
  ######################################################### 
    def __str__(self):
      return self.get_text_summary()
    
    def summarize_num_observations(self):
      ''' Summarize key facts like
          * vocabulary size
          * avg document length (distinct words)
          * avg document length (total words)
      '''
      return '  num obs: %d' % (self.nObsTotal)