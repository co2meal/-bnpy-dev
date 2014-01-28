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
* Document : a collection of words, observed together from the same source
        For example: 
            "apple, berry, berry, pear, pear, pear, walnut"

* nDoc : number of documents in the current, in-memory dataset
* nDocTotal : total number of docs, in entire dataset (for online applications)
'''
from .AdmixMinibatchIterator import AdmixMinibatchIterator
from .DataObj import DataObj
from collections import defaultdict
import numpy as np
import scipy.sparse
import sqlite3
from ..util import RandUtil

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
        doc_range = np.zeros((nDoc,2), dtype=np.uint32)
        ii = 0
        for d in xrange( len(doc_data) ):
            # make sure we subtract 1 for word_ids since python indexes by 0
            temp_word_id = [(int(n)-1) for n in doc_data[d][1].split()]
            word_id.extend(temp_word_id)
            word_count.extend([int(n) for n in doc_data[d][2].split()])
            nUniqueWords = len(temp_word_id)
            doc_range[d,:] = [ii, ii + nUniqueWords]
            ii += nUniqueWords
        
        nObs = len(word_id)
        myDict = dict(word_id = word_id, word_count=word_count, 
                      doc_range=doc_range, nDocTotal=nDocTotal, 
                      vocab_size=vocab_size)
        conn.close()
        return cls(**myDict)

    @classmethod
    def read_from_mat(cls, matfilepath, **kwargs):
        ''' Creates an instance of WordsData from Matlab matfile
        '''
        import scipy.io
        InDict = scipy.io.loadmat(matfilepath, **kwargs)
        if 'word_id' not in InDict:
            raise KeyError('Stored matfile needs to have field "word_id"')
        return cls(**InDict)

    def __init__(self, word_id=None, word_count=None, doc_range=None,
                 vocab_size=0, vocab_dict=None, true_t=None,
                 true_tw=None, true_td=None, true_K=None, true_resp=None,
                 nDoc=None, nDocTotal=None, **kwargs):
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
        self.word_count = np.asarray(np.squeeze(word_count), dtype=np.float64)
        self.doc_range = np.asarray(doc_range, dtype=np.uint32)
        self.vocab_size = int(vocab_size)
        
        self.set_corpus_size_params(nDocTotal)
        self.verify_dimensions()
        
        # Save "true" parameters from toy-data
        if true_tw is not None:
            self.true_tw = true_tw
            self.true_td = true_td
            self.true_K = true_K
        if true_resp is not None:
            self.true_resp = true_resp
        if true_t is not None:
            self.true_t = true_t
        # Add dictionary of vocab words, if provided
        if vocab_dict is not None:
            self.vocab_dict = vocab_dict

    def to_minibatch_iterator(self, **kwargs):
      return AdmixMinibatchIterator(self, **kwargs)

    def getDocIDs(self, wordLocs=None):
        ''' Retrieve the document ids corresponding to given word locations.
        '''
        if wordLocs is None:
          if hasattr(self, "__docid__"):
            return self.__docid__
          self.__docid__ = np.zeros(self.word_id.size, dtype=np.uint32)
          for dd in range(self.nDoc):
            self.__docid__[self.doc_range[dd,0]:self.doc_range[dd,1]] = dd
          return self.__docid__
        docIDs = np.zeros(len(wordLocs))
        for dd in range(self.nDoc):
          if dd == 0:
            matchMask = wordLocs < self.doc_range[dd,1] 
          else:
            matchMask = np.logical_and(wordLocs < self.doc_range[dd,1],
                                       wordLocs >= self.doc_range[dd-1,1])
          docIDs[matchMask] = dd
        return docIDs                    

    def to_sparse_matrix(self):
        ''' Create sparse matrix that represents this dataset
            of size V x nDistinctWords
            where only one entry in each column is non-zero
        '''
        if hasattr(self, "__sparseMat__"):
            return self.__sparseMat__
        nDW = self.word_id.size
        infoTuple = (self.word_count, np.int64(self.word_id), np.arange(nDW+1))
        shape = (self.vocab_size,nDW)
        self.__sparseMat__ = scipy.sparse.csc_matrix(infoTuple, shape=shape)
        return self.__sparseMat__
    
    def to_sparse_docword_matrix(self):
        ''' Create sparse matrix that represents a document
            by vocabulary word matrix. Used for 
            efficient initialization of global parameters
        '''
        if hasattr(self, "__sparseDocWordMat__"):
            return self.__sparseDocWordMat__
        row_ind = list()
        col_ind = list()
        doc_range = self.doc_range
        word_count = self.word_count
        for d in xrange(self.nDoc):
            numDistinct = doc_range[d,1] - doc_range[d,0]
            doc_ind_temp = [d]*numDistinct
            row_ind.extend(doc_ind_temp)
            col_ind.extend(self.word_id[ (doc_range[d,0]):(doc_range[d,1]) ])
        self.__sparseDocWordMat__ = scipy.sparse.csr_matrix(
                  (word_count, (row_ind,col_ind)),
                  shape=(self.nDoc, self.vocab_size), dtype=np.float64)
        return self.__sparseDocWordMat__

    def verify_dimensions(self):
        ''' Basic runtime checks to make sure dimensions are set correctly
        '''
        # Make sure both are 1D vectors.  
        # 2D vectors with shape (nDistinctWords,1) will screw up indexing!
        assert self.word_id.ndim == 1
        assert self.word_count.ndim == 1
        assert self.word_id.max() < self.vocab_size
        assert self.nDoc == self.doc_range.shape[0]
        assert self.nObs == len(self.word_id)
        assert self.doc_range.shape[1] == 2
        assert np.all( self.doc_range[:-1,1] == self.doc_range[1:,0])

    def set_corpus_size_params(self, nDocTotal=None):
        ''' Sets dependent parameters 
        '''
        self.nDoc = self.doc_range.shape[0]
        self.nObs = len(self.word_id)

        if nDocTotal is None:
          self.nDocTotal = self.nDoc
        else:
          self.nDocTotal = nDocTotal
                
    def add_data(self, WData):
        ''' Append provided WordsData to the end of this dataset
        '''
        assert self.vocab_size == WData.vocab_size
        self.word_id = np.hstack([self.word_id, WData.word_id])
        self.word_count = np.hstack([self.word_count, WData.word_count])
        sLoc = self.doc_range[-1,1]
        self.doc_range = np.vstack([self.doc_range, sLoc + WData.doc_range])
        self.nDoc += WData.nDoc
        self.nObs += WData.nObs
        self.nDocTotal += WData.nDocTotal
        self.verify_dimensions()

    def get_text_summary(self, doCommon=True):
      if hasattr(self, 'summary') and doCommon:
        s = self.summary
      elif doCommon:
        s = " nDoc %d, vocab_size %d\n" % (self.nDoc, self.vocab_size)
      else:
        s = ''
      return s + self.get_doc_stats_summary()

    def get_doc_stats_summary(self, pRange=[0,5, 50, 95, 100]):
        ''' Get string summarizing size of this data object
        '''
        nDistinctWordsPerDoc = np.zeros(self.nDoc)
        nTotalWordsPerDoc = np.zeros(self.nDoc)
        for d in range(self.nDoc):
          drange = self.doc_range[d,:]
          nDistinctWordsPerDoc[d] = drange[1] - drange[0]
          nTotalWordsPerDoc[d] = self.word_count[drange[0]:drange[1]].sum()
        assert np.sum(nDistinctWordsPerDoc) == self.word_id.size
        assert np.sum(nTotalWordsPerDoc) == np.sum(self.word_count)
        s = ''
        for p in pRange:
          if p == 0:
            sp = 'min'
          elif p == 100:
            sp = 'max'
          else:
            sp = "%d%%" % (p)
          s += "%5s " % (sp)
        s += '\n'
        for p in pRange:
          s += "%5s " % ("%.0f" % (np.percentile(nDistinctWordsPerDoc, p)))    
        s += ' nDistinctWordsPerDoc\n'
        for p in pRange:
          s += "%5s " % ("%.0f" % (np.percentile(nTotalWordsPerDoc, p)))    
        s += ' nTotalWordsPerDoc'
        return s

    def select_subset_by_mask(self, docMask=None, wordMask=None,
                                    doTrackFullSize=True):
        '''
            Selects subset of this dataset defined by mask arguments,
             and returns a WordsData object representing that subset

            Args
            -------
            docMask : None, or list of document ids to select
            wordMask : None, or list of words to select
                       each entry is an index into self.word_id

            doTrackFullSize : boolean indicator for whether resulting dataset
                       should retain the total size of this set,
                    or should be entirely self contained (nDoc=nDocTotal) 

            Returns
            --------
            WordsData object, where
                nDoc = number of documents in the subset (=len(mask))
                nObs = nDistinctWords in the subset of docs
                nDocTotal defines size of entire dataset (not subset)
        '''
        if docMask is None and wordMask is None:
          raise ValueError("Must provide either docMask or wordMask")

        if docMask is not None:
          nDoc = len(docMask)
          nObs = np.sum(self.doc_range[docMask,1] - self.doc_range[docMask,0])
          word_id = np.zeros(nObs)
          word_count = np.zeros(nObs)
          doc_range = np.zeros((nDoc,2))
        
          # Fill in new word_id, word_count, and doc_range
          startLoc = 0
          for d in xrange(nDoc):
            start,stop = self.doc_range[docMask[d],:]
            endLoc = startLoc + (stop - start)
            word_count[startLoc:endLoc] = self.word_count[start:stop]
            word_id[startLoc:endLoc] = self.word_id[start:stop]
            doc_range[d,:] = [startLoc,endLoc]
            startLoc += (stop - start)

        elif wordMask is not None:
          wordMask = np.sort(wordMask)
          nObs = len(wordMask)
          docIDs = self.getDocIDs(wordMask)
          uDocIDs = np.unique(docIDs)
          nDoc = uDocIDs.size
          doc_range = np.zeros((nDoc,2))

          # Fill in new word_id, word_count, and doc_range
          word_id =  self.word_id[wordMask]
          word_count = self.word_count[wordMask]
          startLoc = 0
          for dd in range(nDoc):
            nWordsInCurDoc = np.sum(uDocIDs[dd] == docIDs)
            doc_range[dd,:] = startLoc, startLoc + nWordsInCurDoc
            startLoc += nWordsInCurDoc           

        # Pack up all relevant params for creating new dataset    
        myDict = dict()
        myDict["doc_range"] = doc_range
        myDict["word_id"] = word_id
        myDict["word_count"] = word_count
        myDict["vocab_size"] = self.vocab_size
        if doTrackFullSize:
          myDict["nDocTotal"] = self.nDocTotal
        return WordsData(**myDict)

    @classmethod
    def genToyData(cls, seed=101, nDocTotal=None, nWordsPerDoc=None, 
                      docTopicParamVec=None, TopicWordProbs=None,
                      **kwargs):
        ''' Generates toy dataset using defined global structure.
        '''
        PRNG = np.random.RandomState(seed)
        TopicWordProbs /= TopicWordProbs.sum(axis=1)[:,np.newaxis]

        K = TopicWordProbs.shape[0]
        V = TopicWordProbs.shape[1]

        # true document x topic proportions
        true_td = np.zeros((K,nDocTotal)) 
    
        doc_range = np.zeros((nDocTotal, 2))
        wordIDsPerDoc = list()
        wordCountsPerDoc = list()
        respPerDoc = list()

        # counter for tracking the start index for current document 
        #  within the corpus-wide word lists
        startPos = 0
        for d in xrange(nDocTotal):
            true_td[:,d] = PRNG.dirichlet(docTopicParamVec) 
            Npercomp = RandUtil.multinomial(nWordsPerDoc, true_td[:,d], PRNG)

            # wordCountBins: V x 1 vector
            #   entry v counts # times vocab word v appears in current doc
            wordCountBins = np.zeros(V)
            for k in xrange(K):
                wordCountBins += RandUtil.multinomial(Npercomp[k], 
                                            TopicWordProbs[k,:], PRNG)

            wIDs = np.flatnonzero(wordCountBins > 0)
            wCounts = wordCountBins[wIDs]
            assert np.allclose( wCounts.sum(), nWordsPerDoc)
        
            curResp = (TopicWordProbs[:, wIDs] * true_td[:,d][:,np.newaxis]).T
          
            wordIDsPerDoc.append(wIDs)
            wordCountsPerDoc.append(wCounts)
            respPerDoc.append(curResp)

            #start and stop ids for documents
            doc_range[d,0] = startPos
            doc_range[d,1] = startPos + wIDs.size  
            startPos += wIDs.size
    
        word_id = np.hstack(wordIDsPerDoc)
        word_count = np.hstack(wordCountsPerDoc)
        assert respPerDoc[0].shape[1] == K
        assert respPerDoc[1].shape[1] == K

        true_resp = np.vstack(respPerDoc)
        true_resp /= true_resp.sum(axis=1)[:,np.newaxis]

        #Insert all important stuff in myDict
        myDict = dict(true_K=K, true_t=docTopicParamVec,
                  true_tw=TopicWordProbs, true_td=true_td,
                  true_resp=true_resp,
                  )
        myDict["doc_range"] = doc_range
        myDict["word_id"] = word_id
        myDict["word_count"] = word_count
        myDict["vocab_size"] = V
        return WordsData(**myDict)
