"""
Classes
-----
WordsData
    Data object for holding a sparse bag-of-word observations,
    organized as a collection of documents, each containing some words.
"""

import numpy as np
import scipy.sparse
import scipy.io
from collections import namedtuple

from bnpy.data.DataObj import DataObj
from bnpy.util import as1D, toCArray
from bnpy.util import numpyToSharedMemArray, sharedMemToNumpyArray


class WordsData(DataObj):

    """  Dataset object for sparse discrete words across many documents.

    Attributes
    ----
    nUniqueToken : int
        Total number of distinct tokens observed.
        Here, distinct means that the pair of wordtype / document id
        has not been seen before.
    word_id : 1D array, size U
        word_id[i] gives the int type for distinct word i.
        Range: [0, 1, 2, 3, ... vocab_size-1]
    word_count : 1D array, size U
        word_count[i] gives the number of times distinct word i
        appears in its corresponding document.
        Range: [1, 2, 3, ...]
    doc_range : 1D array, size nDoc+1
        Defines section of each corpus-wide vector which belongs
        to each document d:
        word_id_d = self.word_id[doc_range[d]:doc_range[d+1]]
        word_ct_d = self.word_count[doc_range[d]:doc_range[d+1]]
    vocab_size : int
        size of set of possible vocabulary words
    vocabList : list of strings
        specifies the word associated with each vocab type id
    nDoc : int
        number of groups (documents) represented in memory
        by this object.
    nDocTotal : int
        total size of the corpus
        will differ from nDoc when this dataset represents a
        small minibatch of some larger corpus.
    TrueParams : None [default], or dict of true parameters.

    """

    @classmethod
    def LoadFromFile_tokenlist(cls, filepath, vocab_size=0, nDocTotal=None,
                               min_word_index=1, **kwargs):
        ''' Constructor for loading from tokenlist-formatted matfile.

        Returns
        -------
        Data : WordsData object
        '''
        doc_sizes = []
        word_id = []
        word_ct = []
        Vars = scipy.io.loadmat(filepath)
        key = 'tokensByDoc'
        if key not in Vars:
            key = 'test'
        nDoc = Vars[key].shape[1]
        for d in xrange(nDoc):
            tokens_d = np.squeeze(Vars[key][0, d])
            word_id_d = np.unique(tokens_d)
            word_ct_d = np.zeros_like(word_id_d, dtype=np.float64)
            for uu, uid in enumerate(word_id_d):
                word_ct_d[uu] = np.sum(tokens_d == uid)
            doc_sizes.append(word_id_d.size)
            word_id.extend(word_id_d - min_word_index)
            word_ct.extend(word_ct_d)
        doc_range = np.hstack([0, np.cumsum(doc_sizes)])
        return cls(word_id=word_id, word_count=word_ct, nDocTotal=nDocTotal,
                   doc_range=doc_range, vocab_size=vocab_size)

    @classmethod
    def LoadFromFile_ldac(
            cls, filepath, vocab_size=0, nDocTotal=None,
            sliceID=None, nSlice=None, filesize=None,
            **kwargs):
        ''' Constructor for loading data from .ldac format files.

        Returns
        -------
        Data : WordsData object
        '''
        doc_sizes = []
        word_id = []
        word_ct = []
        Yvals = []
        with open(filepath, 'r') as f:
            if sliceID is None:
                # Simple case: read the whole file
                for line in f.readlines():
                    nUnique, d_word_id, d_word_ct = \
                        processLine_ldac__fromstring(line)
                    doc_sizes.append(nUnique)
                    word_id.extend(d_word_id)
                    word_ct.extend(d_word_ct)
            else:
                # Parallel access case: read slice of the file
                # slices will be roughly even in DISKSIZE, not in num lines
                # Inspired by:
                # https://xor0110.wordpress.com/2013/04/13/
                # how-to-read-a-chunk-of-lines-from-a-file-in-python/
                if filesize is None:
                    f.seek(0, 2)
                    filesize = f.tell()
                start = filesize * sliceID / nSlice
                stop = filesize * (sliceID + 1) / nSlice
                if start == 0:
                    f.seek(0)  # goto start of file
                else:
                    f.seek(start - 1)  # start at end of prev slice
                    f.readline()  # then jump to next line brk to start reading
                while f.tell() < stop:
                    line = f.readline()
                    nUnique, d_word_id, d_word_ct = \
                        processLine_ldac__fromstring(line)
                    doc_sizes.append(nUnique)
                    word_id.extend(d_word_id)
                    word_ct.extend(d_word_ct)

        doc_range = np.hstack([0, np.cumsum(doc_sizes)])
        Data = cls(word_id=word_id, word_count=word_ct, nDocTotal=nDocTotal,
                   doc_range=doc_range, vocab_size=vocab_size)
        if len(Yvals) > 0:
            Yvals = toCArray(Yvals)
            if np.allclose(Yvals.sum(),
                           np.int32(Yvals).sum()):
                Data.Yb = np.int32(Yvals)
            else:
                Data.Yr = Yvals
        return Data

    @classmethod
    def read_from_mat(cls, matfilepath, vocabfile=None, **kwargs):
        """ Constructor for loading data from MAT file.

        Pre Condition
        -------
        matfilepath has fields named word_id, word_count, doc_range

        Returns
        -------
        Data : WordsData object
        """
        MatDict = scipy.io.loadmat(matfilepath, **kwargs)
        return cls(vocabfile=vocabfile, **MatDict)

    def __init__(self, word_id=None, word_count=None, doc_range=None,
                 vocab_size=0, vocabList=None, vocabfile=None,
                 summary=None,
                 nDocTotal=None, TrueParams=None, **kwargs):
        ''' Constructor for WordsData object.

        Represents bag-of-words dataset via several 1D vectors,
        where each entry gives the id/count of a single token.

        Parameters
        -------
        word_id : 1D array, size U
            word_id[i] gives the int type for distinct word i.
            Range: [0, 1, 2, 3, ... vocab_size-1]
        word_count : 1D array, size U
            word_count[i] gives the number of times distinct word i
            appears in its corresponding document.
            Range: [1, 2, 3, ...]
        doc_range : 1D array, size nDoc+1
            Defines section of each corpus-wide vector which belongs
            to each document d:
            word_id_d = self.word_id[doc_range[d]:doc_range[d+1]]
            word_ct_d = self.word_count[doc_range[d]:doc_range[d+1]]
        vocab_size : int
            size of set of possible vocabulary words
        vocabList : list of strings
            specifies the word associated with each vocab type id
        nDoc : int
            number of groups (documents) represented in memory
            by this object.
        nDocTotal : int
            total size of the corpus
            will differ from nDoc when this dataset represents a
            small minibatch of some larger corpus.
        TrueParams : None [default], or dict of true parameters.
        '''
        self.word_id = as1D(toCArray(word_id, dtype=np.int32))
        self.word_count = as1D(toCArray(word_count, dtype=np.float64))
        self.doc_range = as1D(toCArray(doc_range, dtype=np.int32))
        self.vocab_size = int(vocab_size)
        self.dim = self.vocab_size
        if summary is not None:
            self.summary = summary

        # Store "true" parameters that generated toy-data, if provided
        if TrueParams is not None:
            self.TrueParams = TrueParams

        # Add dictionary of vocab words, if provided
        if vocabList is not None:
            self.vocabList = vocabList
        elif vocabfile is not None:
            with open(vocabfile, 'r') as f:
                self.vocabList = [x.strip() for x in f.readlines()]

        self._verify_attributes()
        self._set_corpus_size_attributes(nDocTotal)

    def _set_corpus_size_attributes(self, nDocTotal=None):
        ''' Sets nDoc, nObs, and nDocTotal attributes of this WordsData object
        '''
        self.nDoc = self.doc_range.size - 1
        self.nTotalToken = np.sum(self.word_count)
        self.nUniqueToken = self.word_id.size
        if nDocTotal is None:
            self.nDocTotal = self.nDoc
        else:
            self.nDocTotal = int(nDocTotal)

    def _verify_attributes(self):
        ''' Basic runtime checks to make sure attribute dims are correct.
        '''
        assert self.vocab_size > 0
        assert self.word_id.ndim == 1
        assert self.word_id.min() >= 0
        assert self.word_id.max() < self.vocab_size
        assert self.word_count.ndim == 1
        assert self.word_count.min() > 0
        assert self.word_count.size == self.word_id.size
        if self.doc_range.ndim == 2:
            self.doc_range = np.hstack([0, self.doc_range[:, 1]])
        assert self.doc_range.ndim == 1

        docEndBiggerThanStart = self.doc_range[1:] - self.doc_range[:-1]
        assert np.all(docEndBiggerThanStart)

        if hasattr(self, 'vocabList'):
            if len(self.vocabList) != self.vocab_size:
                del self.vocabList

    def get_size(self):
        return self.nDoc

    def get_total_size(self):
        return self.nDocTotal

    def get_dim(self):
        return self.vocab_size

    def get_text_summary(self):
        ''' Returns human-readable description of this dataset.

        Summary might describe source of this dataset, author, etc.

        Returns
        -------
        summary : string
        '''
        if hasattr(self, 'summary'):
            s = self.summary
        else:
            s = 'WordsData'
        return s

    def get_stats_summary(self):
        ''' Returns human-readable summary of this dataset's basic properties
        '''
        s = '  size: %d units (documents)\n' % (self.get_size())
        s += '  vocab size: %d\n' % (self.get_dim())
        s += self.get_doc_stats_summary() + "\n"
        s += self.get_wordcount_summary() + "\n"
        s += self.get_docusagebytype_summary()
        return s

    def get_doc_stats_summary(self, pRange=[0, 5, 50, 95, 100]):
        ''' Get human-readable summary of word-count statistics

        e.g. word counts for smallest, largest, and median-length docs.

        Returns
        ------
        s : string
        '''
        nDistinctWordsPerDoc = np.zeros(self.nDoc)
        nTotalWordsPerDoc = np.zeros(self.nDoc)
        for d in range(self.nDoc):
            start = self.doc_range[d]
            stop = self.doc_range[d + 1]
            nDistinctWordsPerDoc[d] = stop - start
            nTotalWordsPerDoc[d] = self.word_count[start:stop].sum()

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
        s += ' nUniqueTokensPerDoc\n'
        for p in pRange:
            s += "%5s " % ("%.0f" % (np.percentile(nTotalWordsPerDoc, p)))
        s += ' nTotalTokensPerDoc'
        return s

    def get_wordcount_summary(self, bins=[1, 2, 3, 10, 100]):
        """ Get human-readable summary of word counts

        Returns
        -------
        s : string
        """
        binedges = np.asarray(bins)
        binedges = np.hstack([binedges[0] - .5, binedges + .5, np.inf])
        binHeaderStr = ''
        binCountStr = ''
        for e in range(binedges.size - 1):
            bincountVec = np.logical_and(self.word_count >= binedges[e],
                                         self.word_count < binedges[e + 1])
            bincount = np.sum(bincountVec)
            fracMassStr = "%.2f" % (bincount / float(self.word_count.size))
            if bincount == 0:
                fracMassStr = "0"
            elif fracMassStr == "0.00":
                fracMassStr = "%d" % (bincount)  # "<0.01"

            binCountStr += " %6s" % (fracMassStr)
            if e == binedges.size - 2:
                binHeaderStr += " %6s" % (">=" + str(bins[-1]))
            elif binedges[e + 1] - binedges[e] > 1:
                binHeaderStr += " %6s" % ("<" + str(bins[e]))
            else:
                binHeaderStr += " %6s" % (str(bins[e]))
        return "Hist of word_count across tokens \n" \
            + "%s\n%s" % (binHeaderStr, binCountStr)

    def get_docusagebytype_summary(self, bins=[1, 10, 100, .1, .2, .5]):
        """ Get human-readable summary of word type usage across docs.

        Returns
        ------
        s : string
        """
        nUniqueDoc = np.sum(self.getDocTypeCountMatrix() > 0, axis=0)
        bbins = list()
        bNames = list()
        gap = 0
        for b in range(len(bins)):
            if bins[b] < 1:
                binval = bins[b] * self.nDoc
                bName = "%.2f" % (bins[b])
            else:
                binval = bins[b]
                bName = str(binval)
            if b > 1:
                gap = bbins[-1] - bbins[-2]
                if binval - bbins[-1] < gap:
                    continue

            bbins.append(binval)
            bNames.append(bName)

        binHeaderStr = ''
        binCountStr = ''
        binedges = np.hstack([0, np.asarray(bbins), np.inf])
        for e in range(binedges.size - 1):
            bincount = np.sum(np.logical_and(nUniqueDoc >= binedges[e],
                                             nUniqueDoc < binedges[e + 1]))

            fracMassStr = "%.2f" % (bincount / float(self.vocab_size))
            if bincount == 0:
                fracMassStr = "0"
            elif fracMassStr == "1.00":
                fracMassStr = ">.99"
            elif fracMassStr == "0.00":
                fracMassStr = "%6d" % (bincount)

            binCountStr += " %6s" % (fracMassStr)
            if e == binedges.size - 2:
                binHeaderStr += " %6s" % (">=" + bNames[-1])
            else:
                binHeaderStr += " %6s" % ("<" + bNames[e])
        return "Hist of unique docs per word type\n" \
            + "%s\n%s" % (binHeaderStr, binCountStr)

    def getTokenTypeCountMatrix(self):
        ''' Get dense matrix counting vocab usage across all words in dataset

        Returns
        --------
        C : 2D array, size U x vocab_size
            C[n,v] = { word_count[n] iff word_id[n] = v
                     { 0 otherwise
            Each distinct word token n is represented by one entire row
            with only one non-zero entry: at column word_id[n]
        '''
        key = '__TokenTypeCountMat'
        if hasattr(self, key):
            return getattr(self, key)

        C = self.getSparseTokenTypeCountMatrix()
        X = C.toarray()
        setattr(self, key, X)
        return X

    def getSparseTokenTypeCountMatrix(self):
        ''' Get sparse matrix counting vocab usage across all words in dataset

        Returns
        --------
        C : sparse CSC matrix, size U x vocab_size
            C[n,v] = { word_count[n] iff word_id[n] = v
                     { 0 otherwise
            Each distinct word token n is represented by one entire row
            with only one non-zero entry: at column word_id[n]
        '''
        key = '__sparseTokenTypeCountMat'
        if hasattr(self, key):
            return getattr(self, key)

        # Create sparse matrix C from scratch
        indptr = np.arange(self.nUniqueToken + 1)
        C = scipy.sparse.csc_matrix((self.word_count, self.word_id, indptr),
                                    shape=(self.vocab_size, self.nUniqueToken))
        setattr(self, key, C)
        return C

    def getDocTypeCountMatrix(self):
        ''' Get dense matrix counting vocab usage for each document.

        Returns
        --------
        C : 2D array, shape nDoc x vocab_size
            C[d,v] = total count of vocab type v in document d
        '''
        key = '__DocTypeCountMat'
        if hasattr(self, key):
            return getattr(self, key)

        C = self.getSparseDocTypeCountMatrix()
        X = C.toarray()
        setattr(self, key, X)
        return X

    def getSparseDocTypeCountMatrix(self, **kwargs):
        ''' Make sparse matrix counting vocab usage for each document.

        Returns
        -------
        C : sparse CSR matrix, shape nDoc x vocab_size
            C[d,v] = total count of vocab type v in document d
        '''
        # Check cache, return the matrix if we've computed it already
        key = '__sparseDocTypeCountMat'
        if hasattr(self, key):
            return getattr(self, key)

        # Create CSR matrix representation
        C = scipy.sparse.csr_matrix(
            (self.word_count, self.word_id, self.doc_range),
            shape=(self.nDoc, self.vocab_size),
            dtype=np.float64)
        setattr(self, key, C)
        return C

    def clearCache(self):
        for key in ['__TokenTypeCountMat', '__sparseTokenTypeCountMat',
                    '__DocTypeCountMat', '__sparseDocTypeCountMat']:
            if hasattr(self, key):
                del self.__dict__[key]

    def getWordTypeCooccurMatrix(self, dtype=np.float64):
        """ Calculate building blocks for word-word cooccur calculation

        Returns
        -------
        Q : 2D matrix, W x W (where W is vocab_size)
        """
        Q, sameWordVec, _ = self.getWordTypeCooccurPieces(dtype=dtype)
        return self._calcWordTypeCooccurMatrix(Q, sameWordVec, self.nDoc)

    def getWordTypeCooccurPieces(self, dtype=np.float32):
        """ Calculate building blocks for word-word cooccur calculation

        These pieces can be used for incremental construction.

        Returns
        -------
        Q : 2D matrix, W x W (where W is vocab_size)
        sameWordVec : 1D array, size W
        nDoc : scalar
        """
        sameWordVec = np.zeros(self.vocab_size)
        data = np.zeros(self.word_count.shape, dtype=dtype)

        for docID in xrange(self.nDoc):
            start = self.doc_range[docID]
            stop = self.doc_range[docID + 1]
            N = self.word_count[start:stop].sum()
            NNm1 = N * (N - 1)
            sameWordVec[self.word_id[start:stop]] += \
                self.word_count[start:stop] / NNm1
            data[start:stop] = self.word_count[start:stop] / np.sqrt(NNm1)

        # Now, create a sparse matrix that's D x V
        sparseDocWordMat = scipy.sparse.csr_matrix(
            (data, self.word_id, self.doc_range),
            shape=(self.nDoc, self.vocab_size),
            dtype=dtype)
        # Q : V x V
        from sklearn.utils.extmath import safe_sparse_dot
        Q = safe_sparse_dot(
            sparseDocWordMat.T, sparseDocWordMat, dense_output=1)
        return Q, sameWordVec, self.nDoc

    def _calcWordTypeCooccurMatrix(self, Q, sameWordVec, nDoc):
        """ Transform building blocks into the final Q matrix

        Returns
        -------
        Q : 2D array, size W x W (where W is vocab_size)
        """
        Q /= nDoc
        sameWordVec /= nDoc
        diagIDs = np.diag_indices(self.vocab_size)
        Q[diagIDs] -= sameWordVec

        # Fix small numerical issues (like diag entries of -1e-15 instead of 0)
        np.maximum(Q, 0, out=Q)
        return Q

    def add_data(self, WData):
        """ Appends (in-place) provided dataset to this dataset.

        Post Condition
        -------
        word_id, word_count grow by appending info from provided DataObj.
        """
        assert self.vocab_size == WData.vocab_size
        self.word_id = np.hstack([self.word_id, WData.word_id])
        self.word_count = np.hstack([self.word_count, WData.word_count])
        self.doc_range = np.hstack([self.doc_range,
                                    WData.doc_range[1:] + self.doc_range[-1]])
        self.nDoc += WData.nDoc
        self.nDocTotal += WData.nDocTotal
        self.nUniqueToken += WData.nUniqueToken
        self.nTotalToken += WData.nTotalToken

        self.clearCache()
        self._verify_attributes()

    def get_random_sample(self, nDoc, randstate=np.random,
                          candidates=None,
                          p=None):
        ''' Create WordsData object for random subsample of this collection

            Args
            -----
            nDoc : number of documents to choose
            randstate : numpy random number generator

            Returns
            -------
            WordsData : bnpy WordsData instance, with at most nDoc documents
        '''
        if candidates is None:
            nSamples = np.minimum(self.nDoc, nDoc)
            docMask = randstate.choice(self.nDoc, nSamples, replace=False)
        else:
            nSamples = np.minimum(len(candidates), nDoc)
            docMask = randstate.choice(
                candidates, nSamples, replace=False, p=p)
        return self.select_subset_by_mask(docMask=docMask,
                                          doTrackFullSize=False)

    def getRawDataAsSharedMemDict(self):
        ''' Create dict with copies of raw data as shared memory arrays
        '''
        dataShMemDict = dict()
        dataShMemDict['doc_range'] = numpyToSharedMemArray(self.doc_range)
        dataShMemDict['word_id'] = numpyToSharedMemArray(self.word_id)
        dataShMemDict['word_count'] = numpyToSharedMemArray(self.word_count)
        dataShMemDict['vocab_size'] = self.vocab_size
        return dataShMemDict

    def select_subset_by_mask(self, docMask,
                              doTrackFullSize=True,
                              doTrackTruth=False):
        ''' Returns WordsData object representing a subset this dataset.

        Args
        -------
        docMask : 1D array, size nDoc
            each entry indicates a document id to include in subset

        doTrackFullSize : boolean
            If True, return dataset whose nDocTotal attribute equals
            the value of self.nDocTotal. If False, nDocTotal=nDoc.

        Returns
        --------
        Dchunk : WordsData object
        '''
        docMask = np.asarray(docMask, dtype=np.int32)
        nDoc = len(docMask)
        assert np.max(docMask) < self.nDoc
        nUniqueTokenPerDoc = \
            self.doc_range[docMask + 1] - self.doc_range[docMask]
        nUniqueToken = np.sum(nUniqueTokenPerDoc)
        word_id = np.zeros(nUniqueToken, dtype=self.word_id.dtype)
        word_count = np.zeros(nUniqueToken, dtype=self.word_count.dtype)
        doc_range = np.zeros(nDoc + 1, dtype=self.doc_range.dtype)

        # Fill in new word_id, word_count, and doc_range
        startLoc = 0
        newMask = list()
        for d in xrange(nDoc):
            start = self.doc_range[docMask[d]]
            stop = self.doc_range[docMask[d] + 1]
            endLoc = startLoc + (stop - start)
            newMask.extend(range(start, stop))
            word_count[startLoc:endLoc] = self.word_count[start:stop]
            word_id[startLoc:endLoc] = self.word_id[start:stop]
            doc_range[d] = startLoc
            startLoc += (stop - start)
        doc_range[-1] = nUniqueToken

        nDocTotal = None
        if doTrackFullSize:
            nDocTotal = self.nDocTotal

        if hasattr(self, 'alwaysTrackTruth'):
            doTrackTruth = doTrackTruth or self.alwaysTrackTruth
        hasTrueZ = hasattr(self, 'TrueParams') and 'Z' in self.TrueParams
        if doTrackTruth and hasTrueZ:
            newTrueParams = dict()
            for key, arr in self.TrueParams.items():
                if key == 'Z' or key == 'resp':
                    newMask = np.asarray(newMask, dtype=np.int32)
                    newTrueParams[key] = arr[newMask]
                elif isinstance(arr, np.ndarray):
                    newTrueParams[key] = arr.copy()
                else:
                    newTrueParams[key] = arr
        else:
            newTrueParams = None

        return WordsData(word_id, word_count, doc_range, self.vocab_size,
                         nDocTotal=nDocTotal, TrueParams=newTrueParams)

    @classmethod
    def CreateToyDataSimple(cls, nDoc=10, nUniqueTokensPerDoc=10,
                            vocab_size=25, **kwargs):
        ''' Creates a toy instance of WordsData (good for debugging)

        Returns
        ------
        Data : WordsData object
        '''
        PRNG = np.random.RandomState(0)
        word_id = list()
        word_count = list()
        doc_range = np.zeros(nDoc + 1)
        for dd in range(nDoc):
            wID = PRNG.choice(
                vocab_size, size=nUniqueTokensPerDoc, replace=False)
            wCount = PRNG.choice(
                np.arange(1, 5), size=nUniqueTokensPerDoc, replace=True)
            word_id.extend(wID)
            word_count.extend(wCount)
            start = nUniqueTokensPerDoc * dd
            doc_range[dd] = start
        doc_range[-1] = start + nUniqueTokensPerDoc

        return cls(word_id=word_id, word_count=word_count,
                   doc_range=doc_range, vocab_size=vocab_size)

    @classmethod
    def CreateToyDataFromMixModel(cls, seed=101,
                                  nDocTotal=None,
                                  nWordsPerDoc=None,
                                  nWordsPerDocFunc=None,
                                  beta=None,
                                  topics=None,
                                  **kwargs):
        ''' Generates WordsData dataset via mixture generative model.

        Returns
        ------
        Data : WordsData object
        '''
        from bnpy.util import RandUtil
        PRNG = np.random.RandomState(seed)

        K = topics.shape[0]
        V = topics.shape[1]
        # Make sure topics sum to one
        topics = topics / topics.sum(axis=1)[:, np.newaxis]
        assert K == beta.size

        doc_range = np.zeros(nDocTotal + 1)
        wordIDsPerDoc = list()
        wordCountsPerDoc = list()

        resp = np.zeros((nDocTotal, K))
        Ks = range(K)

        # startPos : tracks start index for current doc within corpus-wide
        # lists
        startPos = 0
        for d in xrange(nDocTotal):
            # Draw single topic assignment for this doc
            k = RandUtil.choice(Ks, beta, PRNG)
            resp[d, k] = 1

            # Draw the observed words for this doc
            # wordCountBins: V x 1 vector, entry v counts appearance of word v
            wordCountBins = RandUtil.multinomial(nWordsPerDoc,
                                                 topics[k, :], PRNG)

            # Record word_id, word_count, doc_range
            wIDs = np.flatnonzero(wordCountBins > 0)
            wCounts = wordCountBins[wIDs]
            assert np.allclose(wCounts.sum(), nWordsPerDoc)
            wordIDsPerDoc.append(wIDs)
            wordCountsPerDoc.append(wCounts)
            doc_range[d] = startPos
            startPos += wIDs.size

        # Package up all data
        word_id = np.hstack(wordIDsPerDoc)
        word_count = np.hstack(wordCountsPerDoc)
        doc_range[-1] = word_count.size

        # Make TrueParams dict
        TrueParams = dict(K=K, topics=topics, beta=beta, resp=resp)

        Data = WordsData(
            word_id, word_count, doc_range, V, TrueParams=TrueParams)
        return Data

    @classmethod
    def CreateToyDataFromLDAModel(cls, seed=101,
                                  nDocTotal=None,
                                  nWordsPerDoc=None,
                                  nWordsPerDocFunc=None,
                                  topic_prior=None, topics=None,
                                  gamma=None, probs=None,
                                  **kwargs):
        ''' Generates WordsData dataset via LDA generative model.


        Returns
        ------
        Data : WordsData object
        '''
        if topic_prior is None:
            topic_prior = gamma * probs
        from bnpy.util import RandUtil
        PRNG = np.random.RandomState(seed)

        K = topics.shape[0]
        V = topics.shape[1]
        # Make sure topics sum to one
        topics = topics / topics.sum(axis=1)[:, np.newaxis]
        assert K == topic_prior.size

        doc_range = np.zeros(nDocTotal + 1)
        wordIDsPerDoc = list()
        wordCountsPerDoc = list()

        Pi = np.zeros((nDocTotal, K))
        respPerDoc = list()

        # startPos : tracks start index for current doc within corpus-wide
        # lists
        startPos = 0
        for d in xrange(nDocTotal):
            # Draw topic appearance probabilities for this document
            Pi[d, :] = PRNG.dirichlet(topic_prior)

            if nWordsPerDocFunc is not None:
                nWordsPerDoc = nWordsPerDocFunc(PRNG)

            # Draw the topic assignments for this doc
            # Npercomp : K-vector, Npercomp[k] counts appearance of topic k
            Npercomp = RandUtil.multinomial(nWordsPerDoc, Pi[d, :], PRNG)

            # Draw the observed words for this doc
            # wordCountBins: V x 1 vector, entry v counts appearance of word v
            wordCountBins = np.zeros(V)
            for k in xrange(K):
                wordCountBins += RandUtil.multinomial(Npercomp[k],
                                                      topics[k, :], PRNG)

            # Record word_id, word_count, doc_range
            wIDs = np.flatnonzero(wordCountBins > 0)
            wCounts = wordCountBins[wIDs]
            assert np.allclose(wCounts.sum(), nWordsPerDoc)
            wordIDsPerDoc.append(wIDs)
            wordCountsPerDoc.append(wCounts)
            doc_range[d] = startPos
            startPos += wIDs.size

            # Record expected local parameters (LP)
            curResp = (topics[:, wIDs] * Pi[d, :][:, np.newaxis]).T
            respPerDoc.append(curResp)

        word_id = np.hstack(wordIDsPerDoc)
        word_count = np.hstack(wordCountsPerDoc)
        doc_range[-1] = word_count.size

        # Make TrueParams dict
        resp = np.vstack(respPerDoc)
        resp /= resp.sum(axis=1)[:, np.newaxis]
        TrueParams = dict(K=K, topics=topics,
                          beta=topic_prior / topic_prior.sum(),
                          topic_prior=topic_prior, resp=resp)

        Data = WordsData(
            word_id, word_count, doc_range, V, TrueParams=TrueParams)
        return Data

    def WriteToFile_ldac(self, filepath,
                         min_word_index=0):
        ''' Write contents of this dataset to plain-text file in "ldac" format.

        Each line of file represents one document, and has format
        [U] [term1:count1] [term2:count2] ... [termU:countU]

        Args
        -------
        filepath : path where data should be saved

        Post Condition
        -------
        Writes state of this dataset to file.
        '''
        word_id = self.word_id
        if min_word_index > 0:
            word_id = word_id + min_word_index
        with open(filepath, 'w') as f:
            for d in xrange(self.nDoc):
                dstart = self.doc_range[d]
                dstop = self.doc_range[d + 1]
                nUniqueInDoc = dstop - dstart
                idct_list = ["%d:%d" % (word_id[n], self.word_count[n])
                             for n in xrange(dstart, dstop)]
                if hasattr(self, 'Yr'):
                    docstr = "%d %.4f %s" % (
                        nUniqueInDoc, self.Yr[d], ' '.join(idct_list))
                elif hasattr(self, 'Yb'):
                    docstr = "%d %d %s" % (
                        nUniqueInDoc, self.Yb[d], ' '.join(idct_list))
                else:
                    docstr = "%d %s" % (
                        nUniqueInDoc, ' '.join(idct_list))
                f.write(docstr + '\n')

    def WriteToFile_tokenlist(self, filepath, min_word_index=1):
        ''' Write contents of this dataset to MAT file in tokenlist format

        Post Condition
        ------
        Writes state of this dataset to file.
        '''
        word_id = self.word_id
        if min_word_index > 0:
            word_id = word_id + min_word_index

        MatVars = dict()
        MatVars['tokensByDoc'] = np.empty((1, self.nDoc), dtype=object)
        for d in xrange(self.nDoc):
            start = self.doc_range[d]
            stop = self.doc_range[d + 1]
            nTokens = np.sum(self.word_count[start:stop])
            tokenvec = np.zeros(nTokens, dtype=word_id.dtype)

            a = 0
            for n in xrange(start, stop):
                tokenvec[a:a + self.word_count[n]] = word_id[n]
                a += self.word_count[n]

            assert tokenvec.min() >= min_word_index
            MatVars['tokensByDoc'][0, d] = tokenvec
        scipy.io.savemat(filepath, MatVars, oned_as='row')

    def writeWholeDataInfoFile(self, filepath):
        ''' Write relevant whole-dataset key properties to plain text.

        For WordsData, this includes the fields
        * vocab_size
        * nDocTotal

        Post Condition
        --------------
        Writes text file at provided file path.
        '''
        with open(filepath, 'w') as f:
            f.write("vocab_size = %d\n" % (self.vocab_size))
            f.write("nDocTotal = %d\n" % (self.nDocTotal))

    def getDataSliceFunctionHandle(self):
        """ Return function handle that can make data slice objects.

        Useful with parallelized algorithms,
        when we need to use shared memory.

        Returns
        -------
        f : function handle
        """
        return makeDataSliceFromSharedMem


def makeDataSliceFromSharedMem(dataShMemDict,
                               cslice=(0, None),
                               batchID=None):
    """ Create data slice from provided raw arrays and slice indicators.

    Returns
    -------
    Dslice : namedtuple with same fields as WordsData object
        * vocab_size
        * doc_range
        * word_id
        * word_count
        * nDoc
        * dim
        Represents subset of documents identified by cslice tuple.

    Example
    -------
    >>> Data = WordsData.CreateToyDataSimple(nDoc=10)
    >>> shMemDict = Data.getRawDataAsSharedMemDict()
    >>> Dslice = makeDataSliceFromSharedMem(shMemDict)
    >>> np.allclose(Data.doc_range, Dslice.doc_range)
    True
    >>> np.allclose(Data.word_id, Dslice.word_id)
    True
    >>> Data.vocab_size == Dslice.vocab_size
    True
    >>> Aslice = makeDataSliceFromSharedMem(shMemDict, (0, 3))
    >>> Aslice.nDoc
    3
    >>> np.allclose(Aslice.doc_range, Dslice.doc_range[0:4])
    True
    """
    if batchID is not None and batchID in dataShMemDict:
        dataShMemDict = dataShMemDict[batchID]

    # Make local views (NOT copies) to shared mem arrays
    doc_range = sharedMemToNumpyArray(dataShMemDict['doc_range'])
    word_id = sharedMemToNumpyArray(dataShMemDict['word_id'])
    word_count = sharedMemToNumpyArray(dataShMemDict['word_count'])
    vocab_size = int(dataShMemDict['vocab_size'])

    if cslice is None:
        cslice = (0, doc_range.size - 1)
    elif cslice[1] is None:
        cslice = (0, doc_range.size - 1)

    tstart = doc_range[cslice[0]]
    tstop = doc_range[cslice[1]]
    keys = ['vocab_size', 'doc_range',
            'word_id', 'word_count', 'nDoc', 'dim']
    Dslice = namedtuple("WordsDataTuple", keys)(
        vocab_size=vocab_size,
        doc_range=doc_range[cslice[0]:cslice[1] + 1] - doc_range[cslice[0]],
        word_id=word_id[tstart:tstop],
        word_count=word_count[tstart:tstop],
        nDoc=cslice[1] - cslice[0],
        dim=vocab_size,
    )
    return Dslice


def processLine_ldac__splitandzip(line):
    """

    Examples
    --------
    >>> a, b, c = processLine_ldac__splitandzip('5 66:6 77:7 88:8')
    >>> print a
    5
    >>> print b
    ('66', '77', '88')
    >>> print c
    ('6', '7', '8')
    """
    Fields = line.strip().split(' ')
    nUnique = int(Fields[0])
    doc_word_id, doc_word_ct = zip(
        *[x.split(':') for x in Fields[1:]])
    return nUnique, doc_word_id, doc_word_ct


def processLine_ldac__fromstring(line):
    """
    Examples
    --------
    >>> a, b, c = processLine_ldac__fromstring('5 66:6 77:7 88:8')
    >>> print a
    5.0
    >>> print b
    [ 66.  77.  88.]
    >>> print c
    [ 6.  7.  8.]
    """
    line = line.replace(':', ' ')
    data = np.fromstring(line, sep=' ', dtype=np.int32)
    return data[0], data[1::2], data[2::2]