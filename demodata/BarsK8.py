'''
BarsK8.py

Toy Bars data, with K=8 topics
4 horizontal, and 4 vertical.
'''
import numpy as np

from bnpy.data import WordsData, AdmixMinibatchIterator

# DEFAULT PARAMS (can be changed by cmdline args)
Defaults = dict( nDocTotal=1000, nWordsPerDoc=100)

# FIXED DATA GENERATION PARAMS
K = 8 # Number of topics
V = 16 # Vocabulary Size
nWordsPerDoc = 100 # words per document
gamma = 0.5 # hyperparameter over doc-topic distribution

# TOPIC by WORD distribution
true_tw = np.zeros( (K,V) )
true_tw[0,:] = [ 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
true_tw[1,:] = [ 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
true_tw[2,:] = [ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
true_tw[3,:] = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
true_tw[4,:] = [ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
true_tw[5,:] = [ 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
true_tw[6,:] = [ 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]
true_tw[7,:] = [ 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]

# Add "smoothing" term to each entry of the topic-word matrix
# With V = 16 and 8 sets of bars,
#  smoothMass=0.02 yields 0.944 probability of drawing "on topic" word
smoothMass = 0.02
true_tw += smoothMass

# ensure that true_tw is a probability
for k in xrange(K):
    true_tw[k,:] /= np.sum( true_tw[k,:] )

# GLOBAL PROB DISTRIBUTION OVER TOPICS
trueBeta = np.asarray([4., 3, 2, 1, 4, 3, 2, 1], dtype=np.float64)
trueBeta /= trueBeta.sum()

Defaults['docTopicParamVec'] = gamma*trueBeta
Defaults['TopicWordProbs'] = true_tw

def get_data_info(**kwargs):
    if 'nDocTotal' in kwargs:
      nDocTotal = kwargs['nDocTotal']
    else:
      nDocTotal = Defaults['nDocTotal']
    return 'Toy Bars Data. Ktrue=%d. nDocTotal=%d.' % (K, nDocTotal)

def get_data(seed=8675309, **kwargs):
    ''' 
        Args
        -------
        seed
        nDocTotal
        nWordsPerDoc
    '''
    Data = genWordsData(seed, **kwargs)
    Data.summary = get_data_info(**kwargs)
    return Data

def get_minibatch_iterator(seed=8675309, nBatch=10, nLap=1,
                           dataorderseed=0, **kwargs):
    '''
        Args
        -------
        seed
        nDocTotal
        nWordsPerDoc
    '''
    Data = genWordsData(seed, **kwargs)
    DataIterator = AdmixMinibatchIterator(Data, 
                        nBatch=nBatch, nLap=nLap, dataorderseed=dataorderseed)
    DataIterator.summary = get_data_info(**kwargs)
    return DataIterator

def genWordsData(seed=0, nDocTotal=None, nWordsPerDoc=None, 
                      docTopicParamVec=None, TopicWordProbs=None,
                      **kwargs):
    ''' Generates toy bars dataset using defined global structure.
    '''
    if nDocTotal is None:
      nDocTotal = Defaults['nDocTotal']
    if nWordsPerDoc is None:
      nWordsPerDoc = Defaults['nWordsPerDoc']
    if docTopicParamVec is None:
      docTopicParamVec = Defaults['docTopicParamVec']
    if TopicWordProbs is None:
      TopicWordProbs = Defaults['TopicWordProbs']

    K = TopicWordProbs.shape[0]
    V = TopicWordProbs.shape[1]

    PRNG = np.random.RandomState( seed )
    doc_range = np.zeros((nDocTotal, 2))

    # true document x topic proportions
    true_td = np.zeros((K,nDocTotal)) 
    
    wordIDsPerDoc = list()
    wordCountsPerDoc = list()

    # counter for tracking the start index for current document 
    #  within the corpus-wide word lists
    startPos = 0
    for d in xrange(nDocTotal):
        true_td[:,d] = PRNG.dirichlet(docTopicParamVec) 
        Npercomp = PRNG.multinomial(nWordsPerDoc, true_td[:,d])

        # wordCountBins: V x 1 vector
        #   entry v counts # times vocab word v appears in current doc
        wordCountBins = np.zeros(V)
        for k in xrange(K):
            wordCountBins += PRNG.multinomial(Npercomp[k], TopicWordProbs[k,:])

        wIDs = np.flatnonzero(wordCountBins > 0)
        wCounts = wordCountBins[wIDs]
        assert np.allclose( wCounts.sum(), nWordsPerDoc)
        
        wordIDsPerDoc.append(wIDs)
        wordCountsPerDoc.append(wCounts)

        #start and stop ids for documents
        doc_range[d,0] = startPos
        doc_range[d,1] = startPos + wIDs.size  
        startPos += wIDs.size
    
    word_id = np.hstack(wordIDsPerDoc)
    word_count = np.hstack(wordCountsPerDoc)

    #Insert all important stuff in myDict
    myDict = dict(true_K=K, true_beta=trueBeta,
                  true_tw=true_tw, true_td=true_td,
                  )
    # Necessary items
    myDict["doc_range"] = doc_range
    myDict["word_id"] = word_id
    myDict["word_count"] = word_count
    myDict["vocab_size"] = V
    return WordsData(**myDict)

if __name__ == '__main__':
  import bnpy.viz.BarsViz
  WData = genWordsData(1234)
  bnpy.viz.BarsViz.plotExampleBarsDocs(WData)