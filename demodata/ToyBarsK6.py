'''
EasyToyWordData
'''
import numpy as np

from bnpy.data import WordsData
from collections import defaultdict

K = 6 # Number of topics
V = 9 # Vocabulary Size
D = 250 # number of total documents
Nperdoc = 100 #words per document
alpha = 0.5 # hyperparameter over document-topic distributions
beta  = 0.1 # hyperparameter over topic by word distributions

w = np.ones( K ) # vector of ones
w /= w.sum() # uniform distribution over topics

# Create topic by word distribution
true_tw = np.zeros( (K,V) )
true_tw[0,:] = [ 1, 1, 1, 0, 0, 0, 0, 0, 0]
true_tw[1,:] = [ 0, 0, 0, 1, 1, 1, 0, 0, 0]
true_tw[2,:] = [ 0, 0, 0, 0, 0, 0, 1, 1, 1]
true_tw[3,:] = [ 1, 0, 0, 1, 0, 0, 1, 0, 0]
true_tw[4,:] = [ 0, 1, 0, 0, 1, 0, 0, 1, 0]
true_tw[5,:] = [ 0, 0, 1, 0, 0, 1, 0, 0, 1]

# add prior
true_tw += beta

# total number of observations
nObs = V * D

# ensure that true_tw is a probability
for k in xrange(K):
    true_tw[k,:] /= np.sum( true_tw[k,:] )

# 8675309, sounds like a phone number...
def get_data(seed=8675309, nObsTotal=25000, **kwargs):
# words is a dictionary that contains WC, DOCID
    words_dict = get_BoW(seed)
    Data = WordsData( words_dict )
    Data.summary = get_data_info()
    return Data

def get_BoW(seed):
    # DOCID is a list containing the start,stop indices for a document corresponding to the row of WC
    nObsTotal = 0
    DOC_ID = np.zeros( (D, 2) )
    nUniqueEntry = 0 # counter to calculate document id locations
    true_td = np.zeros( (K,D) ) # true document x topic proportions
    PRNG = np.random.RandomState( seed )
    WCD = list() # document based word count list (most efficient)
    for d in xrange( D ):
        # sample topic distribution for document
        true_td[:,d] = PRNG.dirichlet( alpha*np.ones(K) ) 
        Npercomp = np.random.multinomial( Nperdoc, true_td[:,d])
        temp_word_count = defaultdict( int )
        for k in xrange(K):
            wordCounts = np.random.multinomial(  Npercomp[k], true_tw[k,:] )
            for (wordID,count) in enumerate(wordCounts):
                if count == 0: 
                    continue
                temp_word_count[wordID] += count
                nObsTotal += count
        nDistinctEntry = len( temp_word_count )
        WCD.append(temp_word_count)
        
        #start and stop ids for documents
        DOC_ID[d,0] = nUniqueEntry
        DOC_ID[d,1] = nUniqueEntry+nDistinctEntry  
        nUniqueEntry += nDistinctEntry
    
    # WC is a nUniqueEntry x 2 matrix where col1 = word_id, col2 = word_freq
    WC = np.zeros( (nUniqueEntry, 2) )    
    ii = 0
    for d in xrange(D):
        for (key,value) in WCD[d].iteritems():
            WC[ii,0] = key
            WC[ii,1] = value
            ii += 1
        
    #Insert all important stuff in myDict
    myDict = defaultdict()
    myDict["true_tw"] = true_tw
    myDict["true_td"] = true_td
    myDict["true_K"] = K
    myDict["DOC_ID"] = DOC_ID
    myDict["WC"] = WC
    myDict["WCD"] = WCD # Probably not used in inference
    myDict["nObsTotal"] = nObsTotal
    myDict["nObs"] = nUniqueEntry
    myDict["nWords"] = V
    myDict["nDocs"] = D
    return myDict

def get_data_info():
    return 'Toy Bars Data. Ktrue=%d. D=%d.' % (K,D)

def plot_documents():
  words_dict = get_BoW(1234)
  