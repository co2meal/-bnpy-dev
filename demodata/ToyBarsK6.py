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
alpha = 2 # hyperparameter over document-topic distributions
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

'''
def sample_data_as_dict():
  BoW = list()
  nObs = 0
  GroupIDs = list()
  nUniqueEntry = 0
  for docID in xrange( D ):
    w = np.random.dirichlet( alpha*np.ones(K) )
    Npercomp = np.random.multinomial( Nperdoc, w)
    docDict = defaultdict( int )
    for k in xrange(K):
      wordCounts =np.random.multinomial(  Npercomp[k], true_tw[k] )
      for (wordID,count) in enumerate(wordCounts):
        if count == 0: 
          continue
        docDict[wordID] += count
        nObs += count
    nDistinctEntry = len(docDict )
    GroupIDs.append( (nUniqueEntry,nUniqueEntry+nDistinctEntry) )  
    nUniqueEntry += nDistinctEntry
    BoW.append( docDict)
  return BoW, nObs, nUniqueEntry, GroupIDs

def sample_data_as_matrix( Npercomp ):
  X = np.zeros( (Npercomp.sum(), V) )  
  for k in range(K):
    wordCounts =np.random.multinomial(  Npercomp[k], true_tw[k] )
    for (vv,count) in enumerate( wordCounts):
      X[ rowID, vv] = count
  return {'X':X, 'nObs':X.shape[0]}

def sample_data_from_comp( k, Nk ):
  return np.random.multinomial( Nk, true_tw[k] )
  
def print_data_info( modelName ):
  print 'Easy-to-learn toy data for K=3 Bernoulli Obs Model'
  print '  Mix weights:  '
  print '                ', np2flatstr( w )
  print '  Topic-word Probs:  '
  for k in range( K ):
    print '                ', np2flatstr( true_tw[k] )

def get_data_by_groups( seed=8675309, **kwargs ):
  if seed is not None:
    np.random.seed( seed )
  BoW, nObs, nEntry, GroupIDs = sample_data_as_dict()
  Data = dict( BoW=BoW, nObsEntry=nEntry, nObs=nObs, nDoc=D, nVocab=V, GroupIDs=GroupIDs, nGroup=len(GroupIDs) )
  return Data

def minibatch_generator(  batch_size=1000, nBatch=50, nRep=1, seed=8675309, **kwargs):
  for repID in range( nRep ):
    np.random.seed( seed )
    for batchID in range( nBatch ):
      pass

def np2flatstr( X, fmt='% 7.2f' ):
  return ' '.join( [fmt % x for x in X.flatten() ] )  
'''
