'''
FromScratchMult.py

Initialize params of HModel with multinomial observations from scratch.
'''
import numpy as np
import KMeansRex
from scipy.special import digamma
from scipy.cluster import vq

def init_global_params(hmodel, Data, initname='randexamples', seed=0, K=0, **kwargs):
    
    doc_range = Data.doc_range
    PRNG = np.random.RandomState(seed)
    if initname == 'randexamples':
        ''' Choose K documents at random
        '''
        DocWord = Data.to_sparse_docword_matrix()
        chosenDocIDs = PRNG.choice(Data.nDoc, K, replace=False)
        PhiTopicWord = np.asarray(DocWord[chosenDocIDs].todense())
        PhiTopicWord += 0.01 * PRNG.rand(K, Data.vocab_size)
        PhiTopicWord /= PhiTopicWord.sum(axis=1)[:,np.newaxis]
        beta = np.ones(K)
        hmodel.set_global_params(K=K, beta=beta, phi=PhiTopicWord)
        return

    elif initname == 'randsoftpartition':
        ''' Assign responsibility for K topics at random to all words
        '''        
        word_variational = PRNG.rand(Data.nObs, K)
        word_variational /= word_variational.sum(axis=1)[:,np.newaxis]
        doc_variational = np.zeros((Data.nDoc, K))
        for d in xrange(Data.nDoc):
            start,stop = doc_range[d,:]
            doc_variational[d,:] = np.sum(word_variational[start:stop,:],axis=0)
    elif initname == 'plusplus':
        DocWord = Data.to_sparse_docword_matrix().toarray()
        PhiTopicWord = KMeansRex.SampleRowsPlusPlus(DocWord, K, seed=seed)
        PhiTopicWord += 0.01 * PRNG.rand(K, Data.vocab_size)
        PhiTopicWord /= PhiTopicWord.sum(axis=1)[:,np.newaxis]
        beta = np.ones(K)
        hmodel.set_global_params(K=K, beta=beta, phi=PhiTopicWord)
        return
    elif initname == 'kmeansplusplus':
        DocWord = Data.to_sparse_docword_matrix().toarray()
        PhiTopicWord, Z = KMeansRex.RunKMeans(DocWord, K, initname='plusplus',
                                              Niter=10, seed=seed)
        PhiTopicWord += 0.001 * PRNG.rand(K, Data.vocab_size)
        PhiTopicWord /= PhiTopicWord.sum(axis=1)[:,np.newaxis]
        beta = np.ones(K)
        hmodel.set_global_params(K=K, beta=beta, phi=PhiTopicWord)
        return      
    elif initname == 'kmeans':
        ''' Create topics via kmeans analysis of the doc-word matrix
            WARNING: Not sure if this is a good idea yet...
        '''
        # Runs the k-means initialization on sparse matrix dw
        dw = Data.to_sparse_docword_matrix()
        doc_range = Data.doc_range
        dw_w = vq.whiten(dw.todense())
        Z = vq.kmeans2(dw_w, K)
        labels = Z[1] 
        doc_variational = np.zeros((Data.nDoc, K)) + 1.0
        word_variational = np.zeros((Data.nObs, K)) + 1.0
        # Loop through documents and put mass on k that k-means found for documents and tokens
        for d in xrange(Data.nDoc):
            doc_variational[d, labels[d]] += doc_range[d,1] - doc_range[d,0]
            word_variational[doc_range[d,0]:doc_range[d,1], labels[d]] += 1.0  
        
        # Need to normalize word_variational parameters
        word_variational /= word_variational.sum(axis=1)[:,np.newaxis]
        
    elif initname == 'truth':
        word_variational = np.zeros( (Data.nObs, K) ) + .1
        doc_variational = Data.true_td.T
        
        # set each word-token variational parameter 
        #  to its true document x topic weights 
        for d in xrange(Data.nDoc):
            start,stop = doc_range[d,:]
            word_variational[start:stop,:] = doc_variational[d,:]
           
    if hmodel.getAllocModelName().count('HDP') > 0:
      LP = getLPfromResp(word_variational, Data)
    else:
      LP = dict(doc_variational=doc_variational, 
                word_variational=word_variational)

    SS = hmodel.get_global_suff_stats(Data, LP)
    hmodel.update_global_params(SS)

def getLPfromResp(Resp, Data, smoothMass=0.01):
    D = Data.nDoc
    K = Resp.shape[1]
    DocTopicC = np.zeros((D, K))
    for dd in range(D):
      start,stop = Data.doc_range[dd,:]
      DocTopicC[dd,:] = np.dot(Data.word_count[start:stop],        
                               Resp[start:stop,:]
                               )
    # Alpha and ElogPi : D x K+1 matrices
    padCol = smoothMass * np.ones((D,1))
    alph = np.hstack( [DocTopicC + smoothMass, padCol])    
    ElogPi = digamma(alph) - digamma(alph.sum(axis=1))[:,np.newaxis]
    assert ElogPi.shape == (D,K+1)
    return dict(word_variational =Resp, 
              E_logPi=ElogPi, alphaPi=alph,
              DocTopicCount=DocTopicC)
