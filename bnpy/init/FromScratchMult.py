'''
FromScratchMult.py

Initialize params of HModel with multinomial observations from scratch.
'''
import numpy as np
from scipy.special import digamma
from scipy.cluster import vq

def init_global_params(hmodel, Data, initname='randexamples', seed=0, K=0, **kwargs):
    
    nObsTotal = Data.nObsTotal
    doc_range = Data.doc_range
    nDocTotal = Data.nDocTotal
    PRNG = np.random.RandomState(seed)
    if initname == 'randexamples':
        ''' Choose K items uniformly at random from the Data
        '''
        doc_variational = np.zeros( (nDocTotal, K) )
        
        word_variational = PRNG.rand(nObsTotal, K)
        word_variational /= word_variational.sum(axis=1)[:,np.newaxis]
        
        for d in xrange(nDocTotal):
            start,stop = doc_range[d,:]
            doc_variational[d,:] = np.sum(word_variational[start:stop,:],axis=0)
 
    elif initname == 'randselect':
        doc_variational = np.zeros( (nDocTotal, K) )
        word_variational = np.zeros(nObsTotal, K)
        # Pick K observed words at random
        nSelect = np.maximum(Data.nObs/100, K)
        selectIDs = PRNG.choice(Data.nObs, nSelect, replace=False)
        for pos in range(len(selectIDs)):
          word_variational[selectIDs[pos], pos % K] = 1.0
          
    elif initname == 'kmeans':
        # Runs the k-means initialization on sparse matrix dw
        dw = Data.to_sparse_dw()
        doc_range = Data.doc_range
        # for now we will make it dense since scipy kmeans is not working all that well
        dw_w = vq.whiten(dw.todense())
        Z = vq.kmeans2(dw_w, K)
        # will this be used?
        # lambda_temp = Z[0] 
        labels = Z[1] 
        doc_variational = np.zeros((nDocTotal, K)) + 1.0
        word_variational = np.zeros((nObsTotal, K)) + 1.0
        # Loop through documents and put mass on k that k-means found for documents and tokens
        for d in xrange(nDocTotal):
            doc_variational[d, labels[d]] += doc_range[d,1] - doc_range[d,0]
            word_variational[doc_range[d,0]:doc_range[d,1], labels[d]] += 1.0  
        
        # Need to normalize word_variational parameters
        word_variational /= word_variational.sum(axis=1)[:,np.newaxis]
        
    elif initname == 'truth':
        word_variational = np.zeros( (nObsTotal, K) ) + .1
        doc_variational = Data.true_td.T
        
        # set each word-token variational parameter 
        #  to its true document x topic weights 
        for d in xrange(nDocTotal):
            start,stop = doc_range[d,:]
            word_variational[start:stop,:] = doc_variational[d,:]
           
    if hmodel.getAllocModelName().count('HDP') > 0:
      LP = getLPfromResp(word_variational, Data)
      #zeroPad = np.zeros((Data.nDoc,1))
      #alph = np.hstack([doc_variational, zeroPad])
      #alph = np.maximum(alph, 1e-20)
      #LP = dict(alphaPi=alph,
      #           word_variational=word_variational)
      #LP = hmodel.allocModel.calc_ElogPi(LP)

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
