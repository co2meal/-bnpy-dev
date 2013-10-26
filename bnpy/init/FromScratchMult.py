'''
FromScratchMult.py

Initialize params of HModel with multinomial observations from scratch.
'''
import numpy as np
from scipy.special import digamma

def init_global_params(hmodel, Data, initname='randexamples', seed=0, K=0, **kwargs):
    
    nObsTotal = Data.nObsTotal
    doc_range = Data.doc_range
    nDocTotal = Data.nDocTotal
    PRNG = np.random.RandomState(seed)
    if initname == 'randexamples':
        ''' Choose K items uniformly at random from the Data
        '''
        print 'Initializing a completely random Admixture Model'

        doc_variational = np.zeros( (nDocTotal, K) )
        
        word_variational = PRNG.rand(nObsTotal, K)
        word_variational /= word_variational.sum(axis=1)[:,np.newaxis]
        
        for d in xrange(nDocTotal):
            start,stop = doc_range[d,:]
            doc_variational[d,:] = np.sum(word_variational[start:stop,:],axis=0)
            
        print 'Finished Random Initialization'
            
    elif initname == 'randselect':
        doc_variational = np.zeros( (nDocTotal, K) )
        word_variational = np.zeros(nObsTotal, K)
        # Pick K observed words at random
        nSelect = np.maximum(Data.nObs/100, K)
        selectIDs = PRNG.choice(Data.nObs, nSelect, replace=False)
        for pos in range(len(selectIDs)):
          word_variational[selectIDs[pos], pos % K] = 1.0

    elif initname == 'truth':
        word_variational = np.zeros( (nObsTotal, K) ) + .1
        doc_variational = Data.true_td.T
        
        # set each word-token variational parameter 
        #  to its true document x topic weights 
        for d in xrange(nDocTotal):
            start,stop = doc_range[d,:]
            word_variational[start:stop,:] = doc_variational[d,:]
           
    if hmodel.getAllocModelName().count('HDP') > 0:
      zeroPad = np.zeros((Data.nDoc,1))
      alph = np.hstack([doc_variational, zeroPad])
      alph = np.maximum(alph, 1e-20)
      LP = dict(E_logPi=np.log(alph),
                 word_variational=word_variational)
    else:
      LP = dict(doc_variational=doc_variational, 
                word_variational=word_variational)

    SS = hmodel.get_global_suff_stats(Data, LP)
    hmodel.update_global_params(SS)

def getLPfromResp(Resp, Data, smoothMass=0.01):
    D = Data.nDoc
    K = Resp.shape[1]
    # DocTopicCount matrix : D x K matrix
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
