'''
FromScratchGauss.py

Initialize params of a mixture model with gaussian observations from scratch.
'''
import numpy as np

def init_global_params(hmodel, Data, initname='randexamples', seed=0, K=0, **kwargs):
    
    nObs = Data.nObs
    DOC_ID = Data.DOC_ID
    nDocs = Data.nDocs
    if initname == 'randexamples':
        ''' Choose K items uniformly at random from the Data'''
        word_variational = np.zeros( (nObs, K) ) + .1 # initialize local word-level variational parameters word_variational
        doc_variational = np.zeros( (nDocs, K) ) # initialize local document-level variational parameters doc_variational 
        for i in xrange(nObs):
            k = np.round(np.random.rand()*K)-1
            word_variational[i,k] = 10.0
            word_variational[i,:] = word_variational[i,:] / word_variational[i,:].sum()
      
        for d in xrange(nDocs):
            start,stop = DOC_ID[d,:]
            doc_variational[d,:] = np.sum(word_variational[start:stop,:],axis=0)
        
    LP = dict(doc_variational=doc_variational, word_variational=word_variational)
    SS = hmodel.get_global_suff_stats(Data, LP)
    hmodel.update_global_params(SS)
