'''
FromScratchMult.py

Initialize params of HModel with multinomial observations from scratch.
'''
import numpy as np

def init_global_params(hmodel, Data, initname='randexamples', seed=0, K=0, **kwargs):
    
    nObsTotal = Data.nObsTotal
    doc_range = Data.doc_range
    nDocTotal = Data.nDocTotal
    PRNG = np.random.RandomState(seed)
    if initname == 'randexamples':
        ''' Choose K items uniformly at random from the Data
        '''
        word_variational = np.zeros( (nObsTotal, K) ) + .1 # initialize local word-level variational parameters word_variational
        doc_variational = np.zeros( (nDocTotal, K) ) # initialize local document-level variational parameters doc_variational
         
        for i in xrange(nObsTotal):
            k = np.round(PRNG.rand()*K)-1
            word_variational[i,k] = 10.0
            word_variational[i,:] = word_variational[i,:] / word_variational[i,:].sum()
      
        for d in xrange(nDocTotal):
            start,stop = doc_range[d,:]
            doc_variational[d,:] = np.sum(word_variational[start:stop,:],axis=0)
            
    elif initname == 'truth':
        word_variational = np.zeros( (nObsTotal, K) ) + .1 # initialize local word-level variational parameters word_variational
        doc_variational = Data.true_td.T# initialize local document-level variational parameters doc_variational
        
        # set each word-token variational parameter to its true document x topic weights 
        for d in xrange(nDocTotal):
            start,stop = doc_range[d,:]
            word_variational[start:stop,:] = doc_variational[d,:]
            #word_variational[i,:] = word_variational[i,:] / word_variational[i,:].sum()
        
    LP = dict(doc_variational=doc_variational, word_variational=word_variational)
    SS = hmodel.get_global_suff_stats(Data, LP)
    hmodel.update_global_params(SS)
