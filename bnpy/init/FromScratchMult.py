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
        print 'Initializing a completely random Admixture Model'

        doc_variational = np.zeros( (nDocTotal, K) )
        
        word_variational = PRNG.rand(nObsTotal, K)
        word_variational /= word_variational.sum(axis=1)[:,np.newaxis]
        
        for d in xrange(nDocTotal):
            start,stop = doc_range[d,:]
            doc_variational[d,:] = np.sum(word_variational[start:stop,:],axis=0)
            
        print 'Finished Random Initialization'
            
    elif initname == 'truth':
        word_variational = np.zeros( (nObsTotal, K) ) + .1
        doc_variational = Data.true_td.T
        
        # set each word-token variational parameter 
        #  to its true document x topic weights 
        for d in xrange(nDocTotal):
            start,stop = doc_range[d,:]
            word_variational[start:stop,:] = doc_variational[d,:]
           
    if hmodel.getAllocModelName().count('HDP') > 0:
      LP = dict(E_logPi=np.log(doc_variational + 1e-20),
                 word_variational=word_variational)
    else:
      LP = dict(doc_variational=doc_variational, 
                word_variational=word_variational)

    SS = hmodel.get_global_suff_stats(Data, LP)
    hmodel.update_global_params(SS)
