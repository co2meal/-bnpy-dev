'''
SeqXData.py

Subclass of XData used for holding multiple sequences of real data.
'''


import numpy as np
from .DataObj import DataObj
from bnpy.data import XData

class SeqXData(XData):
    def __init__(self, X, seqInds, nObsTotal = None, TrueZ = None):
        ''' X must be a matrix containing all data from all sequences.
            seqIndicies gives the starting indicies of each sequence in X (e.g.
              seqIndicies = [0, 9, 12] means that X[0:8,:] is the first sequence and
              X[9:12,:] is the second sequence).
            TrueZ is also indexed by seqIndicies
            nObsTotal represents the size of the overall dataset that this data
              came from.  If set to None, it will be set to the size of X
        '''
        #super(SeqXData, self).__init__(X=X, nObsTotal=nObsTotal, TrueZ=TrueZ)
        self.seqInds = seqInds
        self.nSeqs = np.size(seqInds) - 1

        X = np.asarray(X)
        if X.ndim < 2:
            X = X[np.newaxis,:]
        self.X = np.float64(X.newbyteorder('=').copy())
        
        self.set_dependent_params(nObsTotal=nObsTotal)
        self.check_dims()
        if TrueZ is not None:
            self.addTrueLabels(TrueZ)

    def to_minibatch_iterator(self, **kwargs):
        return MinibatchIterator(self, **kwargs)

      
  #########################################################  DataObj operations
  ######################################################### 


    def select_subset_by_mask(self, mask, doTrackFullSize = True):
        ''' Creates a new SeqXData object with data from the sequences specified
              by mask
            If doTrackFullSize = True, then the new object will have its
              nObsTotal attribute set to the same as the full dataset
         '''
        xNew = None
        seqNew = np.array([0])
    
        for i in xrange(len(mask)):
            start = self.seqInds[mask[i]]
            end = self.seqInds[mask[i] + 1]
            if xNew is None:
                xNew = self.X[start:end]
            else:
                xNew = np.vstack((xNew, self.X[start:end]))
            seqNew = np.append(seqNew, end - start + seqNew[i])
                
        if doTrackFullSize:
            return SeqXData(xNew, seqNew, nObsTotal=self.nObsTotal)
        return SeqXData(xNew, seqNew)

    def add_data(self, seqXDataObj):
        if not self.dim == SeqXDataObj.dim:
            raise ValueError("Dimensions of current and added data must mastch")
        self.nObs += seqXDataObj.nObs
        self.nObsTotal += seqXDataObj.nObsTotal
        self.X = np.vstack((self.X, seqXDataObj.X))
        self.seqInds = \
            np.vstack((self.seqInds, seqXDataObj.seqInds))
        self.nSeqs += seqXDataObj.nSeqs
    
        if TrueZ is not None:
            self.TrueLabels = \
                np.vstack((self.TrueLabels, seqXDataObj.TrueLabels))

    def get_random_sample(self, nObs, randState = np.random):
        ''' Selects a random sample of sequences to return; the individual
              sequences will *not* be broken up.
        '''
        nSeqs = np.minimum(np.size(self.seqInds), nObs)
        mask = randstate.permutation(self.nSeqs)[:nSeqs]
        return self.select_subset_by_mask(mask, doTrackFullSize = False)

    def generate_batch_ids(self, nBatch, dataorderseed, sizeBatch):
        prng = np.random.RandomState(dataorderseed)
        seqIDs = prng.permutation(self.nSeqs).tolist()
        seqIDByBatch = dict()
        for batchID in xrange(nBatch-1):
            seqIDByBatch[batchID] = seqIDs[:sizeBatch]
            del seqIDs[:sizeBatch]
        seqIDByBatch[nBatch-1] = seqIDs
        
        return seqIDByBatch
        


  #########################################################  I/O methods
  ######################################################### 

    def summarize_num_observations(self):
        return '  num obs: %d' % np.shape(self.X)[0]
