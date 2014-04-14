'''
QuadTreeData.py

Data object for holding quad-tree structured data in a matrix
A node at nth row has children in rows 4n+1, 4n+2, 4n+3 and
4n+4.

Example
--------
>> import numpy as np
>> from bnpy.data import QuadTreeData
>> Tree = np.random.randn(85, 2) # Create a tree with 64 leaf nodes
>> myData = QuadTreeData(Tree)
>> print myData.nObs
85
>> print myData.dim
2
>> print myData.Tree.shape
(1000,3)
'''

import numpy as np
from .TreeData import TreeData
from .DataObj import DataObj
from .MinibatchIterator import MinibatchIterator

class QuadTreeData(TreeData, DataObj):
    
    @classmethod
    def read_from_mat(cls, matfilepath, nObsTotal=None, nCollectionTotal=None, **kwargs):
        import scipy.io
        InDict = scipy.io.loadmat( matfilepath, **kwargs)
        if 'Tree' not in InDict:
            raise KeyError('Stored mat file needs to have data in field named Tree')
        return cls( InDict['Tree'], nObsTotal )


    def __init__(self, Tree, nObsTotal=None, nCollectionTotal=None,TrueZ=None):
        '''
        Create an instance of QuadTreeData given an array (Tree) of observations
        nObsTotal: Number of total observations
        nCollectionTotal: A list that contains the end index for each tree in the collection
        TrueZ: True labels for each observation
        '''
        Tree = np.asarray(Tree)
        self.Tree = np.float64(Tree.newbyteorder('=').copy())
        self.set_dependent_parameters(nObsTotal, nCollectionTotal)
        if TrueZ is not None:
            self.add_true_labels(TrueZ)
        
        
    def set_dependent_parameters(self, nObsTotal, nCollectionTotal):
        self.nObs = self.Tree.shape[0]
        self.dim = self.Tree.shape[1]
        if nObsTotal is None:
            self.nObsTotal = self.nObs
        else:
            self.nObsTotal = nObsTotal
        self.nCollectionTotal = nCollectionTotal
        
    def add_true_labels(self, TrueZ):
        '''
        Set the true labels of the observations
        '''
        assert self.nObs == TrueZ.size
        self.TrueLabels = TrueZ
        
    def add_data(self, TreeDataObj):
        if not self.dim == TreeDataObj.dim:
            raise ValueError("Dimensions must match!")
        self.nObs += TreeDataObj.nObs
        self.nObsTotal += TreeDataObj.nObsTotal
        self.Tree = np.vstack([self.Tree, TreeDataObj.Tree])
        self.TrueZ = np.vstack([self.TrueLabels, TreeDataObj.TrueLabels])
        
    def get_child_indices(self, n):
        '''
        Given an index, get that specific node's children (indices)
        '''
        end_of_file = 0
        for i in xrange(len(self.nCollectionTotal)):
            if self.nCollectionTotal[i] > n:
                end_of_file = self.nCollectionTotal[i]
                break
        if end_of_file == 0:
            return None
        elif 4*n+1 > end_of_file:
            return None
        else:
            l = list()
            for j in range(4):
                l.append(4*n+j+1)
            return l
        
        
    def get_collection_data(self, n):
        '''
        get the nth tree in the collection
        '''
        if n < 0 or n > len(self.nCollectionTotal)-1:
            return None
        elif n == 0:
            begin = 0
        else:
            begin = self.nCollectionTotal[n-1]
        end = self.nCollectionTotal[n]+1
        l = list()
        l.append(end-1)
        return QuadTreeData(self.Tree[begin:end], end-begin, l, self.TrueLabels[begin:end])
        
    def to_minibatch_iterator(self, **kwargs):
        return MinibatchIterator(self, **kwargs)
        
    def select_subset_by_mask(self, mask, doTrackFullSize=True):
        if doTrackFullSize:
            return QuadTreeData(self.X[mask], nObsTotal=self.nObsTotal)
        return QuadTreeData(self.X[mask])

    def get_parent_index(self, n):
        '''
        Given an index, get the parent's index of that specific node
        '''
        if n == 0 or n-1 in self.nCollectionTotal:
            return None #it is a root
        elif n%4 == 0:
            return (n-1)/4
        else:
            return n/4