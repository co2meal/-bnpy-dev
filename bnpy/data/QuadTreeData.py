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
    def read_from_mat(cls, matfilepath, nTrees=None, tree_delims=None, **kwargs):
        import scipy.io
        InDict = scipy.io.loadmat( matfilepath, **kwargs)
        if 'X' not in InDict:
            raise KeyError('Stored mat file needs to have data in field named X')
        return cls( InDict['X'], nTrees )


    def __init__(self, X, nTrees=None, tree_delims=None,TrueZ=None):
        '''
        Create an instance of QuadTreeData given an array (X) of observations
        nTrees: Number of trees in the collection
        tree_delims: A list that contains the end index for each tree in the collection
        TrueZ: True labels for each observation
        '''
        X = np.asarray(X)
        self.X = np.float64(X.newbyteorder('=').copy())
        self.set_dependent_params(nTrees, tree_delims)
        if TrueZ is not None:
            self.add_true_labels(TrueZ)
        self.set_mask(4)
        
        
    def set_dependent_params(self, nTrees, tree_delims):
        self.nObs = self.X.shape[0]
        self.dim = self.X.shape[1]
        self.tree_delims = tree_delims
        if nTrees is None:
            if tree_delims is None:
                nTrees = 1
            else:
                self.nTrees = len(self.tree_delims)
        else:
            self.nTrees = nTrees
        
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
        self.nTrees += TreeDataObj.nTrees
        self.X = np.vstack([self.X, TreeDataObj.X])
        self.TrueLabels = np.vstack([self.TrueLabels, TreeDataObj.TrueLabels])
        
    def get_child_indices(self, n):
        '''
        Given an index, get that specific node's children (indices)
        '''
        end_of_file = 0
        for i in xrange(len(self.tree_delims)):
            if self.tree_delims[i] > n:
                end_of_file = self.tree_delims[i]
                break
        if end_of_file == 0:
            return None
        elif 4*n+1 > end_of_file:
            return None
        else:
            myList = [4*n+j+1 for j in range(4)]
            return myList
        
        
    def get_single_tree(self, n):
        '''
        get the nth tree in the collection
        '''
        if n < 0 or n > len(self.tree_delims)-1:
            return None
        elif n == 0:
            begin = 0
        else:
            begin = self.tree_delims[n-1]
        end = self.tree_delims[n]+1
        l = list()
        l.append(end-1)
        if self.TrueLabels is None:
            return QuadTreeData(X=self.X[begin:end], nTrees=1, tree_delims=l)
        else: 
            return QuadTreeData(X=self.X[begin:end], nTrees=1, tree_delims=l, TrueZ=self.TrueLabels[begin:end])
        
    def to_minibatch_iterator(self, **kwargs):
        return MinibatchIterator(self, **kwargs)
        
    def select_subset_by_mask(self, mask, doTrackFullSize=True):
        if doTrackFullSize:
            return QuadTreeData(self.X[mask], nTrees=self.nTrees)
        return QuadTreeData(self.X[mask])

    def get_parent_index(self, n):
        '''
        Given an index, get the parent's index of that specific node
        '''
        if n == 0 or n-1 in self.tree_delims:
            return None #it is a root
        elif n%4 == 0:
            return (n-1)/4
        else:
            return n/4

    def set_mask(self, nBranches):
        self.mask = np.empty((nBranches, (self.nObs-1)/4))
        for b in xrange(nBranches):
            self.mask[b,:] = [i for i in xrange(b+1, self.nObs, nBranches)]
