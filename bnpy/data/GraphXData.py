"""
Classes
-----
GraphXData
    Data object for holding dense observations about edges of a network/graph.
    Organized as a list of edges, each with associated observations in field X.
"""

import numpy as np
import scipy.io
from bnpy.util import as1D, as2D, as3D, toCArray
from XData import XData

class GraphXData(XData):

    ''' Dataset object for dense observations about edges in network/graph.

    Attributes
    -------
    edges : 2D array, shape nEdges x 2
        Each row gives the source and destination node of an observed edge.
    X : 2D array, shape nEdges x D
        Row e contains the vector observation associated with edge e.
    nEdges : int
        Total number of edges observed in current, in-memory batch.
        Always equal to edges.shape[0].
    nNodesTotal : int
        Total number of nodes in the dataset.
    nodes : 1D array, size nNodes
        Node IDs represented in the current in-memory batch.

    Optional Attributes
    ------------------- 
    TrueParams : dict
        Holds dataset's true parameters, including fields
        * Z :
        * w : 2D array, size K x K
            w[j,k] gives probability of edge between block j and block k

    Example
    --------
    >>> import numpy as np
    >>> from bnpy.data import GraphXData
    >>> AdjMat = np.asarray([[0, 1, 1], \
                             [0, 0, 1], \
                             [1, 0, 0]])
    >>> Data = GraphXData(AdjMat=AdjMat)
    >>> Data.nNodesTotal
    3
    >>> Data.nodes
    array([0, 1, 2])
    '''

    def __init__(self, edges=None, X=None,
                 AdjMat=None,
                 nNodesTotal=None,
                 TrueParams=None, name=None, summary=None,
                 nodes=None, nNodes=None, heldOut=None,
                 **kwargs):
        ''' Construct a GraphXData object.

        Pass either a full adjacency matrix (nNodes x nNodes x D), 
        or a list of edges and associated observations.

        Args
        -----
        edges : 2D array, shape nEdges x 2
        X : 2D array, shape nEdges x D
        AdjMat : 3D array, shape nNodes x nNodes x D
            Defines adjacency matrix of desired graph.
            Assumes D=1 if 2D array specified.

        Returns
        --------
        Data : GraphXData
        '''
        self.isSparse = False
        self.TrueParams = TrueParams

        if AdjMat is not None:
            AdjMat = np.asarray(AdjMat)
            if AdjMat.ndim == 2:
                AdjMat = AdjMat[:, :, np.newaxis]
            nNodes = AdjMat.shape[0]
            edges = makeEdgesForDenseGraphWithNNodes(nNodes)
            X = np.zeros((edges.shape[0], AdjMat.shape[-1]))
            for eid, (i,j) in enumerate(edges):
                X[eid] = AdjMat[i,j]

        if X is None or edges is None:
            ValueError(
                'Must specify adjacency matrix AdjMat, or ' + 
                'a list of edges and corresponding dense observations X')

        # Create core attributes
        self.edges = toCArray(as2D(edges), dtype=np.int32)
        self.X = toCArray(as2D(X), dtype=np.float64)

        # Verify all edges are unique (raise error otherwise)
        N = self.edges.max() + 1
        edgeAsBaseNInteger = self.edges[:,0]*N + self.edges[:,1]
        nUniqueEdges = np.unique(edgeAsBaseNInteger).size
        if nUniqueEdges < self.edges.shape[0]:
            raise ValueError("Provided edges must be unique.")

        # Discard self loops
        nonselfloopmask = self.edges[:,0] != self.edges[:,1]
        if np.sum(nonselfloopmask) < self.edges.shape[0]:
            self.edges = self.edges[nonselfloopmask].copy()
            self.X = self.X[nonselfloopmask].copy()

        self._set_size_attributes(nNodesTotal=nNodesTotal)
        self._verify_attributes()
        if nodes is None:
            self.nodes = np.arange(self.nNodes)
        else:
            self.nodes = nodes
        # TODO Held out data

    def _verify_attributes(self):
        ''' Basic runtime checks to make sure attribute dims are correct.
        '''
        assert self.edges.ndim == 2
        assert self.edges.min() >= 0
        assert self.edges.max() < self.nNodes
        nSelfLoops = np.sum(self.edges[:,0] == self.edges[:,1])
        assert nSelfLoops == 0

    def _set_size_attributes(self, nNodesTotal=None):
        ''' Set internal fields that define sizes/dims.

        Post condition
        --------------
        Fields nNodes and nNodes total have proper, consistent values.
        '''
        if nNodesTotal is None:
            self.nNodesTotal = self.edges.max() + 1
        else:
            self.nNodesTotal = nNodesTotal
        self.nNodes = self.nNodesTotal
        self.nEdges = self.edges.shape[0]
        self.dim = self.X.shape[1]


    def get_stats_summary(self):
        ''' Returns human-readable summary of this dataset's basic properties
        '''
        s = 'Graph with %d nodes, %d edges and %d-dimensional observations' % (
            self.nNodesTotal, self.nEdges, self.dim)
        return s

    def get_total_size(self):
        return self.nNodesTotal

    def get_size(self):
        return self.nNodes

    def get_dataset_scale(self):
        return self.nNodesTotal**2 - self.nNodesTotal

    def select_subset_by_mask(self, mask, doTrackFullSize=True):
        ''' Creates new GraphXData object using a subset of edges.

        Args
        ----
        mask : 1D array_like
            Contains integer ids of edges to keep.
        doTrackFullSize : boolean
            if True, retain nObsTotal and nNodesTotal attributes.

        Returns
        -------
        subsetData : GraphXData object
        '''
        mask = np.asarray(mask, dtype=np.int32)
        edges = self.edges[mask]
        X = self.X[mask]

        if doTrackFullSize:
            nNodesTotal = self.nNodesTotal
            nodes = self.nodes
        else:
            nNodesTotal = None
            nodes = None
        return GraphXData(edges=edges, X=X, 
                          nNodesTotal=nNodesTotal, nodes=nodes)

    def add_data(self, otherDataObj):
        ''' Updates (in-place) this object by adding new nodes.
        '''
        nodes = np.union1d(self.nodes, otherDataObj.nodes)
        self.X = np.vstack([self.X, otherDataObj.X])
        self.edges = np.vstack([self.edges, otherDataObj.edges])
        self._set_size_attributes(
            nNodesTotal=self.nNodesTotal+otherDataObj.nNodesTotal)
        self.nodes = nodes

    def toAdjacencyMatrix(self):
        ''' Return adjacency matrix representation of this dataset.

        Returns
        -------
        AdjMat : 3D array, nNodes x nNodes x D
        '''
        AdjMat = np.zeros((self.nNodes, self.nNodes, self.dim))
        AdjMat[self.edges[:,0], self.edges[:,1]] = self.X
        return AdjMat

    @classmethod
    def LoadFromFile(cls, filepath,
                     nNodesTotal=None, isSparse=False, **kwargs):
        ''' Static constructor for loading data from disk into XData instance
        '''
        if filepath.endswith('.mat'):
            return cls.read_from_mat(filepath, **kwargs)
        elif filepath.endswith('.txt'):
            return cls.read_from_txt(filepath, isSparse=isSparse,
                                     nNodesTotal=nNodesTotal)
        raise NotImplemented('File extension not supported')

    @classmethod
    def read_from_txt(cls, filepath,
                      nNodesTotal=None, isSparse=False, **kwargs):
        ''' Static constructor loading .txt file into GraphXData instance.
        '''
        txt = np.loadtxt(filepath, dtype=np.int32)
        sourceID = txt[:, 0]
        destID = txt[:, 1]
        edgeSet = set(zip(sourceID, destID))
        return cls(X=None, edgeSet=edgeSet, isSparse=isSparse,
                   nNodesTotal=nNodesTotal, nNodes=nNodesTotal)

    @classmethod
    def read_from_mat(cls, matfilepath, nObsTotal=None, **kwargs):
        ''' Static constructor loading .mat file into GraphXData instance.
        '''
        InDict = scipy.io.loadmat(matfilepath, **kwargs)
        if 'X' not in InDict:
            raise KeyError(
                'Stored matfile needs to have data in field named X')
        N = InDict['X'].shape[0]
        InDict['nNodesTotal'] = N
        InDict['nNodes'] = N
        if 'TrueZ' in InDict:
            InDict['TrueParams'] = {'Z': InDict['TrueZ']}
        return cls(**InDict)


def makeEdgesForDenseGraphWithNNodes(N):
    ''' Make edges array for a directed graph with N nodes.
      
    Returns
    --------
    edges : 2D array, shape nEdges x 2
        contains all non-self-loop edges
    '''
    edges = list()
    for s in xrange(N):
        for t in xrange(N):
            if s == t:
                continue 
            edges.append((s,t))
    return np.asarray(edges, dtype=np.int32)
