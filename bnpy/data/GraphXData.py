'''
GraphXData.py

Data object for holding observations of edges of a graph / network.

Attributes
-------
X : 2D array, size nObs x D
sourceID : 1D array, size nObs, holding the ID of the source node of edge i
destID : 1D array, size nObs, holding the ID of the destination node of edge i
nObs : int total number of unique observations in the current, in-memory batch
TrueParams : (optional) dict
summary : (optional) string providing human-readable description of this data

edgeRange : Tuple (start,end), indicating which edges this dataset has from
             the list of N^2-N edges (1,2),(1,3),...,(2,1), etc.  Used in
             making minibatches of a larger dataset

'''

import numpy as np
import scipy.io

from XData import XData


class GraphXData(XData):

    @classmethod
    def LoadFromFile(
            cls, filepath, nNodesTotal=None, isSparse=False, **kwargs):
        ''' Static constructor for loading data from disk into XData instance
        '''
        if filepath.endswith('.mat'):
            return cls.read_from_mat(filepath, **kwargs)
        elif filepath.endswith('.txt'):
            return cls.read_from_txt(filepath, isSparse=isSparse,
                                     nNodesTotal=nNodesTotal)
        raise NotImplemented('File extension not supported')

    @classmethod
    def read_from_txt(
            cls, filepath, isSparse=False, nNodesTotal=None, **kwargs):
        txt = np.loadtxt(filepath, dtype=np.int32)
        sourceID = txt[:, 0]
        destID = txt[:, 1]
        edgeSet = set(zip(sourceID, destID))
        return cls(X=None, edgeSet=edgeSet, isSparse=isSparse,
                   nNodesTotal=nNodesTotal, nNodes=nNodesTotal)

    @classmethod
    def read_from_mat(cls, matfilepath, nObsTotal=None, **kwargs):
        ''' Static constructor loading from .mat file into GraphXData instance.

        If no sourceID/destID field is given, it's assumed that 'X' is of size
        N^2-N, and gives edges in order 0->1, ..., 0->N-1, 1->0 1->2, etc.
        '''
        InDict = scipy.io.loadmat(matfilepath, **kwargs)
        if 'X' not in InDict:
            raise KeyError(
                'Stored matfile needs to have data in field named X')
        N = InDict['X'].shape[0]
        InDict['nNodesTotal'] = N
        InDict['nNodes'] = N
        InDict['sourceID'] = None
        InDict['destID'] = None
        if 'TrueZ' in InDict:
            InDict['TrueParams'] = {'Z': InDict['TrueZ']}

        return cls(**InDict)

    def __init__(self, X=None, edgeSet=None, nNodesTotal=None,
                 TrueParams=None, summary=None, isSparse=False,
                 nodes=None, nNodes=None, heldOut=None,
                 **kwargs):
        self.isSparse = isSparse
        self.TrueParams = TrueParams

        if X is None and edgeSet is None:
            ValueError(
                'Either specify the full adjacency matrix X ' +
                'or the sparse edgeSet')

        if X is not None and X.ndim < 3:
            X = np.expand_dims(X, axis=-1)

        if self.isSparse:
            self.edgeSet = edgeSet
            self.nNodesTotal = self._mapDownIDs()
            if nNodesTotal is not None:
                self.nNodesTotal = nNodesTotal
            if nNodes is None:
                self.nNodes = self.nNodesTotal
            else:
                self.nNodes = nNodes
            self.X = None
            self.dim = 1
            self.nEdges = len(edgeSet)
            self.nObs = self.nEdges

            if nodes is None:
                self.nodes = np.arange(self.nNodes)
            else:
                self.nodes = nodes
            self._create_resp_indices()

        else:  # Setup for non-sparse (e.g. real-valued) data
            self.dim = np.shape(X)[-1]
            self.Xmatrix = X
            self.X = np.reshape(X, (X.shape[0] * X.shape[1],) + (X.shape[2:]))
            self._set_dependent_params(nNodesTotal=nNodesTotal)
            if nodes is None:
                self.nodes = np.arange(self.nNodes)
            else:
                self.nodes = nodes
            self.nObs = self.X.shape[0]

        # Held out data. Only passed in if this is a subset of a larger dataset
        self.heldOut = heldOut
        if self.heldOut is not None:
            self._map_down_heldOut()
            self.heldOutSet = set(zip(self.heldOut[0], self.heldOut[1]))

    def holdOutData(self, holdOutFraction):
        np.random.seed(123)
        linksHeldOut = int(len(self.edgeSet) * holdOutFraction)
        linksInds = np.random.choice(
            np.arange(len(self.edgeSet)), replace=False,
            size=(linksHeldOut,))
        self.heldOut = self.respInds[linksInds, :].T

        # This will re-choose *some* links, but we'll still get roughly
        #  holdOutFraction of non-links (assuming the graph is sparse)
        self.heldOut = np.append(
            self.heldOut,
            np.random.randint(low=0, high=self.nNodes, size=(2, linksHeldOut)),
            axis=1)

        # Ensure no self-loops got chosen
        for i in xrange(self.heldOut.shape[1]):
            if self.heldOut[0, i] == self.heldOut[1, i]:
                while self.heldOut[0, i] == self.heldOut[1, i]:
                    self.heldOut[:, i] = np.random.randint(
                        low=0, high=self.nNodes, size=(2,))
        self.heldOutSet = set(zip(self.heldOut[0], self.heldOut[1]))

    def _map_down_heldOut(self):
        np.random.seed(123)
        heldOut = np.zeros((2, len(self.heldOut[0])), dtype=np.int32)
        for ee in xrange(self.heldOut[1].size):
            ind = np.where(self.nodes == self.heldOut[0, ee])[0][0]
            heldOut[0, ee] = ind
        heldOut[1, :] = self.heldOut[1, :]
        self.heldOut = heldOut

    def _check_dims(self):
        if self.isSparse:
            assert self.sourceID.ndim == 1
            assert self.sourceID.flags.c_contiguous
            assert self.sourceID.flags.owndata
            assert self.sourceID.flags.aligned
            assert self.sourceID.flags.writeable

            assert self.destID.ndim == 1
            assert self.destID.flags.c_contiguous
            assert self.destID.flags.owndata
            assert self.destID.flags.aligned
            assert self.destID.flags.writeable

            assert self.destID.shape[0] == self.sourceID.shape[0]
            assert self.sourceID.shape[0] == self.X.shape[0]

        super(GraphXData, self)._check_dims()

    def _set_dependent_params(self, nNodesTotal=None, nObsTotal=None):
        if not self.isSparse:
            self.nNodes = self.Xmatrix.shape[0]
        if nNodesTotal is None:
            self.nNodesTotal = self.nNodes
        else:
            self.nNodesTotal = nNodesTotal

    def get_stats_summary(self):
        ''' Returns human-readable summary of this dataset's basic properties
        '''
        if self.isSparse:
            s = 'Graph with N = %d nodes and %d edges\n' % (
                self.nNodes, self.nObs)
        else:
            s = 'Graph with N = %d nodes and real-valued edges of ' % (
                self.nNodes)
        s += ' dimension: %d' % (self.get_dim())
        return s

    def get_total_size(self):
        return self.nNodesTotal

    def get_size(self):
        return self.nNodes

    def get_dataset_scale(self):
        return self.nNodesTotal**2 - self.nNodesTotal

    def _create_resp_indices(self):
        respInds = np.zeros((len(self.edgeSet), 2))
        for ee, e in enumerate(self.edgeSet):
            respInd = np.where(self.nodes == e[0])[0][0]
            respInds[ee, 0] = respInd
            respInds[ee, 1] = e[1]
        self.respInds = respInds.astype(int)

    def _mapDownIDs(self):
        ''' Maps node IDs in edgeSet to continuous range 0, ... N.

        For sparse data.
        '''
        edges = np.asarray([[e[0], e[1]] for e in self.edgeSet])
        sourceID = edges[:, 0]
        destID = edges[:, 1]
        if len(sourceID) == np.max(sourceID) + 1:
            return

        uniqueS = np.unique(sourceID)
        indMap = dict(zip(uniqueS, np.arange(len(uniqueS))))
        uniqueD = np.unique(destID)
        N = len(uniqueS)

        for node in uniqueD:
            if node not in indMap:
                indMap[node] = N
                N += 1

        self.indMap = indMap
        sourceID = np.asarray([indMap[node] for node in sourceID])
        destID = np.asarray([indMap[node] for node in destID])
        self.edgeSet = set(zip(sourceID, destID))
        return max(max(sourceID), max(destID)) + 1

    def makeAssortativeMask(self, delta):
        self.assortativeMask = np.logical_and(self.Xmatrix <= delta,
                                              self.Xmatrix >= -delta)
        self.assortativeMask = self.assortativeMask.squeeze()
        self.raveledMask = np.ravel(self.assortativeMask)

    def select_subset_by_mask(self, mask, doTrackFullSize=True):
        ''' Creates new XData object by selecting certain rows (observations)

        If doTrackFullSize is True,
            ensure nObsTotal and nNodesTotal attributes are the same
            as the full dataset.
        '''
        if self.isSparse:
            return self._select_subset_by_mask_sparse(mask, doTrackFullSize)
        else:
            return self._select_subset_by_mask_nonsparse(mask, doTrackFullSize)

    def _select_subset_by_mask_nonsparse(self, mask, doTrackFullSize=True):
        # TODO : WORK WITH HELD OUT DATA
        N = self.nNodes
        mask = np.asarray(mask)
        X = self.Xmatrix[mask, :]

        if doTrackFullSize:
            return GraphXData(X=X, sourceID=None, destID=None,
                              nNodesTotal=self.nNodesTotal, nNodes=len(mask),
                              isSparse=False, nodes=mask)

        return GraphXData(X=X, nNodes=len(mask),
                          isSparse=False, nodes=mask)

    def _select_subset_by_mask_sparse(self, mask, doTrackFullSize=True):
        N = self.nNodes
        mask = np.asarray(mask)
        edgeRange = (np.min(mask), np.max(mask))

        edgeSet = set()
        if self.heldOut is not None:
            heldOut = np.asarray([[], []])
        else:
            heldOut = None
        for i in mask:
            edgeRange = (i * N, i * N + N - 1)
            for j in xrange(self.nNodes):
                if (i, j) in self.edgeSet:
                    edgeSet.add((i, j))
                if (self.heldOut is not None) and (i, j) in self.heldOutSet:
                    heldOut = np.append(
                        heldOut,
                        np.asarray([[i, j]], dtype=np.int32).T,
                        axis=1)

        if doTrackFullSize:
            return GraphXData(X=None, edgeSet=edgeSet,
                              nNodesTotal=self.nNodesTotal, nNodes=len(mask),
                              nodes=mask, isSparse=True, heldOut=heldOut)
        return GraphXData(X=None, edgeSet=edgeSet,
                          nodes=mask, edgeRange=edgeRange, isSparse=True,
                          nNodes=len(mask), heldOut=heldOut)

    def get_random_sample(self, nObs, randstate=np.random):
        nObs = np.minimum(nObs, self.nObs)
        mask = randstate.permutation(self.nObs)[:nObs]
        Data = self.select_subset_by_mask(mask, doTrackFullSize=False)
        return Data

    def add_data(self, GraphXDataObj):
        ''' Updates (in-place) this object by adding new data
        '''
        super(GraphXData, self).add_data(GraphXDataObj)

        self.nNodes += GraphXDataObj.nNodes
        self.nNodesTotal += GraphXDataObj.nNodesTotal
        self.edgeSet = set.union(GraphXDataObj.edgeSet)

    def __str__(self):
        return super(GraphXData, self).__str__() + \
            '\n edgeSet:' + self.edgeSet.__str__()
