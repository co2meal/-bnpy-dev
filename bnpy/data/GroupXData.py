'''
Classes
-----
GroupXData
    Data object for holding a dense matrix X of real 64-bit floats,
    organized contiguously based on provided group structure.
'''

import numpy as np
from XData import XData
from bnpy.util import as1D, as2D, as3D, toCArray


class GroupXData(XData):
    """ Dataset object for dense real vectors organized in groups.

    GroupXData can represent situations like:
    * obseved image patches, across many images
        group=image, observation=patch
    * observed test results for patients, across many hospitals
        group=hospital, obsevation=patient test result

    Attributes
    ------
    X : 2D array, size N x D
        each row is a single dense observation vector
    Xprev : 2D array, size N x D, optional
        "previous" observations for auto-regressive likelihoods
    D : int
        the dimension of each observation
    nObs : int
        the number of in-memory observations for this instance
    nObsTotal : int
        the total size of the dataset which in-memory X is a part of.
    TrueParams : dict
        key/value pairs represent names and arrays of true parameters

    Example
    --------
    # Create 1000 observations, each one a 3D vector
    >>> X = np.random.randn(1000, 3)

    # Assign items 0-499 to doc 1, 500-1000 to doc 2
    >>> doc_range = [0, 500, 1000]
    >>> myData = GroupXData(X, doc_range)
    >>> print myData.nObs
    1000
    >>> print myData.X.shape
    (1000, 3)
    >>> print myData.nDoc
    2
    """
    @classmethod
    def LoadFromFile(cls, filepath, nDocTotal=None, **kwargs):
        ''' Constructor for loading data from disk into XData instance
        '''
        if filepath.endswith('.mat'):
            return cls.read_from_mat(filepath, nDocTotal=nDocTotal, **kwargs)
        raise NotImplemented('Only .mat file supported.')

    def save_to_mat(self, matfilepath):
        ''' Save contents of current object to disk
        '''
        import scipy.io
        SaveVars = dict(X=self.X, nDoc=self.nDoc, doc_range=self.doc_range)
        if hasattr(self, 'Xprev'):
            SaveVars['Xprev'] = self.Xprev
        if hasattr(self, 'TrueParams') and 'Z' in self.TrueParams:
            SaveVars['TrueZ'] = self.TrueParams['Z']
        scipy.io.savemat(matfilepath, SaveVars, oned_as='row')

    @classmethod
    def read_from_mat(cls, matfilepath, nDocTotal=None, **kwargs):
        ''' Constructor for building an instance of GroupXData from disk
        '''
        import scipy.io
        InDict = scipy.io.loadmat(matfilepath, **kwargs)
        if 'X' not in InDict:
            raise KeyError(
                'Stored matfile needs to have data in field named X')
        if 'doc_range' not in InDict:
            raise KeyError(
                'Stored matfile needs to have field named doc_range')
        if nDocTotal is not None:
            InDict['nDocTotal'] = nDocTotal
        return cls(**InDict)

    def __init__(self, X=None, doc_range=None, nDocTotal=None,
                 Xprev=None, TrueZ=None,
                 TrueParams=None, fileNames=None, summary=None, **kwargs):
        ''' Create an instance of GroupXData for provided array X

        Post Condition
        ---------
        self.X : 2D array, size N x D
            with standardized dtype, alignment, byteorder.
        self.Xprev : 2D array, size N x D
            with standardized dtype, alignment, byteorder.
        self.doc_range : 1D array, size nDoc+1
        '''
        self.X = as2D(toCArray(X, dtype=np.float64))
        self.doc_range = as1D(toCArray(doc_range, dtype=np.int32))
        if summary is not None:
            self.summary = summary
        if Xprev is not None:
            self.Xprev = as2D(toCArray(Xprev), dtype=np.float64)

        # Verify attributes are consistent
        self._set_dependent_params(doc_range, nDocTotal)
        self._check_dims()

        # Add optional true parameters / true hard labels
        if TrueParams is not None:
            self.TrueParams = dict()
            for key, arr in TrueParams:
                self.TrueParams[key] = toFloat64CArray(arr)

        if TrueZ is not None:
            if not hasattr(self, 'TrueParams'):
                self.TrueParams = dict()
            self.TrueParams['Z'] = as1D(toCArray(TrueZ))

        # Add optional source files for each group/sequence
        if fileNames is not None:
            if len(fileNames) > 1:
                self.fileNames = [str(x).strip()
                                  for x in np.squeeze(fileNames)]
            else:
                self.fileNames = [str(fileNames[0])]

    def _set_dependent_params(self, doc_range, nDocTotal=None):
        self.nObs = self.X.shape[0]
        self.dim = self.X.shape[1]
        self.nDoc = self.doc_range.size - 1
        if nDocTotal is None:
            self.nDocTotal = self.nDoc
        else:
            self.nDocTotal = nDocTotal

    def _check_dims(self):
        assert self.X.ndim == 2
        assert self.X.flags.c_contiguous
        assert self.X.flags.owndata
        assert self.X.flags.aligned
        assert self.X.flags.writeable

        assert self.doc_range.ndim == 1
        assert self.doc_range.size == self.nDoc + 1
        assert self.doc_range[0] == 0
        assert self.doc_range[-1] == self.nObs
        assert np.all(self.doc_range[1:] - self.doc_range[:-1] >= 1)

    def get_size(self):
        return self.nDoc

    def get_total_size(self):
        return self.nDocTotal

    def get_dim(self):
        return self.dim

    def get_text_summary(self):
        ''' Returns human-readable description of this dataset
        '''
        if hasattr(self, 'summary'):
            s = self.summary
        else:
            s = 'GroupXData'
        return s

    def get_stats_summary(self):
        ''' Returns human-readable summary of this dataset's basic properties
        '''
        s = '  size: %d units (documents)\n' % (self.get_size())
        s += '  dimension: %d' % (self.get_dim())
        return s

    def toXData(self):
        ''' Return simplified XData instance, losing group structure
        '''
        if hasattr(self, 'Xprev'):
            return XData(self.X, Xprev=self.Xprev)
        else:
            return XData(self.X)

    # Create Subset
    #########################################################
    def select_subset_by_mask(self, docMask=None,
                              atomMask=None,
                              doTrackFullSize=True):
        """ Get subset of this dataset identified by provided unit IDs.

        Parameters
        -------
        docMask : 1D array_like of ints
            Identifies units (documents) to use to build subset.
        doTrackFullSize : boolean, optional
            default=True
            If True, return DataObj with same nObsTotal value as this
            dataset. If False, returned DataObj has smaller size.
        atomMask : 1D array_like of ints, optional
            default=None
            If present, identifies rows of X to return as XData

        Returns
        -------
        Dchunk : bnpy.data.GroupXData instance
        """

        if atomMask is not None:
            return self.toXData().select_subset_by_mask(atomMask)

        if len(docMask) < 1:
            raise ValueError('Cannot select empty subset')

        newXList = list()
        newXPrevList = list()
        newDocRange = np.zeros(len(docMask) + 1)
        newPos = 1
        for d in xrange(len(docMask)):
            start = self.doc_range[docMask[d]]
            stop = self.doc_range[docMask[d] + 1]
            newXList.append(self.X[start:stop])
            if hasattr(self, 'Xprev'):
                newXPrevList.append(self.Xprev[start:stop])
            newDocRange[newPos] = newDocRange[newPos - 1] + stop - start
            newPos += 1
        newX = np.vstack(newXList)
        if hasattr(self, 'Xprev'):
            newXprev = np.vstack(newXPrevList)
        else:
            newXprev = None
        if doTrackFullSize:
            nDocTotal = self.nDocTotal
        else:
            nDocTotal = None
        return GroupXData(newX, newDocRange,
                          Xprev=newXprev,
                          nDocTotal=nDocTotal)

    def add_data(self, XDataObj):
        """ Appends (in-place) provided dataset to this dataset.

        Post Condition
        -------
        self.Data grows by adding all units from provided DataObj.
        """
        if not self.dim == XDataObj.dim:
            raise ValueError("Dimensions must match!")
        self.nObs += XDataObj.nObs
        self.nDocTotal += XDataObj.nDocTotal
        self.nDoc += XDataObj.nDoc
        self.X = np.vstack([self.X, XDataObj.X])
        if hasattr(self, 'Xprev'):
            self.Xprev = np.vstack([self.Xprev, XDataObj.Xprev])

        new_doc_range = XDataObj.doc_range[1:] + self.doc_range[-1]
        self.doc_range = np.hstack([self.doc_range, new_doc_range])
        self._check_dims()

    def get_random_sample(self, nDoc, randstate=np.random):
        nDoc = np.minimum(nDoc, self.nDoc)
        mask = randstate.permutation(self.nDoc)[:nDoc]
        Data = self.select_subset_by_mask(mask, doTrackFullSize=False)
        return Data

    def __str__(self):
        return self.X.__str__()
