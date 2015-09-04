'''
Classes
-----
XData
    Data object for holding a dense matrix X of real 64-bit floats,
    useful for Gaussian or auto-regressive Gaussian likelihods.
'''

import numpy as np
import scipy.io
from collections import namedtuple

from DataObj import DataObj
from bnpy.util import as1D, as2D, toCArray
from bnpy.util import numpyToSharedMemArray, sharedMemToNumpyArray


class XData(DataObj):

    """ Dataset object for dense vectors of real-valued observations.

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
    -------
    >>> X = np.zeros((1000, 3)) # Create 1000x3 matrix

    >>> myData = XData(X) # Convert to an XData object

    >>> print myData.nObs
    1000
    >>> print myData.dim
    3
    >>> print myData.X.shape
    (1000, 3)
    >>> mySubset = myData.select_subset_by_mask([0])

    >>> mySubset.X.shape
    (1, 3)
    >>> mySubset.X[0]
    array([ 0.,  0.,  0.])
    """

    @classmethod
    def LoadFromFile(cls, filepath, nObsTotal=None, **kwargs):
        ''' Constructor for loading data from disk into XData instance.
        '''
        if filepath.endswith('.mat'):
            return cls.read_from_mat(filepath, nObsTotal, **kwargs)
        try:
            X = np.load(filepath)
        except Exception as e:
            X = np.loadtxt(filepath)
        return cls(X, nObsTotal=nObsTotal, **kwargs)

    @classmethod
    def read_from_mat(cls, matfilepath, nObsTotal=None, **kwargs):
        ''' Constructor for loading .mat file into XData instance.
        '''
        InDict = scipy.io.loadmat(matfilepath)
        if 'X' not in InDict:
            raise KeyError(
                'Stored matfile needs to have data in field named X')
        return cls(InDict['X'], nObsTotal)

    def __init__(self, X, nObsTotal=None, TrueZ=None, Xprev=None,
                 TrueParams=None, summary=None):
        ''' Constructor for XData instance for provided array data X.

        Post Condition
        ---------
        self.X : 2D array, size N x D
            with standardized dtype, alignment, byteorder.
        self.Xprev : 2D array, size N x D
            with standardized dtype, alignment, byteorder.
        '''
        self.X = as2D(toCArray(X, dtype=np.float64))
        if Xprev is not None:
            self.Xprev = as2D(toCArray(Xprev, dtype=np.float64))

        # Verify attributes are consistent
        self._set_dependent_params(nObsTotal=nObsTotal)
        self._check_dims()

        # Add optional true parameters / true hard labels
        if TrueParams is not None:
            self.TrueParams = TrueParams
        if TrueZ is not None:
            if not hasattr(self, 'TrueParams'):
                self.TrueParams = dict()
            self.TrueParams['Z'] = as1D(toCArray(TrueZ))
            self.TrueParams['K'] = np.unique(self.TrueParams['Z']).size
        if summary is not None:
            self.summary = summary

    def _set_dependent_params(self, nObsTotal=None):
        self.nObs = self.X.shape[0]
        self.dim = self.X.shape[1]
        if nObsTotal is None:
            self.nObsTotal = self.nObs
        else:
            self.nObsTotal = nObsTotal

    def _check_dims(self):
        assert self.X.ndim == 2
        assert self.X.flags.c_contiguous
        assert self.X.flags.owndata
        assert self.X.flags.aligned
        assert self.X.flags.writeable

    def get_size(self):
        """ Get number of observations in memory for this object.

        Returns
        ------
        n : int
        """
        return self.nObs

    def get_total_size(self):
        """ Get total number of observations for this dataset.

        This may be much larger than self.nObs.

        Returns
        ------
        n : int
        """
        return self.nObsTotal

    def get_dim(self):
        return self.dim

    def get_text_summary(self):
        ''' Get human-readable description of this dataset.

        Returns
        -------
        s : string
        '''
        if hasattr(self, 'summary'):
            s = self.summary
        else:
            s = 'X Data'
        return s

    def get_stats_summary(self):
        ''' Get human-readable summary of this dataset's basic properties

        Returns
        -------
        s : string
        '''
        s = '  size: %d units (single observations)\n' % (self.get_size())
        s += '  dimension: %d' % (self.get_dim())
        return s

    def select_subset_by_mask(self, mask,
                              doTrackFullSize=True,
                              doTrackTruth=False):
        ''' Get subset of this dataset identified by provided unit IDs.

        Parameters
        -------
        mask : 1D array_like
            Identifies units (rows) of X to use for subset.
        doTrackFullSize : boolean
            If True, return DataObj with same nObsTotal value as this
            dataset. If False, returned DataObj has smaller size.

        Returns
        -------
        Dchunk : bnpy.data.XData instance
        '''
        if hasattr(self, 'Xprev'):
            newXprev = self.Xprev[mask]
        else:
            newXprev = None
        newX = self.X[mask]

        if hasattr(self, 'alwaysTrackTruth'):
            doTrackTruth = doTrackTruth or self.alwaysTrackTruth
        hasTrueZ = hasattr(self, 'TrueParams') and 'Z' in self.TrueParams
        if doTrackTruth and hasTrueZ:
            TrueZ = self.TrueParams['Z']
            newTrueZ = TrueZ[mask]
        else:
            newTrueZ = None

        if doTrackFullSize:
            nObsTotal = self.nObsTotal
        else:
            nObsTotal = None

        return XData(X=newX, Xprev=newXprev,
                     TrueZ=newTrueZ, nObsTotal=nObsTotal)

    def add_data(self, XDataObj):
        """ Appends (in-place) provided dataset to this dataset.

        Post Condition
        -------
        self.Data grows by adding all units from provided DataObj.
        """
        if not self.dim == XDataObj.dim:
            raise ValueError("Dimensions must match!")
        self.nObs += XDataObj.nObs
        self.nObsTotal += XDataObj.nObsTotal
        self.X = np.vstack([self.X, XDataObj.X])
        if hasattr(self, 'Xprev'):
            assert hasattr(XDataObj, 'Xprev')
            self.Xprev = np.vstack([self.Xprev, XDataObj.Xprev])
        self._check_dims()

    def get_random_sample(self, nObs, randstate=np.random):
        nObs = np.minimum(nObs, self.nObs)
        mask = randstate.permutation(self.nObs)[:nObs]
        Data = self.select_subset_by_mask(mask, doTrackFullSize=False)
        return Data

    def __str__(self):
        return self.X.__str__()

    def getRawDataAsSharedMemDict(self):
        ''' Create dict with copies of raw data as shared memory arrays
        '''
        dataShMemDict = dict()
        dataShMemDict['X'] = numpyToSharedMemArray(self.X)
        dataShMemDict['nObsTotal'] = self.nObsTotal
        if hasattr(self, 'Xprev'):
            dataShMemDict['Xprev'] = numpyToSharedMemArray(self.Xprev)
        return dataShMemDict

    def getDataSliceFunctionHandle(self):
        """ Return function handle that can make data slice objects.

        Useful with parallelized algorithms,
        when we need to use shared memory.

        Returns
        -------
        f : function handle
        """
        return makeDataSliceFromSharedMem


def makeDataSliceFromSharedMem(dataShMemDict,
                               cslice=(0, None),
                               batchID=None):
    """ Create data slice from provided raw arrays and slice indicators.

    Returns
    -------
    Dslice : namedtuple with same fields as XData object
        * X
        * nObs
        * nObsTotal
        * dim
    Represents subset of documents identified by cslice tuple.

    Example
    -------
    >>> Data = XData(np.random.rand(25,2))
    >>> shMemDict = Data.getRawDataAsSharedMemDict()
    >>> Dslice = makeDataSliceFromSharedMem(shMemDict)
    >>> np.allclose(Data.X, Dslice.X)
    True
    >>> np.allclose(Data.nObs, Dslice.nObs)
    True
    >>> Data.dim == Dslice.dim
    True
    >>> Aslice = makeDataSliceFromSharedMem(shMemDict, (0, 2))
    >>> Aslice.nObs
    2

    TODO: Make compatible with Xprev field in autoreg models.
    """
    if batchID is not None and batchID in dataShMemDict:
        dataShMemDict = dataShMemDict[batchID]

    # Make local views (NOT copies) to shared mem arrays
    X = sharedMemToNumpyArray(dataShMemDict['X'])
    nObsTotal = int(dataShMemDict['nObsTotal'])

    N, dim = X.shape
    if cslice is None:
        cslice = (0, N)
    elif cslice[1] is None:
        cslice = (0, N)

    keys = ['X', 'Xprev', 'nObs', 'dim', 'nObsTotal']

    if 'Xprev' in dataShMemDict:
        Xprev = sharedMemToNumpyArray(
            dataShMemDict['Xprev'][cslice[0]:cslice[1]])
    else:
        Xprev = None

    Dslice = namedtuple("XDataTuple", keys)(
        X=X[cslice[0]:cslice[1]],
        Xprev=Xprev,
        nObs=cslice[1] - cslice[0],
        dim=dim,
        nObsTotal=nObsTotal,
    )
    return Dslice
