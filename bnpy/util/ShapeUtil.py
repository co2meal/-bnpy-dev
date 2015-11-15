import numpy as np


def argsort_bigtosmall_stable(x):
    ''' Sort indices of vector x so the values in x ranked big to small

    Sort guaranteed to be stable, meaning any adjacent pairs of x 
    that are already sorted will not be permuted.

    Returns
    -------
    sortids : 1D array
        x[sortids[i]] >= x[sortids[i+1]] for all i.
    
    Examples
    --------
    >>> xs = np.asarray([3, 4, 5, 2, 2, 2, 1])
    >>> print argsort_bigtosmall_stable(xs)
    [2 1 0 3 4 5 6]
    '''
    x = as1D(x)
    if x.ndim != 1:
        raise ValueError(
            "1D input array required. Instead found %d dims" % (x.ndim))
    return np.argsort(-1 * x, kind='mergesort')

def is_sorted_bigtosmall(xvec):
    ''' Returns True if provided 1D array is sorted largest to smallest.

    Returns
    -------
    isSorted : boolean

    Examples
    --------
    >>> is_sorted_bigtosmall([3, 3, 3])
    True
    >>> is_sorted_bigtosmall([3, 2, 1])
    True
    >>> is_sorted_bigtosmall([3, 3, 3, 2, 2])
    True
    >>> is_sorted_bigtosmall([1.0, 2.0, 3.0])
    False
    >>> is_sorted_bigtosmall([5, 4, 3, 2, 2, 1, 1, 0.5])
    True
    '''
    uvals = np.unique(np.sign(np.diff(as1D(xvec))))
    if uvals.size == 2:
        # Look for -1 or 0. 0 means entries are equal.
        return np.allclose(uvals, [-1.0, 0.0])
    elif uvals.size == 1:
        return uvals[0] == -1.0 or uvals[0] == 0.0
    else:
        # Failure case
        return False

def toCArray(X, dtype=np.float64):
    """ Convert input into numpy array of C-contiguous order.

    Ensures returned array is aligned and owns its own data,
    not a view of another array.

    Returns
    -------
    X : ND array

    Examples
    -------
    >>> Q = np.zeros(10, dtype=np.int32, order='F')
    >>> toCArray(Q).flags.c_contiguous
    True
    >>> toCArray(Q).dtype.byteorder
    '='
    """
    X = np.asarray_chkfinite(X, dtype=dtype, order='C')
    if X.dtype.byteorder != '=':
        X = X.newbyteorder('=').copy()
    if not X.flags.owndata or X.flags.aligned:
        X = X.copy()
    assert X.flags.owndata
    assert X.flags.aligned
    return X

def as1D(x):
    """ Convert input into to 1D numpy array.

    Returns
    -------
    x : 1D array

    Examples
    -------
    >>> as1D(5)
    array([5])
    >>> as1D([1,2,3])
    array([1, 2, 3])
    >>> as1D([[3,4,5,6]])
    array([3, 4, 5, 6])
    """
    if not isinstance(x, np.ndarray):
        x = np.asarray_chkfinite(x)
    if x.ndim < 1:
        x = np.asarray_chkfinite([x])
    elif x.ndim > 1:
        x = np.squeeze(x)
    return x


def as2D(x):
    """ Convert input into to 2D numpy array.


    Returns
    -------
    x : 2D array

    Examples
    -------
    >>> as2D(5)
    array([[5]])
    >>> as2D([1,2,3])
    array([[1, 2, 3]])
    >>> as2D([[3,4,5,6]])
    array([[3, 4, 5, 6]])
    """
    if not isinstance(x, np.ndarray):
        x = np.asarray_chkfinite(x)
    if x.ndim < 1:
        x = np.asarray_chkfinite([x])
    while x.ndim < 2:
        x = x[np.newaxis, :]
    return x


def as3D(x):
    """ Convert input into to 3D numpy array.

    Returns
    -------
    x : 3D array

    Examples
    -------
    >>> as3D(5)
    array([[[5]]])
    >>> as3D([1,2,3])
    array([[[1, 2, 3]]])
    >>> as3D([[3,4,5,6]])
    array([[[3, 4, 5, 6]]])
    """
    if not isinstance(x, np.ndarray):
        x = np.asarray_chkfinite(x)
    if x.ndim < 1:
        x = np.asarray_chkfinite([x])
    while x.ndim < 3:
        x = x[np.newaxis, :]
    return x
