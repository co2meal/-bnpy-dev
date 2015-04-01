
def numpyToSharedMemArray(X):
    """ Get copy of X accessible as shared memory

    Returns
    --------
    Xsh : RawArray (same size as X)
        Uses separate storage than original array X.
    """
    Xtmp = np.ctypeslib.as_ctypes(X)
    Xsh = multiprocessing.sharedctypes.RawArray(Xtmp._type_, Xtmp)
    return Xsh

def sharedMemToNumpyArray(Xsh):
    """ Get view (not copy) of shared memory as numpy array.

    Returns
    -------
    X : ND numpy array (same size as X)
        Any changes to X will also influence data stored in Xsh.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.ctypeslib.as_array(Xsh)