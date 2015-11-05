import os
import numpy as np

def loadDataFromSavedTask(taskoutpath, **kwargs):
    ''' Load data object used for training a specified saved run.

    Args
    ----
    taskoutpath : full path to saved results of bnpy training run

    Returns
    -------
    Dslice : bnpy Data object
        used for training the specified task.

    Example
    -------
    >>> import bnpy
    >>> os.environ['BNPYOUTDIR'] = '/tmp/'
    >>> hmodel, Info = bnpy.run(
    ...     'AsteriskK8', 'FiniteMixtureModel', 'Gauss', 'VB',
    ...     nLap=1, nObsTotal=144, K=10,
    ...     doWriteStdOut=False)
    >>> outputdir = Info['outputdir']
    >>> Data2 = loadDataFromSavedTask(outputdir)
    >>> print Data2.nObsTotal
    144
    >>> np.allclose(Info['Data'].X, Data2.X)
    True
    '''
    dataName = getDataNameFromTaskpath(taskoutpath)
    dataKwargs = loadDataKwargsFromDisk(taskoutpath)
    datamod = __import__(dataName, fromlist=[])
    Data = datamod.get_data(**dataKwargs)
    return Data

def loadDataKwargsFromDisk(taskoutpath):
    ''' Load keyword options used to load specific dataset.

    Returns
    -------
    dataKwargs : dict with options for loading dataset 
    '''
    txtfilepath = os.path.join(taskoutpath, 'args-DatasetPrefs.txt')

    dataKwargs = dict()
    with open(txtfilepath, 'r') as f:
        for line in f.readlines():
            fields = line.strip().split(' ')
            assert len(fields) == 2
            try:
                val = int(fields[1])
            except Exception:
                val = str(fields[1])
            dataKwargs[fields[0]] = val
    return dataKwargs


def loadLPKwargsFromDisk(taskoutpath):
    ''' Load keyword options used to load specific dataset.

    Returns
    -------
    dataKwargs : dict with options for loading dataset 
    '''
    from bnpy.ioutil.BNPYArgParser import algChoices
    for algName in algChoices:
        txtfilepath = os.path.join(taskoutpath, 'args-%s.txt' % (algName))
        if os.path.exists(txtfilepath):
            break

    if not os.path.exists(txtfilepath):
        raise ValueError("Cannot find alg preferences for task:\n %s" % (
            taskoutpath))

    LPkwargs = dict()
    with open(txtfilepath, 'r') as f:
        for line in f.readlines():
            fields = line.strip().split(' ')
            assert len(fields) == 2
            try:
                val = int(fields[1])
            except Exception:
                val = str(fields[1])
            LPkwargs[fields[0]] = val
    return LPkwargs

def getDataNameFromTaskpath(taskoutpath):
    ''' Extract dataset name from bnpy output filepath.

    Returns
    -------
    dataName : string identifier of a dataset

    Examples
    --------
    >>> os.environ['BNPYOUTDIR'] = '/tmp/'
    >>> taskoutpath = '/tmp/MyDataName/myjobname/1/'
    >>> dataName = getDataNameFromTaskpath(taskoutpath)
    >>> print dataName
    MyDataName
    '''
    # Make it a proper absolute path
    taskoutpath = os.path.abspath(taskoutpath)
    # Extract the dataset name from taskoutpath
    strippedpath = taskoutpath.replace(os.environ['BNPYOUTDIR'], '')
    if strippedpath.startswith(os.path.sep):
        strippedpath = strippedpath[1:]
    # The very next segment must be the data name
    dataName = strippedpath[:strippedpath.index(os.path.sep)]
    print '>>>', strippedpath
    print '>>>', dataName
    return dataName
