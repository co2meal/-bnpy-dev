'''
ModelReader.py

Load bnpy models from disk

See Also
-------
ModelWriter.py : save bnpy models to disk.
'''
import numpy as np
import scipy.io
import os
import glob

from ModelWriter import makePrefixForLap
from bnpy.allocmodel import AllocModelConstructorsByName
from bnpy.obsmodel import ObsModelConstructorsByName
from bnpy.util import toCArray

def getPrefixForLapQuery(taskpath, lapQuery):
    ''' Search among checkpoint laps for one nearest to query.

    Returns
    --------
    prefix : str
        For lap 1, prefix = 'Lap0001.000'.
        For lap 5.5, prefix = 'Lap0005.500'.
    lap : int
        lap checkpoint for saved params closed to lapQuery
    '''
    try:
        saveLaps = np.loadtxt(os.path.join(taskpath, 'laps-saved-params.txt'))
    except IOError:
        fileList = glob.glob(os.path.join(taskpath, 'Lap*TopicModel.mat'))
        saveLaps = list()
        for fpath in sorted(fileList):
            basename = fpath.split(os.path.sep)[-1]
            lapstr = basename[3:11]
            saveLaps.append(float(lapstr))
        saveLaps = np.sort(np.asarray(saveLaps))
    if lapQuery is None:
        bestLap = saveLaps[-1]  # take final saved value
    else:
        distances = np.abs(lapQuery - saveLaps)
        bestLap = saveLaps[np.argmin(distances)]
    return makePrefixForLap(bestLap), bestLap


def loadModelForLap(matfilepath, lapQuery):
    ''' Loads saved model with lap closest to provided lapQuery.

    Returns
    -------
    model : bnpy.HModel
        Model object for saved at checkpoint lap=bestLap.
    bestLap : int
        lap checkpoint for saved model closed to lapQuery
    '''
    prefix, bestLap = getPrefixForLapQuery(matfilepath, lapQuery)
    model = load_model(matfilepath, prefix=prefix)
    return model, bestLap


def load_model(matfilepath, prefix='Best', lap=None):
    ''' Load model stored to disk by ModelWriter

    Returns
    ------
    model : bnpy.HModel
        Model object for saved at checkpoint indicated by prefix or lap.
    '''
    # Avoids circular import
    import bnpy.HModel as HModel

    if lap is not None:
        prefix, _ = getPrefixForLapQuery(matfilepath, lap)
    try:
        obsModel = load_obs_model(matfilepath, prefix)
        allocModel = load_alloc_model(matfilepath, prefix)
        model = HModel(allocModel, obsModel)
    except IOError as e:
        if prefix == 'Best':
            matList = glob.glob(os.path.join(matfilepath, '*TopicModel.mat'))
            if len(matList) < 1:
                raise e
            matList.sort()  # ascending order, so most recent is last
            prefix = matList[-1].split(os.path.sep)[-1][:11]
        model = loadTopicModel(matfilepath, prefix=prefix)
    return model


def load_alloc_model(matfilepath, prefix):
    """ Load allocmodel stored to disk in bnpy .mat format.

    Parameters
    ------
    matfilepath : str
        String file system path to folder where .mat files are stored.
        Usually this path is a "taskoutpath" like where bnpy.run
        saves its output.
    prefix : str
        Indicates which stored checkpoint to use.
        Can look like 'Lap0005.000'.

    Returns
    ------
    allocModel : bnpy.allocmodel object
        This object has valid set of global parameters
        and valid hyperparameters that define its prior.
    """
    apriorpath = os.path.join(matfilepath, 'AllocPrior.mat')
    amodelpath = os.path.join(matfilepath, prefix + 'AllocModel.mat')
    APDict = loadDictFromMatfile(apriorpath)
    ADict = loadDictFromMatfile(amodelpath)
    AllocConstr = AllocModelConstructorsByName[ADict['name']]
    amodel = AllocConstr(ADict['inferType'], APDict)
    amodel.from_dict(ADict)
    return amodel


def load_obs_model(matfilepath, prefix):
    """ Load observation model object stored to disk in bnpy mat format.

    Parameters
    ------
    matfilepath : str
        String file system path to folder where .mat files are stored.
        Usually this path is a "taskoutpath" like where bnpy.run
        saves its output.
    prefix : str
        Indicates which stored checkpoint to use.
        Can look like 'Lap0005.000'.

    Returns
    ------
    allocModel : bnpy.allocmodel object
        This object has valid set of global parameters
        and valid hyperparameters that define its prior.
    """
    obspriormatfile = os.path.join(matfilepath, 'ObsPrior.mat')
    PriorDict = loadDictFromMatfile(obspriormatfile)
    ObsConstr = ObsModelConstructorsByName[PriorDict['name']]
    obsModel = ObsConstr(**PriorDict)

    obsmodelpath = os.path.join(matfilepath, prefix + 'ObsModel.mat')
    ParamDict = loadDictFromMatfile(obsmodelpath)
    if obsModel.inferType == 'EM':
        obsModel.setEstParams(**ParamDict)
    else:
        obsModel.setPostFactors(**ParamDict)
    return obsModel


def loadDictFromMatfile(matfilepath):
    ''' Load dict of numpy arrays from a .mat-format file on disk.

    This is a wrapper around scipy.io.loadmat,
    which makes the returned numpy arrays in standard aligned format.

    Returns
    --------
    D : dict
        Each key/value pair is a parameter name and a numpy array
        loaded from the provided mat file.
        We ensure before returning that each array has properties:
            * C alignment
            * Original 2D shape has been squeezed as much as possible
                * (1,1) becomes a size=1 1D array
                * (1,N) or (N,1) become 1D arrays
            * flags.aligned is True
            * flags.owndata is True
            * dtype.byteorder is '='

    Examples
    -------
    >>> import scipy.io
    >>> Dorig = dict(scalar=5, scalar1DN1=np.asarray([3.14,]))
    >>> Dorig['arr1DN3'] = np.asarray([1,2,3])
    >>> scipy.io.savemat('Dorig.mat', Dorig, oned_as='row')
    >>> D = loadDictFromMatfile('Dorig.mat')
    >>> D['scalar']
    array(5)
    >>> D['scalar1DN1']
    array(3.14)
    >>> D['arr1DN3']
    array([1, 2, 3])
    '''
    Dtmp = scipy.io.loadmat(matfilepath)
    D = dict([x for x in Dtmp.items() if not x[0].startswith('__')])
    for key in D:
        if not isinstance(D[key], np.ndarray):
            continue
        x = D[key]
        if x.ndim == 2:
            x = np.squeeze(x)
        if str(x.dtype).count('int'):
            arr = toCArray(x, dtype=np.int32)
        else:
            arr = toCArray(x, dtype=np.float64)
        assert arr.dtype.byteorder == '='
        assert arr.flags.aligned is True
        assert arr.flags.owndata is True
        D[key] = arr
    return D


def loadWordCountMatrixForLap(matfilepath, lapQuery, toDense=True):
    ''' Load word counts
    '''
    prefix, bestLap = getPrefixForLapQuery(matfilepath, lapQuery)
    _, WordCounts = loadTopicModel(matfilepath, prefix, returnWordCounts=1)
    return WordCounts


def loadTopicModel(matfilepath, prefix=None, returnWordCounts=0, returnTPA=0):
    ''' Load saved topic model
    '''
    # avoids circular import
    from bnpy.HModel import HModel
    if prefix is not None:
        matfilepath = os.path.join(matfilepath, prefix + 'TopicModel.mat')
    Mdict = loadDictFromMatfile(matfilepath)
    if 'SparseWordCount_data' in Mdict:
        data = np.asarray(Mdict['SparseWordCount_data'], dtype=np.float64)
        K = int(Mdict['K'])
        vocab_size = int(Mdict['vocab_size'])
        try:
            indices = Mdict['SparseWordCount_indices']
            indptr = Mdict['SparseWordCount_indptr']
            WordCounts = scipy.sparse.csr_matrix((data, indices, indptr),
                                                 shape=(K, vocab_size))
        except KeyError:
            rowIDs = Mdict['SparseWordCount_i'] - 1
            colIDs = Mdict['SparseWordCount_j'] - 1
            WordCounts = scipy.sparse.csr_matrix((data, (rowIDs, colIDs)),
                                                 shape=(K, vocab_size))
        Mdict['WordCounts'] = WordCounts.toarray()
    if returnTPA:
        if 'WordCounts' in Mdict:
            topics = Mdict['WordCounts'] + Mdict['lam']
        else:
            topics = Mdict['topics']
        K = topics.shape[0]

        try:
            probs = Mdict['probs']
        except KeyError:
            probs = (1.0/K) * np.ones(K)
        try:
            alpha = float(Mdict['alpha'])
        except KeyError:
            if 'alpha' in os.environ:
                alpha = float(os.environ['alpha'])
            else:
                raise ValueError('Unknown parameter alpha')
        return topics, probs, alpha

    infAlg = 'VB'
    if 'gamma' in Mdict:
        aPriorDict = dict(alpha=Mdict['alpha'], gamma=Mdict['gamma'])
        HDPTopicModel = AllocModelConstructorsByName['HDPTopicModel']
        amodel = HDPTopicModel(infAlg, aPriorDict)
    else:
        FiniteTopicModel = AllocModelConstructorsByName['FiniteTopicModel']
        amodel = FiniteTopicModel(infAlg, dict(alpha=Mdict['alpha']))
    omodel = ObsModelConstructorsByName['Mult'](infAlg, **Mdict)
    hmodel = HModel(amodel, omodel)

    hmodel.set_global_params(**Mdict)
    if returnWordCounts:
        return hmodel, Mdict['WordCounts']
    return hmodel
