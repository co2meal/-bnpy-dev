'''
ModelWriter.py

Save bnpy models to disk

See Also
-------
ModelReader.py : load bnpy models from disk.
'''
import numpy as np
import scipy.io
import os
from distutils.dir_util import mkpath
from shutil import copy2
from sys import platform


def makePrefixForLap(lap):
    """ Get string prefix for saving lap-specific info to disk.

    Returns
    -----
    s : str
    """
    return 'Lap%08.3f' % (lap)


def save_model(hmodel, outputdir, prefix, doSavePriorInfo=True,
               doSaveObsModel=True, doLinkBest=False):
    ''' Saves HModel object to .mat format file on disk.

    Parameters
    --------
    hmodel: bnpy.HModel
    outputdir: str
        Absolute full path of directory to save in
    prefix: str
        Prefix for file name, like 'Lap0055.000' or 'Best'
        See makePrefixForLap function.
    doSavePriorInfo: boolean
        if True, save parameters that define model prior to disk.
    doSaveObsModel : boolean
        if True, save global params that define observation model.
    doLinkBest : boolean
        if True, copy saved global parameter files under prefix 'Best'
        This allows rapid access of the most recent (best) parameters.

    Post Condition
    --------
    outputdir defines a valid directory on disk.
        If it doesn't exist at first, this directory will be created.
    File outputdir/<prefix>AllocModel.mat will be written.
    if doSaveObsModel:
        File outputdir/<prefix>ObsModel.mat will be written.
    if doSavePriorInfo:
        File outputdir/<prefix>AllocPrior.mat will be written.
        File outputdir/<prefix>ObsPrior.mat will be written.
    '''
    if not os.path.exists(outputdir):
        mkpath(outputdir)
    save_alloc_model(hmodel.allocModel, outputdir, prefix,
                     doLinkBest=doLinkBest)

    if doSaveObsModel:
        save_obs_model(hmodel.obsModel, outputdir, prefix,
                       doLinkBest=doLinkBest)

    if doSavePriorInfo:
        save_alloc_prior(hmodel.allocModel, outputdir)
        save_obs_prior(hmodel.obsModel, outputdir)


def save_alloc_model(amodel, fpath, prefix, doLinkBest=False):
    """ Save allocmodel object global parameters to disk in .mat format.

    Post Condition
    -----
    File <fpath>/<prefix>AllocModel.mat written to disk.
    """
    amatname = prefix + 'AllocModel.mat'
    outmatfile = os.path.join(fpath, amatname)
    adict = amodel.to_dict()
    adict.update(amodel.to_dict_essential())
    scipy.io.savemat(outmatfile, adict, oned_as='row')
    if doLinkBest and prefix != 'Best':
        create_best_link(outmatfile, os.path.join(fpath, 'BestAllocModel.mat'))


def save_alloc_prior(amodel, fpath):
    """ Save allocmodel object prior parameters to disk in .mat format.

    Post Condition
    -----
    File <fpath>/<prefix>AllocPrior.mat written to disk.
    """
    outpath = os.path.join(fpath, 'AllocPrior.mat')
    adict = amodel.get_prior_dict()
    if len(adict.keys()) == 0:
        return None
    scipy.io.savemat(outpath, adict, oned_as='row')


def save_obs_model(obsmodel, fpath, prefix, doLinkBest=False):
    """ Save obsmodel object global parameters to disk in .mat format.

    Post Condition
    -----
    File <fpath>/<prefix>ObsModel.mat written to disk.
    """
    amatname = prefix + 'ObsModel.mat'
    outmatfile = os.path.join(fpath, amatname)
    myDict = obsmodel.to_dict()
    scipy.io.savemat(outmatfile, myDict, oned_as='row')
    if doLinkBest and prefix != 'Best':
        create_best_link(outmatfile, os.path.join(fpath, 'BestObsModel.mat'))


def save_obs_prior(obsModel, fpath):
    """ Save obsmodel object prior parameters to disk in .mat format.

    Post Condition
    -----
    File <fpath>/<prefix>ObsPrior.mat written to disk.
    """
    outpath = os.path.join(fpath, 'ObsPrior.mat')
    adict = obsModel.get_prior_dict()
    if len(adict.keys()) == 0:
        return None
    scipy.io.savemat(outpath, adict, oned_as='row')


def create_best_link(hardmatfile, linkmatfile):
    ''' Creates file linkmatfile that links to or copies hardmatfile.

    On Unix platforms, we save space by doing symbolic links.
    On Windows platforms, we simply copy the first file into the second.

    Parameters
    -----
    hardmatfile : str
    linkmatfile : str
    '''
    if os.path.islink(linkmatfile):
        os.unlink(linkmatfile)
    if os.path.exists(linkmatfile):
        os.remove(linkmatfile)
    if os.path.exists(hardmatfile):
        # Symlink support varies across Windows, so hard copy instead
        # Possible alternative is win32file.CreateSymbolicLink()
        if platform.startswith('win32') or platform.startswith('cygwin'):
            copy2(hardmatfile, linkmatfile)
        else:
            os.symlink(hardmatfile, linkmatfile)


def saveTopicModel(hmodel, SS, fpath, prefix, doLinkBest=False,
                   sparseEPS=0.002):
    ''' Write TopicModel to .mat formatted file on disk.

    Post Condition
    ------
    Topic model info written to file at location
        fpath/prefixTopicModel.mat
    '''
    EstPDict = dict()

    # Active comp probabilities
    if hasattr(hmodel.allocModel, 'rho'):
        EstPDict['rho'] = hmodel.allocModel.rho
        EstPDict['omega'] = hmodel.allocModel.omega
    EstPDict['probs'] = np.asarray(hmodel.allocModel.get_active_comp_probs(),
                                   dtype=np.float32)
    if hasattr(hmodel.allocModel, 'alpha'):
        EstPDict['alpha'] = hmodel.allocModel.alpha
    if hasattr(hmodel.allocModel, 'gamma'):
        EstPDict['gamma'] = hmodel.allocModel.gamma
    lamPrior = hmodel.obsModel.Prior.lam
    if np.allclose(lamPrior, lamPrior[0]):
        lamPrior = lamPrior[0]
    EstPDict['lam'] = np.asarray(lamPrior, dtype=np.float32)

    EstPDict['K'] = hmodel.obsModel.K
    EstPDict['vocab_size'] = hmodel.obsModel.D
    if SS is not None:
        if hasattr(SS, 'nDoc'):
            EstPDict['nDoc'] = SS.nDoc
        EstPDict['nTotalToken'] = np.sum(SS.WordCounts, axis=1)

    # Obsmodel parameters
    if str(type(hmodel.obsModel)).count('Mult') and SS is not None:
        SparseWordCounts = np.asarray(SS.WordCounts, dtype=np.float32)
        SparseWordCounts[SparseWordCounts < sparseEPS] = 0
        SparseWordCounts = scipy.sparse.csr_matrix(SparseWordCounts)
        EstPDict['SparseWordCount_data'] = SparseWordCounts.data
        EstPDict['SparseWordCount_indices'] = SparseWordCounts.indices
        EstPDict['SparseWordCount_indptr'] = SparseWordCounts.indptr
    else:
        # Temporary point estimate of topic-by-word matrix
        # TODO: handle EM case where these estimates already exist
        hmodel.obsModel.setEstParamsFromPost(hmodel.obsModel.Post)
        EstPDict['topics'] = hmodel.obsModel.EstParams.phi
        delattr(hmodel.obsModel, 'EstParams')

    outmatfile = os.path.join(fpath, prefix + 'TopicModel')
    scipy.io.savemat(outmatfile, EstPDict, oned_as='row')
