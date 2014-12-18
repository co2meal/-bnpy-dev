'''
'''
import numpy as np
import scipy.io
import os
from distutils.dir_util import mkpath
from shutil import copy2
from sys import platform

def makePrefixForLap(lap):
  return 'Lap%08.3f' % (lap)

def saveTopicModel(hmodel, SS, fpath, prefix, doLinkBest=False,
                   sparseEPS=0.002):
  ''' Write TopicModel to mat file persistently

      Returns
      -------
      None. MAT file created on disk.
  '''
  EstPDict = dict()

  ## Active comp probabilities
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

  ## Obsmodel parameters
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

def save_model(hmodel, fname, prefix, doSavePriorInfo=True,
               doSaveObsModel=True, doLinkBest=False):
  ''' saves HModel object to mat file persistently
      
      Args
      --------
      hmodel: HModel to save
      fname: absolute full path of directory to save in
      prefix: prefix for file name, like 'Iter00004' or 'Best'
      doSavePriorInfo: whether to save prior info
  '''
  if not os.path.exists( fname):
    mkpath( fname )
  save_alloc_model(hmodel.allocModel, fname, prefix, doLinkBest=doLinkBest )

  if doSaveObsModel:
    save_obs_model(hmodel.obsModel, fname, prefix, doLinkBest=doLinkBest )

  if doSavePriorInfo:
    save_alloc_prior(hmodel.allocModel, fname)
    save_obs_prior(hmodel.obsModel, fname)
    
def save_alloc_model(amodel, fpath, prefix, doLinkBest=False):
  amatname = prefix + 'AllocModel.mat'
  outmatfile = os.path.join( fpath, amatname )
  adict = amodel.to_dict()
  adict.update( amodel.to_dict_essential() )
  scipy.io.savemat( outmatfile, adict, oned_as='row')
  if doLinkBest and prefix != 'Best':
    create_best_link( outmatfile, os.path.join(fpath,'BestAllocModel.mat'))
          
def save_alloc_prior( amodel, fpath):
  outpath = os.path.join( fpath, 'AllocPrior.mat')
  adict = amodel.get_prior_dict()
  if len( adict.keys() ) == 0:
    return None
  scipy.io.savemat( outpath, adict, oned_as='row')


def save_obs_model(obsmodel, fpath, prefix, doLinkBest=False):  
  amatname = prefix + 'ObsModel.mat'
  outmatfile = os.path.join( fpath, amatname )
  myDict = obsmodel.to_dict()
  scipy.io.savemat(outmatfile, myDict, oned_as='row')
  if doLinkBest and prefix != 'Best':
    create_best_link(outmatfile, os.path.join(fpath,'BestObsModel.mat'))

def save_obs_prior( obsModel, fpath):
  outpath = os.path.join( fpath, 'ObsPrior.mat')
  adict = obsModel.get_prior_dict()
  if len( adict.keys() ) == 0:
    return None
  scipy.io.savemat( outpath, adict, oned_as='row')

def create_best_link( hardmatfile, linkmatfile):
  ''' Creates a symlink file named linkmatfile that points to hardmatfile,
      where both are full valid absolute file system paths 
  '''
  if os.path.islink( linkmatfile):
    os.unlink( linkmatfile )
  if os.path.exists(linkmatfile):
    os.remove(linkmatfile)
  if os.path.exists( hardmatfile ):
      # symlink support varies across Windows releases, so hard copy instead
      # possible alternative is win32file.CreateSymbolicLink()
      if platform.startswith('win32'):
          copy2( hardmatfile, linkmatfile )
      else:
          os.symlink( hardmatfile, linkmatfile )
