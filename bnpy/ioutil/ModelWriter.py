'''
'''
import numpy as np
import scipy.io
import os
from distutils.dir_util import mkpath

def save_model( hmodel, fname, prefix, doSavePriorInfo=True):
  ''' saves HModel to mat file persistently
      Parameters
      --------
      hmodel: HModel to save
      fname: directory to save in
      prefix: 'Iter00004' or 'Best'
      doSavePriorInfo: whether to save prior info
  '''
  if not os.path.exists( fname):
    mkpath( fname )
  save_alloc_model( hmodel.allocModel, fname, prefix )
  save_obs_model( hmodel.obsModel, fname, prefix )
  if doSavePriorInfo:
    save_alloc_prior( hmodel.allocModel, fname)
    save_obs_prior( hmodel.obsModel, fname)
    
def save_alloc_model( amodel, fpath, prefix ):
  amatname = prefix + 'AllocModel.mat'
  outmatfile = os.path.join( fpath, amatname )
  adict = amodel.to_dict()
  adict.update( amodel.to_dict_essential() )
  scipy.io.savemat( outmatfile, adict, oned_as='row')
  create_best_link( outmatfile, os.path.join(fpath,'BestAllocModel.mat'))
          
def save_obs_model( obsmodel, fpath, prefix ):
  '''
  '''
  amatname = prefix + 'ObsModel.mat'
  outmatfile = os.path.join( fpath, amatname )
  compList = list()
  for k in xrange( obsmodel.K ):
    compList.append( obsmodel.comp[k].to_dict() )
  myDict = obsmodel.to_dict_essential()
  for key in compList[0].keys():
    myDict[key] = np.squeeze(np.dstack([ compDict[key] for compDict in compList]))
  scipy.io.savemat( outmatfile, myDict, oned_as='row')
  create_best_link( outmatfile, os.path.join(fpath,'BestObsModel.mat'))
  
def save_alloc_prior( amodel, fpath):
  outpath = os.path.join( fpath, 'AllocPrior.mat')
  adict = amodel.get_prior_dict()
  if len( adict.keys() ) == 0:
    return None
  scipy.io.savemat( outpath, adict, oned_as='row')

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
    os.symlink( hardmatfile, linkmatfile )

'''        
def save_model_types( hmodel, fname ):
  atype = type(hmodel.allocModel).__name__
  otype = type(hmodel.obsModel).__name__ 
  dtype = type(hmodel.obsModel.obsPrior).__name__
  with open( os.path.join(fname, 'AllocModelType.txt'), 'w') as f:
    f.write( atype)
  with open( os.path.join(fname, 'ObsModelType.txt'),'w') as f:
    f.write( otype)
  if dtype is not None:
    with open( os.path.join(fname, 'ObsDistrType.txt'),'w') as f:
      f.write( dtype) 
'''