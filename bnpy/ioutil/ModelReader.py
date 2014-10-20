'''
ModelReader.py

Read in a bnpy model from disk

Related
-------
ModelWriter.py
'''
import numpy as np
import scipy.io
import os
import glob

from ModelWriter import makePrefixForLap
from bnpy.allocmodel import *
from bnpy.obsmodel import *
from bnpy.util import as1D

GDict = globals()

def getPrefixForLapQuery(taskpath, lapQuery):
  ''' Search among the saved lap params in taskpath for the lap nearest query.

      Returns
      --------
      prefix : string like 'Lap0001.000' that indicates lap for saved params.
  '''
  try:
    saveLaps = np.loadtxt(os.path.join(taskpath,'laps-saved-params.txt'))
  except IOError:
    fileList = glob.glob(os.path.join(taskpath, 'Lap*TopicModel.mat'))
    saveLaps = list()
    for fpath in sorted(fileList):
      basename = fpath.split(os.path.sep)[-1]
      lapstr = basename[3:11]
      saveLaps.append(float(lapstr))
    saveLaps = np.sort(np.asarray(saveLaps))
  if lapQuery is None:
    bestLap = saveLaps[-1] # take final saved value
  else:
    distances = np.abs(lapQuery - saveLaps)
    bestLap = saveLaps[np.argmin(distances)]
  return makePrefixForLap(bestLap), bestLap

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
    data = Mdict['SparseWordCount_data']
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
      topics = Mdict['topics'] + 0
    probs = Mdict['probs']
    try:
      alpha = float(Mdict['alpha'])
    except KeyError:
      if 'alpha' in os.environ:
        alpha = float(os.environ['alpha'])
      else:
        raise ValueError('Unknown parameter alpha')
    return topics, probs, alpha

  infAlg = 'VB'
  amodel = HDPDir(infAlg, dict(alpha=Mdict['alpha'], gamma=Mdict['gamma']))
  omodel = MultObsModel(infAlg, **Mdict)
  hmodel = HModel(amodel, omodel)
  hmodel.set_global_params(**Mdict)
  if returnWordCounts:
    return hmodel, Mdict['WordCounts']
  return hmodel

def loadModelForLap(matfilepath, lapQuery):
  ''' Loads saved model with lap closest to provided lapQuery
      Returns
      -------
      model, true-lap-id
  '''
  prefix, bestLap = getPrefixForLapQuery(matfilepath, lapQuery)
  try:
    model = load_model(matfilepath, prefix=prefix)
  except IOError:
    model = loadTopicModel(matfilepath, prefix=prefix)
  return model, bestLap

def load_model( matfilepath, prefix='Best'):
  ''' Load model stored to disk by ModelWriter
  '''
  # avoids circular import
  import bnpy.HModel as HModel
  obsModel = load_obs_model(matfilepath, prefix)
  allocModel = load_alloc_model(matfilepath, prefix)
  return HModel(allocModel, obsModel)
  
def load_alloc_model(matfilepath, prefix):
  apriorpath = os.path.join(matfilepath,'AllocPrior.mat')
  amodelpath = os.path.join(matfilepath,prefix+'AllocModel.mat')
  APDict = loadDictFromMatfile(apriorpath)
  ADict = loadDictFromMatfile(amodelpath)
  AllocConstr = GDict[ADict['name']]
  amodel = AllocConstr( ADict['inferType'], APDict )
  amodel.from_dict( ADict)
  return amodel
  
def load_obs_model(matfilepath, prefix):
  obspriormatfile = os.path.join(matfilepath,'ObsPrior.mat')
  PriorDict = loadDictFromMatfile(obspriormatfile)
  ObsConstr = GDict[PriorDict['name']]
  obsModel = ObsConstr(**PriorDict)

  obsmodelpath = os.path.join(matfilepath,prefix+'ObsModel.mat')
  ParamDict = loadDictFromMatfile(obsmodelpath)
  if obsModel.inferType == 'EM':
    obsModel.setEstParams(**ParamDict)
  else:
    obsModel.setPostFactors(**ParamDict)
  return obsModel
  
def get_list_of_comp_dicts( K, Dict ):
  ''' We store all component params stacked together in an array.
      This function extracts them into individual components.
  '''
  MyList = [ dict() for k in xrange(K)]
  for k in xrange(K):
    for key in Dict:
      if type(Dict[key]) is not np.ndarray:
        continue
      x = Dict[key]
      if K == 1 and (key != 'min_covar' and key != 'K'):
        MyList[k][key] = x.copy()
      elif x.ndim == 1 and x.size > 1:
        MyList[k][key] = x[k].copy()
      elif x.ndim == 2:
        MyList[k][key] = x[:,k].copy()
      elif x.ndim == 3:
        MyList[k][key] = x[:,:,k].copy()
  return MyList
  
def loadDictFromMatfile(matfilepath):
  ''' Returns
      --------
       dict D where all numpy entries have good byte order, flags, etc.
  '''
  Dtmp = scipy.io.loadmat( matfilepath )
  D = dict( [x for x in Dtmp.items() if not x[0].startswith('__')] )
  for key in D:
    if type( D[key] ) is not np.ndarray:
      continue
    x = D[key]
    if x.ndim == 1:
      x = x[0]
    elif x.ndim == 2:
      x = np.squeeze(x)
    D[key] = x.newbyteorder('=').copy()
  return D
