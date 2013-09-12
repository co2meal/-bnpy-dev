import numpy as np
import scipy.io
import os
import inspect

from bnpy.allocmodel import *
from bnpy.obsmodel import *
from bnpy.distr import *
from bnpy.HModel import HModel

GDict = globals()

def load_model( matfilepath, prefix='Best'):
  '''
  '''
  obsModel = load_obs_model(matfilepath, prefix)
  allocModel = load_alloc_model(matfilepath, prefix)
  return HModel(allocModel, obsModel)
  
def load_alloc_model( matfilepath, prefix):
  APDict = load_dict_from_matfile( os.path.join(matfilepath,'AllocPrior.mat'))
  ADict = load_dict_from_matfile( os.path.join(matfilepath,prefix+'AllocModel.mat'))
  AllocConstr = GDict[ADict['name']]
  amodel = AllocConstr( ADict['inferType'], APDict )
  amodel.from_dict( ADict)
  return amodel
  
def load_obs_model( matfilepath, prefix='Best'):
  obspriormatfile = os.path.join(matfilepath,'ObsPrior.mat')
  PDict = load_dict_from_matfile(obspriormatfile)
  if PDict['name'] == 'NoneType':
    obsPrior = None
  else:
    PriorConstr = GDict[PDict['name']]
    obsPrior = PriorConstr( **PDict)
  
  ODict = load_dict_from_matfile( os.path.join(matfilepath,prefix+'ObsModel.mat'))
  ObsConstr = GDict[ODict['name']]
  CompDicts = get_list_of_comp_dicts( ODict['K'], ODict)
  return ObsConstr.InitFromCompDicts( ODict, obsPrior, CompDicts)
  

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
      if x.ndim == 1 and x.size > 1:
        MyList[k][key] = x[k].copy()
      elif x.ndim == 2:
        MyList[k][key] = x[:,k].copy()
      elif x.ndim == 3:
        MyList[k][key] = x[:,:,k].copy()
  return MyList
  
def load_dict_from_matfile(matfilepath):
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
