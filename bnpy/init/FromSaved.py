'''
FromSaved.py

Initialize params of a bnpy model from a previous result saved to disk.
'''
import numpy as np
from bnpy.ioutil import ModelReader

def init_global_params(hmodel, Data, initname=None, prefix='Best', **kwargs):
  ''' Initialize (in-place) the global params of the given hmodel
      by copying the global parameters of a previously saved hmodel

      This does NOT transfer prior parameter values, only the global parameters

      Args
      -------
      hmodel : bnpy model object to initialize
      Data   : bnpy Data object whose dimensions must match resulting hmodel
      initname : valid filesystem path to stored result 
                 called "initname" because init from disk is one of many cmd line options.
  '''
  storedModel = ModelReader.load_model(initname, prefix)
  if storedModel.obsModel.D != Data.dim:
    raise ValueError("Stored model's output dimension does not match provided data!")
  aTypesMatch = type(storedModel.allocModel) == type(hmodel.allocModel)
  oTypesMatch = type(storedModel.obsModel) == type(hmodel.obsModel)
  inferTypesMatch = storedModel.inferType == hmodel.inferType

  LP = storedModel.calc_local_params(Data)
  SS = hmodel.get_global_suff_stats(Data, LP)
  hmodel.update_global_params(SS)
  # TODO: if aTypesMatch,oTypesMatch,inferTypesMatch all agree, should just copy params directly?
  
