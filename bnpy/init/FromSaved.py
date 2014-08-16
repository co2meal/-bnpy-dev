'''
FromSaved.py

Initialize params of a bnpy model from a previous result saved to disk.
'''
import numpy as np
import scipy.io
import os
from bnpy.ioutil import ModelReader

def init_global_params(hmodel, Data, initname=None, **kwargs):
  ''' Initialize (in-place) the global params of the given hmodel
      by copying the global parameters of a previously saved hmodel

      Only global parameters are modified.
      This does NOT alter settings of hmodel's prior distribution.

      Args
      -------
      hmodel : bnpy model object to initialize
      Data   : bnpy Data object whose dimensions must match resulting hmodel
      initname : valid filesystem path to stored result 
            
      Returns
      -------
      None. hmodel modified in-place.
  '''
  if os.path.isdir(initname):
    try:
      init_global_params_from_bnpy_format(hmodel, Data, initname, **kwargs)
    except:
      initname2 = os.path.join(initname, str(kwargs['taskid']))
      init_global_params_from_bnpy_format(hmodel, Data, initname2, **kwargs)
  elif initname.count('.mat') > 0:
    # Handle external external formats (not bnpy models) saved as MAT file
    MatDict = scipy.io.loadmat(initname)
    hmodel.set_global_params(**MatDict)  
  else:
    raise ValueError('Unrecognized init file: %s' % (initname))


def init_global_params_from_bnpy_format(hmodel, Data, initname, 
                                        initLapFrac=-1,
                                        prefix='Best', **kwargs):
  if initLapFrac > -1:
    storedModel, lap = ModelReader.loadModelForLap(initname, initLapFrac)
  else:
    storedModel = ModelReader.load_model(initname, prefix)
  try:
    hmodel.set_global_params(hmodel=storedModel)
  except AttributeError:
    LP = storedModel.calc_local_params(Data)
    SS = hmodel.get_global_suff_stats(Data, LP)
    hmodel.update_global_params(SS)
  
