'''
pepperHMT.py

A module to create HMT data from black and white pepper image
'''
import numpy as np
from bnpy.data import GroupXData
from bnpy.distr.GaussDistr import GaussDistr
from bnpy.allocmodel.tree import HMTUtil
from scipy.io import loadmat

path = '/Users/mertterzihan/Desktop/pepperimd256.mat'

def get_data(seed=8675309, nObsTotal=25000, **kwargs):
  d = loadmat(path)
  graph = np.zeros(d['graph'].shape)
  graph[:] = d['graph']
  doc_range = np.zeros(d['doc_range'].shape)
  doc_range[:] = d['doc_range']
  return GroupXData(graph, doc_range)

def get_data_info():
  return 'HMT data produced from black and white pepper image'

def get_short_name( ):
  ''' Return short string used in filepaths to store solutions
  '''
  return 'pepperHMT'